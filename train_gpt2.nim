#[
This file trains the GPT-2 model.
This version is the clean, minimal, reference. As such:
- it runs on CPU.
- it does not make the code too complex; it is readable.
- it does not use any processor-specific instructions, intrinsics and such.
- it _does_ use a few OpenMP pragmas because this is a large speedup at very low cost
There will be other versions of this code that specialize it and make it fast.
]#

# ----------------------------------------------------------------------------
# all the individual layers' forward and backward passes
import std / [math, strformat, monotimes, os]
from std / times import inMilliseconds

## Helper to fuse loops on the Nim side (for `parallel for collapse` equivalence)
import fuse_loops

when defined(danger):
  {.passC: "-ffast-math".} # fast math *seems* to be fine here (i.e. not affected by
                           # https://github.com/karpathy/llm.c/issues/19 )

when defined(openmp): ## If we use `openmp` add the required
  {.passC: "-fopenmp".}
  {.passL: "-lgomp".}

## Helper buffer view type
type
  MView[T] = ptr UncheckedArray[T]

proc `+!`(x: pointer, idx: int | int32): pointer =
  cast[pointer](cast[uint](x) + uint(idx))
proc `{}`[T](b: MView[T], idx: int | int32): MView[T] =
  MView[T](cast[ptr UncheckedArray[T]](b[idx].addr))
proc toMView[T](p: pointer): MView[T] = MView[T](cast[ptr UncheckedArray[T]](p))

proc encoder_forward(outp: MView[float32],
                     inp: MView[int32], wte: MView[float32], wpe: MView[float32],
                     B: int32, T: int32, C: int32) =
  for b in 0 ..< B:
    for t in 0 ..< T:
      # seek to the output position in out[b,t,:]
      let out_bt = outp{b * T * C + t * C}
      # get the index of the token at inp[b, t]
      let ix = inp[b * T + t]
      # seek to the position in wte corresponding to the token
      let wte_ix = wte{ix * C}
      # seek to the position in wpe corresponding to the position
      let wpe_t = wpe{t * C}
      # add the two vectors and store the result in outp[b,t,:]
      for i in 0 ..< C:
        out_bt[i] = wte_ix[i] + wpe_t[i]

proc encoder_backward(dwte: MView[float32], dwpe: MView[float32],
                      dout: MView[float32], inp: MView[int32],
                      B: int32, T: int32, C: int32) =
  for b in 0 ..< B:
    for t in 0 ..< T:
      let dout_bt = dout{b * T * C + t * C}
      let ix = inp[b * T + t]
      let dwte_ix = dwte{ix * C}
      let dwpe_t = dwpe{t * C}
      for i in 0 ..< C:
        let d = dout_bt[i]
        dwte_ix[i] += d
        dwpe_t[i] += d

proc layernorm_forward(outp: MView[float32], mean: MView[float32], rstd: MView[float32],
                       inp: MView[float32], weight: MView[float32], bias: MView[float32],
                       B: int32, T: int32, C: int32) =
  let eps = 1e-5f
  for b in 0 ..< B:
    for t in 0 ..< T:
      # seek to the input position inp[b,t,:]
      let x = inp{b * T * C + t * C}
      # calculate the mean
      var m = 0.0'f32
      for i in 0 ..< C:
        m += x[i]
      m = m/C.float32
      # calculate the variance (without any bias correction)
      var v = 0.0'f32
      for i in 0 ..< C:
        let xshift = x[i] - m
        v += xshift * xshift
      v = v/C.float32
      # calculate the rstd
      let s = 1.0'f32 / sqrt(v + eps)
      # seek to the output position in out[b,t,:]
      let out_bt = outp{b * T * C + t * C}
      for i in 0 ..< C:
        let n = (s * (x[i] - m)) # normalized output
        let o = n * weight[i] + bias[i] # scale and shift it
        out_bt[i] = o # write
      # cache the mean and rstd for the backward pass later
      mean[b * T + t] = m
      rstd[b * T + t] = s

proc layernorm_backward(dinp: MView[float32], dweight: MView[float32], dbias: MView[float32],
                        dout: MView[float32], inp: MView[float32], weight: MView[float32], mean: MView[float32], rstd: MView[float32],
                        B: int32, T: int32, C: int32) =
  for b in 0 ..< B:
    for t in 0 ..< T:
      let dout_bt = dout{b * T * C + t * C}
      let inp_bt = inp{b * T * C + t * C}
      let dinp_bt = dinp{b * T * C + t * C}
      let mean_bt = mean[b * T + t]
      let rstd_bt = rstd[b * T + t]

      # first: two reduce operations
      var dnorm_mean = 0.0f
      var dnorm_norm_mean = 0.0f
      for i in 0 ..< C:
        let norm_bti = (inp_bt[i] - mean_bt) * rstd_bt
        let dnorm_i = weight[i] * dout_bt[i]
        dnorm_mean += dnorm_i
        dnorm_norm_mean += dnorm_i * norm_bti
      dnorm_mean = dnorm_mean / C.float32
      dnorm_norm_mean = dnorm_norm_mean / C.float32

      # now iterate again and accumulate all the gradients
      for i in 0 ..< C:
        let norm_bti = (inp_bt[i] - mean_bt) * rstd_bt
        let dnorm_i = weight[i] * dout_bt[i]
        # gradient contribution to bias
        dbias[i] += dout_bt[i]
        # gradient contribution to weight
        dweight[i] += norm_bti * dout_bt[i]
        # gradient contribution to input
        var dval = 0.0'f32
        dval += dnorm_i # term 1
        dval -= dnorm_mean # term 2
        dval -= norm_bti * dnorm_norm_mean # term 3
        dval *= rstd_bt # final scale
        dinp_bt[i] += dval

proc matmul_forward(outp: MView[float32],
                    inp: MView[float32], weight: MView[float32], bias: MView[float32],
                    B: int32, T: int32, C: int32, OC: int32) =
  # most of the running time is spent here and in matmul_backward
  # OC is short for "output channels"
  # inp is (B,T,C), weight is (OC, C), bias is (OC)
  # outp will be (B,T,OC)
  fuseLoops("parallel for"):
    for b in 0 ..< B:
      for t in 0 ..< T:
        let out_bt = outp{b * T * OC + t * OC}
        let inp_bt = inp{b * T * C + t * C}
        for o in nofuse(0 ..< OC):
          var val = if bias != nil: bias[o] else: 0.0'f32
          let wrow = weight{o*C}
          for i in nofuse(0 ..< C):
            val += inp_bt[i] * wrow[i]
          out_bt[o] = val

proc matmul_backward(dinp: MView[float32], dweight: MView[float32], dbias: MView[float32],
                     dout: MView[float32], inp: MView[float32], weight: MView[float32],
                     B: int32, T: int32, C: int32, OC: int32) =
  # most of the running time is spent here and in matmul_forward
  # this backward could be done in a single "round" of loops
  # but that doesn't afford an efficient parallelization strategy

  # backward into inp first, parallelize over B,T
  fuseLoops("parallel for"):
    for b in 0 ..< B:
      for t in 0 ..< T:
        let dout_bt = dout{b * T * OC + t * OC}
        let dinp_bt = dinp{b * T * C + t * C}
        for o in nofuse(0 ..< OC):
          let wrow = weight{o*C}
          let d = dout_bt[o]
          for i in 0 ..< C:
            dinp_bt[i] += wrow[i] * d
  # backward into weight/bias, parallelize over output channels OC
  for o in `||`(0, OC, "parallel for"):
    for b in 0 ..< B:
      for t in 0 ..< T:
        let dout_bt = dout{b * T * OC + t * OC}
        let inp_bt = inp{b * T * C + t * C}
        let dwrow = dweight{o*C}
        let d = dout_bt[o]
        if dbias != nil: dbias[o] += d
        for i in 0 ..< C:
          dwrow[i] += inp_bt[i] * d

proc attention_forward(outp: MView[float32], preatt: MView[float32], att: MView[float32],
                       inp: MView[float32],
                       B: int32, T: int32, C: int32, NH: int32) =
  # input is (B, T, 3C) Q,K,V
  # preatt, att are (B, NH, T, T)
  # output is (B, T, C)
  let C3 = C*3
  let hs = C div NH # head size
  let scale = 1.0'f32 / sqrt(hs.float32)

  fuseLoops("parallel for"):
    for b in 0 ..< B:
      for t in 0 ..< T:
        for h in 0 ..< NH:
          let query_t = inp{b * T * C3 + t * C3 + h * hs}
          let preatt_bth = preatt{b*NH*T*T + h*T*T + t*T}
          let att_bth = att{b*NH*T*T + h*T*T + t*T}

          # pass 1: calculate query dot key and maxval
          var maxval = -10000.0'f32 # TODO something better
          for t2 in nofuse(0 .. t):
            let key_t2 = inp{b * T * C3 + t2 * C3 + h * hs + C} # +C because it's key

            # (query_t) dot (key_t2)
            var val = 0.0'f32
            for i in 0 ..< hs:
              val += query_t[i] * key_t2[i]
            val *= scale
            if val > maxval:
              maxval = val

            preatt_bth[t2] = val

          # pass 2: calculate the exp and keep track of sum
          var expsum = 0.0'f32
          for t2 in nofuse(0 .. t):
            let expv = exp(preatt_bth[t2] - maxval)
            expsum += expv
            att_bth[t2] = expv
          let expsum_inv = if expsum == 0.0'f32: 0.0'f32 else: 1.0'f32 / expsum

          # pass 3: normalize to get the softmax
          for t2 in nofuse(0 ..< T):
            if t2 <= t:
                att_bth[t2] *= expsum_inv
            else:
                # causal attention mask. not strictly necessary to set to zero here
                # only doing this explicitly for debugging and checking to PyTorch
                att_bth[t2] = 0.0f


          # pass 4: accumulate weighted values into the output of attention
          let out_bth = outp{b * T * C + t * C + h * hs}
          for i in nofuse(0 ..< hs):
            out_bth[i] = 0.0f
          for t2 in nofuse(0 ..< t):
            let value_t2 = inp{b * T * C3 + t2 * C3 + h * hs + C*2} # +C*2 because it's value
            let att_btht2 = att_bth[t2]
            for i in 0 ..< hs:
              out_bth[i] += att_btht2 * value_t2[i]

proc attention_backward(dinp: MView[float32], dpreatt: MView[float32], datt: MView[float32],
                        dout: MView[float32], inp: MView[float32], att: MView[float32],
                        B: int32, T: int32, C: int32, NH: int32) =
  # inp/dinp are (B, T, 3C) Q,K,V
  # att/datt/dpreatt are (B, NH, T, T)
  # dout is (B, T, C)
  let C3 = C*3
  let hs = C div NH # head size
  let scale = 1.0'f32 / sqrt(hs.float32)

  for b in 0 ..< B:
    for t in 0 ..< T:
      for h in 0 ..< NH:
        let att_bth = att{b*NH*T*T + h*T*T + t*T}
        let datt_bth = datt{b*NH*T*T + h*T*T + t*T}
        let dpreatt_bth = dpreatt{b*NH*T*T + h*T*T + t*T}
        let dquery_t = dinp{b * T * C3 + t * C3 + h * hs}
        let query_t = inp{b * T * C3 + t * C3 + h * hs}

        # backward pass 4, through the value accumulation
        let dout_bth = dout{b * T * C + t * C + h * hs}
        for t2 in 0 .. t:
          let value_t2 = inp{b * T * C3 + t2 * C3 + h * hs + C*2} # +C*2 because it's value
          let dvalue_t2 = dinp{b * T * C3 + t2 * C3 + h * hs + C*2}
          for i in 0 ..< hs:
            # in the forward pass this was:
            # out_bth[i] += att_bth[t2] * value_t2[i]
            # so now we have:
            datt_bth[t2] += value_t2[i] * dout_bth[i]
            dvalue_t2[i] += att_bth[t2] * dout_bth[i]

        # backward pass 2 & 3, the softmax
        # note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
        for t2 in 0 .. t:
          for t3 in 0 .. t:
            let indicator = if t2 == t3: 1.0'f32 else: 0.0'f32
            let local_derivative = att_bth[t2] * (indicator - att_bth[t3])
            dpreatt_bth[t3] += local_derivative * datt_bth[t2]

        # backward pass 1, the query @ key matmul
        for t2 in 0 .. t:
          let key_t2 = inp{b * T * C3 + t2 * C3 + h * hs + C} # +C because it's key
          let dkey_t2 = dinp{b * T * C3 + t2 * C3 + h * hs + C} # +C because it's key
          for i in 0 ..< hs:
            # in the forward pass this was:
            # preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale
            # so now we have:
            dquery_t[i] += key_t2[i] * dpreatt_bth[t2] * scale
            dkey_t2[i] += query_t[i] * dpreatt_bth[t2] * scale

proc gelu_forward(outp: MView[float32], inp: MView[float32], N: int32) =
  let s = sqrt(2.0'f32 / PI)
  for i in 0 ..< N:
    let x = inp[i]
    let cube = 0.044715'f32 * x * x * x
    outp[i] = 0.5f * x * (1.0f + tanh(s * (x + cube)))

proc gelu_backward(dinp: MView[float32], inp: MView[float32], dout: MView[float32], N: int32) =
  const s = sqrt(2.0'f32 / PI)
  for i in 0 ..< N:
    let x = inp[i]
    let cube = 0.044715'f32 * x * x * x
    let tanh_arg = s * (x + cube)
    let tanh_out = tanh(tanh_arg)
    let coshf_out = cosh(tanh_arg)
    let sech_out = 1.0'f32 / (coshf_out * coshf_out)
    let local_grad = 0.5'f32 * (1.0f + tanh_out) + x * 0.5f * sech_out * s * (1.0f + 3.0f * 0.044715f * x * x)
    dinp[i] += local_grad * dout[i]

proc residual_forward(outp, inp1, inp2: MView[float32], N: int32) =
  for i in 0 ..< N:
    outp[i] = inp1[i] + inp2[i]

proc residual_backward(dinp1, dinp2, dout: MView[float32], N: int32) =
  for i in 0 ..< N:
    dinp1[i] += dout[i]
    dinp2[i] += dout[i]

proc softmax_forward(probs: MView[float32], logits: MView[float32], B: int32, T: int32, V: int32) =
  # output: probs are (B,T,V) of the probabilities
  # input: logits is (B,T,V) of the unnormalized log probabilities
  fuseLoops("parallel for"):
    for b in 0 ..< B:
      for t in 0 ..< T:
        # probs <- softmax(logits)
        let logits_bt = logits{b * T * V + t * V}
        let probs_bt = probs{b * T * V + t * V}

        var maxval = -10000.0'f32 # TODO something better
        for i in nofuse(0 ..< V):
          if logits_bt[i] > maxval:
              maxval = logits_bt[i]

        var sum = 0.0'f32
        for i in nofuse(0 ..< V):
          probs_bt[i] = exp(logits_bt[i] - maxval)
          sum += probs_bt[i]
        for i in nofuse(0 ..< V):
          probs_bt[i] /= sum

proc crossentropy_forward(losses: MView[float32],
                          probs: MView[float32], targets: MView[int32],
                          B: int32, T: int32, V: int32) =
  # output: losses is (B,T) of the individual losses at each position
  # input: probs are (B,T,V) of the probabilities
  # input: targets is (B,T) of integers giving the correct index in logits
  for b in 0 ..< B:
    for t in 0 ..< T:
      # loss = -log(probs[target])
      let probs_bt = probs{b * T * V + t * V}
      let ix = targets[b * T + t]
      losses[b * T + t] = -ln(probs_bt[ix]) # `logf` is `ln`

proc crossentropy_softmax_backward(dlogits: MView[float32],
                           dlosses: MView[float32], probs: MView[float32], targets: MView[int32],
                           B: int32, T: int32, V: int32) =
  # backwards through both softmax and crossentropy
  for b in 0 ..< B:
    for t in 0 ..< T:
      let dlogits_bt = dlogits{b * T * V + t * V}
      let probs_bt = probs{b * T * V + t * V}
      let dloss = dlosses[b * T + t]
      let ix = targets[b * T + t]
      for i in 0 ..< V:
        let p = probs_bt[i]
        let indicator = if i == ix: 1.0'f32 else: 0.0'f32
        dlogits_bt[i] += (p - indicator) * dloss

# ----------------------------------------------------------------------------
# GPT-2 model definition

# the parameters of the model
const NUM_PARAMETER_TENSORS = 16
type
  ## NOTE: The order of the fields is *important*. We use `fieldPairs`
  ## together with the size of each buffer (`param_sizes`) to assign them
  ## to a single allocated buffer where each `MView[float32]` points to its correct
  ## starting location in memory.
  ParameterTensors = object
    wte:      MView[float32] # (V, C)
    wpe:      MView[float32] # (maxT, C)
    ln1w:     MView[float32] # (L, C)
    ln1b:     MView[float32] # (L, C)
    qkvw:     MView[float32] # (L, 3*C, C)
    qkvb:     MView[float32] # (L, 3*C)
    attprojw: MView[float32] # (L, C, C)
    attprojb: MView[float32] # (L, C)
    ln2w:     MView[float32] # (L, C)
    ln2b:     MView[float32] # (L, C)
    fcw:      MView[float32] # (L, 4*C, C)
    fcb:      MView[float32] # (L, 4*C)
    fcprojw:  MView[float32] # (L, C, 4*C)
    fcprojb:  MView[float32] # (L, C)
    lnfw:     MView[float32] # (C)
    lnfb:     MView[float32] # (C)

# allocate memory for the parameters and point the individual tensors to the right places
proc malloc_and_point[T](arg: var T, sizes: openArray[csize_t]): MView[float32] =
  let num = sizes.sum
  # malloc all data all at once
  let memory = toMView[float32](alloc_shared0(num.int * sizeof(float32)))
  # assign all the tensors
  # We use `fieldPairs` to walk all object fields and assign their buffers directly
  # based on the known size of each previous buffer
  var i = 0
  var offset = 0
  for field, val in fieldPairs(arg):
    cast[var pointer](val.addr) = memory{offset}
    offset += sizes[i].int
    inc i
  result = memory

const NUM_ACTIVATION_TENSORS = 23
type
  ## NOTE: Again, the order of the fields is *important*!
  ActivationTensors = object
    encoded:   MView[float32] # (B, T, C)
    ln1:       MView[float32] # (L, B, T, C)
    ln1_mean:  MView[float32] # (L, B, T)
    ln1_rstd:  MView[float32] # (L, B, T)
    qkv:       MView[float32] # (L, B, T, 3*C)
    atty:      MView[float32] # (L, B, T, C)
    preatt:    MView[float32] # (L, B, NH, T, T)
    att:       MView[float32] # (L, B, NH, T, T)
    attproj:   MView[float32] # (L, B, T, C)
    residual2: MView[float32] # (L, B, T, C)
    ln2:       MView[float32] # (L, B, T, C)
    ln2_mean:  MView[float32] # (L, B, T)
    ln2_rstd:  MView[float32] # (L, B, T)
    fch:       MView[float32] # (L, B, T, 4*C)
    fch_gelu:  MView[float32] # (L, B, T, 4*C)
    fcproj:    MView[float32] # (L, B, T, C)
    residual3: MView[float32] # (L, B, T, C)
    lnf:       MView[float32] # (B, T, C)
    lnf_mean:  MView[float32] # (B, T)
    lnf_rstd:  MView[float32] # (B, T)
    logits:    MView[float32] # (B, T, V)
    probs:     MView[float32] # (B, T, V)
    losses:    MView[float32] # (B, T)

type
  GPT2Config = object
    max_seq_len: int32 # max sequence length, e.g. 1024
    vocab_size: int32 # vocab size, e.g. 50257
    num_layers: int32 # number of layers, e.g. 12
    num_heads: int32 # number of heads in attention, e.g. 12
    channels: int32 # number of channels, e.g. 768

  GPT2 = object
    config: GPT2Config
    # the weights of the model, and their sizes
    params: ParameterTensors
    param_sizes: array[NUM_PARAMETER_TENSORS, csize_t]
    params_memory: MView[float32]
    num_parameters: int32
    # gradients of the weights
    grads: ParameterTensors
    grads_memory: MView[float32]
    # buffers for the AdamW optimizer
    m_memory: MView[float32]
    v_memory: MView[float32]
    # the activations of the model, and their sizes
    acts: ActivationTensors
    act_sizes: array[NUM_ACTIVATION_TENSORS, csize_t]
    acts_memory: MView[float32]
    num_activations: int32
    # gradients of the activations
    grads_acts: ActivationTensors
    grads_acts_memory: MView[float32]
    # other run state configuration
    batch_size: int32 # the batch size (B) of current forward pass
    seq_len: int32 # the sequence length (T) of current forward pass
    inputs: MView[int32] # the input tokens for the current forward pass
    targets: MView[int32] # the target tokens for the current forward pass
    mean_loss: float32 # after a forward pass with targets, will be populated with the mean loss

proc `=copy`(a: var GPT2, b: GPT2) {.error: "GPT2 cannot be copied.".}

proc `=destroy`(model: GPT2) =
  dealloc_shared(model.params_memory)
  dealloc_shared(model.grads_memory)
  dealloc_shared(model.m_memory)
  dealloc_shared(model.v_memory)
  dealloc_shared(model.acts_memory)
  dealloc_shared(model.grads_acts_memory)
  dealloc_shared(model.inputs)
  dealloc_shared(model.targets)

proc gpt2_build_from_checkpoint(model: var GPT2, checkpoint_path: string) =

  # read in model from a checkpoint file
  var model_file = open(checkpoint_path, fmRead)
  if model_file == nil: echo "Error opening model file"; quit(1)
  var model_header: array[256, int32]

  ## XXX: check that this is correct!
  discard model_file.readBuffer(cast[pointer](model_header.addr), sizeof(int32) * 256)
  #read(model_header, sizeof(int32), 256, model_file)
  if model_header[0] != 20240326: echo "Bad magic model file"; quit(1)
  if model_header[1] != 1: echo "Bad version in model file"; quit(1)

  # read in hyperparameters
  var maxT, V, L, NH, C: int
  template asgn(x,y,z): untyped =
    x = z
    y = z
  asgn model.config.max_seq_len, maxT, model_header[2]
  asgn model.config.vocab_size, V, model_header[3]
  asgn model.config.num_layers, L, model_header[4]
  asgn model.config.num_heads, NH, model_header[5]
  asgn model.config.channels, C, model_header[6]
  echo "[GPT-2]"
  echo "max_seq_len: ", maxT
  echo "vocab_size: ", V
  echo "num_layers: ", L
  echo "num_heads: ", NH
  echo "channels: ", C

  # allocate space for all the parameters and read them in
  model.param_sizes[0] = (V * C).csize_t
  model.param_sizes[1] = (maxT * C).csize_t
  model.param_sizes[2] = (L * C).csize_t
  model.param_sizes[3] = (L * C).csize_t
  model.param_sizes[4] = (L * (3 * C) * C).csize_t
  model.param_sizes[5] = (L * (3 * C)).csize_t
  model.param_sizes[6] = (L * C * C).csize_t
  model.param_sizes[7] = (L * C).csize_t
  model.param_sizes[8] = (L * C).csize_t
  model.param_sizes[9] = (L * C).csize_t
  model.param_sizes[10] = (L * (4 * C) * C).csize_t
  model.param_sizes[11] = (L * (4 * C)).csize_t
  model.param_sizes[12] = (L * C * (4 * C)).csize_t
  model.param_sizes[13] = (L * C).csize_t
  model.param_sizes[14] = (C).csize_t
  model.param_sizes[15] = (C).csize_t

  # cound the number of paramaters
  let num_parameters = model.param_sizes.sum
  echo "num_parameters: ", num_parameters
  model.num_parameters = num_parameters.int32

  # read in all the parameters from file
  model.params_memory = malloc_and_point(model.params, model.param_sizes)
  ## XXX: CHECK THIS TOO
  discard model_file.readBuffer(cast[pointer](model.params_memory), sizeof(float32) * num_parameters.int)
  #read(model.params_memory, sizeof(float32), num_parameters, model_file)
  close(model_file)

  # other inits
  model.acts_memory = nil
  model.grads_memory = nil
  model.m_memory = nil
  model.v_memory = nil
  model.grads_acts_memory = nil
  model.inputs = nil
  model.targets = nil
  model.batch_size = 0
  model.seq_len = 0
  model.mean_loss = -1.0f # -1.0f will designate no loss

proc gpt2_forward(model: var GPT2, inputs: MView[int32], targets: MView[int32], B: int32, T: int32) =
  # targets are optional and could be nil

  # ensure the model was initialized or error out
  if model.params_memory == nil:
    echo "Error: model was not initialized properly."
    quit(1)
  # convenience parameters
  let V = model.config.vocab_size
  let L = model.config.num_layers
  let NH = model.config.num_heads
  let C = model.config.channels

  # allocate space for all the activations if needed (done here, lazily)
  if model.acts_memory == nil:
    # record the current B,T as well
    model.batch_size = B
    model.seq_len = T
    # and now allocate the space
    model.act_sizes[0] = (B * T * C).csize_t
    model.act_sizes[1] = (L * B * T * C).csize_t
    model.act_sizes[2] = (L * B * T).csize_t
    model.act_sizes[3] = (L * B * T).csize_t
    model.act_sizes[4] = (L * B * T * 3*C).csize_t
    model.act_sizes[5] = (L * B * T * C).csize_t
    model.act_sizes[6] = (L * B * NH * T * T).csize_t
    model.act_sizes[7] = (L * B * NH * T * T).csize_t
    model.act_sizes[8] = (L * B * T * C).csize_t
    model.act_sizes[9] = (L * B * T * C).csize_t
    model.act_sizes[10] = (L * B * T * C).csize_t
    model.act_sizes[11] = (L * B * T).csize_t
    model.act_sizes[12] = (L * B * T).csize_t
    model.act_sizes[13] = (L * B * T * 4*C).csize_t
    model.act_sizes[14] = (L * B * T * 4*C).csize_t
    model.act_sizes[15] = (L * B * T * C).csize_t
    model.act_sizes[16] = (L * B * T * C).csize_t
    model.act_sizes[17] = (B * T * C).csize_t
    model.act_sizes[18] = (B * T).csize_t
    model.act_sizes[19] = (B * T).csize_t
    model.act_sizes[20] = (B * T * V).csize_t
    model.act_sizes[21] = (B * T * V).csize_t
    model.act_sizes[22] = (B * T).csize_t

    let num_activations = model.act_sizes.sum
    echo "num_activations: ", num_activations
    model.num_activations = num_activations.int32
    model.acts_memory = malloc_and_point(model.acts, model.act_sizes)
    # also create memory for caching inputs and targets
    model.inputs = toMView[int32](alloc_shared0(B * T * sizeof(int32)))
    model.targets = toMView[int32](alloc_shared0(B * T * sizeof(int32))) # might be unused if we never have targets but it's small
  else:
    # validate B,T is no larger than what was previously allocated
    # in principle, we could re-allocate a larger chunk of memory, for now we just error out
    if B > model.batch_size or T > model.seq_len:
      echo "Error: batch size or sequence length is inadequately large"
      echo &"Model: B={model.batch_size} T={model.seq_len}, Desired: B={B} T={T}"
      quit(1)

  # cache the inputs/targets
  copyMem(model.inputs, inputs, B * T * sizeof(int32))
  if targets != nil:
    copyMem(model.targets, targets, B * T * sizeof(int32))

  # forward pass
  let params = model.params # for brevity
  let acts = model.acts
  var residual: MView[float32]
  encoder_forward(acts.encoded, inputs, params.wte, params.wpe, B, T, C) # encoding goes into residual[0]
  for l in 0 ..< L:
    residual = if l == 0: acts.encoded else: acts.residual3{(l-1) * B * T * C}

    # get the pointers of the weights for this layer
    let l_ln1w = params.ln1w{l * C}
    let l_ln1b = params.ln1b{l * C}
    let l_qkvw = params.qkvw{l * 3*C * C}
    let l_qkvb = params.qkvb{l * 3*C}
    let l_attprojw = params.attprojw{l * C * C}
    let l_attprojb = params.attprojb{l * C}
    let l_ln2w = params.ln2w{l * C}
    let l_ln2b = params.ln2b{l * C}
    let l_fcw = params.fcw{l * 4*C * C}
    let l_fcb = params.fcb{l * 4*C}
    let l_fcprojw = params.fcprojw{l * C * 4*C}
    let l_fcprojb = params.fcprojb{l * C}

    # get the pointers of the activations for this layer
    let l_ln1 = acts.ln1{l * B * T * C}
    let l_ln1_mean = acts.ln1_mean{l * B * T}
    let l_ln1_rstd = acts.ln1_rstd{l * B * T}
    let l_qkv = acts.qkv{l * B * T * 3*C}
    let l_atty = acts.atty{l * B * T * C}
    let l_preatt = acts.preatt{l * B * NH * T * T}
    let l_att = acts.att{l * B * NH * T * T}
    let l_attproj = acts.attproj{l * B * T * C}
    let l_residual2 = acts.residual2{l * B * T * C}
    let l_ln2 = acts.ln2{l * B * T * C}
    let l_ln2_mean = acts.ln2_mean{l * B * T}
    let l_ln2_rstd = acts.ln2_rstd{l * B * T}
    let l_fch = acts.fch{l * B * T * 4*C}
    let l_fch_gelu = acts.fch_gelu{l * B * T * 4*C}
    let l_fcproj = acts.fcproj{l * B * T * C}
    let l_residual3 = acts.residual3{l * B * T * C}

    # now do the forward pass
    layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C)
    matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C)
    attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH)
    matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C)
    residual_forward(l_residual2, residual, l_attproj, B*T*C)
    layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C)
    matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C)
    gelu_forward(l_fch_gelu, l_fch, B*T*4*C)
    matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C)
    residual_forward(l_residual3, l_residual2, l_fcproj, B*T*C)
  residual = acts.residual3{(L-1) * B * T * C} # last residual is in residual3
  layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C)
  matmul_forward(acts.logits, acts.lnf, params.wte, nil, B, T, C, V)
  softmax_forward(acts.probs, acts.logits, B, T, V)

  # also forward the cross-entropy loss function if we have the targets
  if targets != nil:
    crossentropy_forward(model.acts.losses, model.acts.probs, targets, B, T, V)
    # for convenience also evaluate the mean loss
    var mean_loss = 0.0'f32
    for i in 0 ..< B*T: mean_loss += model.acts.losses[i]
    mean_loss /= (B*T).float32
    model.mean_loss = mean_loss
  else:
    # if we don't have targets, we don't have a loss
    model.mean_loss = -1.0f

import system / memory
proc gpt2_zero_grad(model: var GPT2) =
  if model.grads_memory != nil:
    nimSetMem(model.grads_memory, 0, model.num_parameters * sizeof(float32))
  if model.grads_acts_memory != nil:
    nimSetMem(model.grads_acts_memory, 0, model.num_activations * sizeof(float32))

proc gpt2_backward(model: var GPT2) =
  # double check we forwarded previously, with targets
  if model.mean_loss == -1.0f:
    echo "Error: must forward with targets before backward"
    quit(1)

  # lazily allocate the memory for gradients of the weights and activations, if needed
  if model.grads_memory == nil:
    model.grads_memory = malloc_and_point(model.grads, model.param_sizes)
    model.grads_acts_memory = malloc_and_point(model.grads_acts, model.act_sizes)
    gpt2_zero_grad(model)

  # convenience shortcuts
  let B = model.batch_size
  let T = model.seq_len
  let V = model.config.vocab_size
  let L = model.config.num_layers
  let NH = model.config.num_heads
  let C = model.config.channels

  # backward pass
  let params = model.params # for brevity
  let grads = model.grads
  let acts = model.acts
  let grads_acts = model.grads_acts

  # we kick off the chain by filling in dlosses with 1.0f/(B*T), to get the mean loss
  let dloss_mean = 1.0'f32 / (B*T).float32
  for i in 0 ..< B*T: grads_acts.losses[i] = dloss_mean

  crossentropy_softmax_backward(grads_acts.logits, grads_acts.losses, acts.probs, model.targets, B, T, V)
  matmul_backward(grads_acts.lnf, grads.wte, nil, grads_acts.logits, acts.lnf, params.wte, B, T, C, V)
  var residual = acts.residual3{(L-1) * B * T * C} # last layer's residual
  var dresidual = grads_acts.residual3{(L-1) * B * T * C} # write to last layer's residual
  layernorm_backward(dresidual, grads.lnfw, grads.lnfb, grads_acts.lnf, residual, params.lnfw, acts.lnf_mean, acts.lnf_rstd, B, T, C)

  for l in countdown(L-1, 0):
    residual =  if l == 0: acts.encoded else: acts.residual3{(l-1) * B * T * C}
    dresidual = if l == 0: grads_acts.encoded else: grads_acts.residual3{(l-1) * B * T * C}

    # get the pointers of the weights for this layer
    let l_ln1w = params.ln1w{l * C}
    let l_qkvw = params.qkvw{l * 3*C * C}
    let l_attprojw = params.attprojw{l * C * C}
    let l_ln2w = params.ln2w{l * C}
    let l_fcw = params.fcw{l * 4*C * C}
    let l_fcprojw = params.fcprojw{l * C * 4*C}
    # get the pointers of the gradients of the weights for this layer
    let dl_ln1w = grads.ln1w{l * C}
    let dl_ln1b = grads.ln1b{l * C}
    let dl_qkvw = grads.qkvw{l * 3*C * C}
    let dl_qkvb = grads.qkvb{l * 3*C}
    let dl_attprojw = grads.attprojw{l * C * C}
    let dl_attprojb = grads.attprojb{l * C}
    let dl_ln2w = grads.ln2w{l * C}
    let dl_ln2b = grads.ln2b{l * C}
    let dl_fcw = grads.fcw{l * 4*C * C}
    let dl_fcb = grads.fcb{l * 4*C}
    let dl_fcprojw = grads.fcprojw{l * C * 4*C}
    let dl_fcprojb = grads.fcprojb{l * C}
    # get the pointers of the activations for this layer
    let l_ln1 = acts.ln1{l * B * T * C}
    let l_ln1_mean = acts.ln1_mean{l * B * T}
    let l_ln1_rstd = acts.ln1_rstd{l * B * T}
    let l_qkv = acts.qkv{l * B * T * 3*C}
    let l_atty = acts.atty{l * B * T * C}
    let l_att = acts.att{l * B * NH * T * T}
    let l_residual2 = acts.residual2{l * B * T * C}
    let l_ln2 = acts.ln2{l * B * T * C}
    let l_ln2_mean = acts.ln2_mean{l * B * T}
    let l_ln2_rstd = acts.ln2_rstd{l * B * T}
    let l_fch = acts.fch{l * B * T * 4*C}
    let l_fch_gelu = acts.fch_gelu{l * B * T * 4*C}
    # get the pointers of the gradients of the activations for this layer
    let dl_ln1 = grads_acts.ln1{l * B * T * C}
    let dl_qkv = grads_acts.qkv{l * B * T * 3*C}
    let dl_atty = grads_acts.atty{l * B * T * C}
    let dl_preatt = grads_acts.preatt{l * B * NH * T * T}
    let dl_att = grads_acts.att{l * B * NH * T * T}
    let dl_attproj = grads_acts.attproj{l * B * T * C}
    let dl_residual2 = grads_acts.residual2{l * B * T * C}
    let dl_ln2 = grads_acts.ln2{l * B * T * C}
    let dl_fch = grads_acts.fch{l * B * T * 4*C}
    let dl_fch_gelu = grads_acts.fch_gelu{l * B * T * 4*C}
    let dl_fcproj = grads_acts.fcproj{l * B * T * C}
    let dl_residual3 = grads_acts.residual3{l * B * T * C}

    # backprop this layer
    residual_backward(dl_residual2, dl_fcproj, dl_residual3, B*T*C)
    matmul_backward(dl_fch_gelu, dl_fcprojw, dl_fcprojb, dl_fcproj, l_fch_gelu, l_fcprojw, B, T, 4*C, C)
    gelu_backward(dl_fch, l_fch, dl_fch_gelu, B*T*4*C)
    matmul_backward(dl_ln2, dl_fcw, dl_fcb, dl_fch, l_ln2, l_fcw, B, T, C, 4*C)
    layernorm_backward(dl_residual2, dl_ln2w, dl_ln2b, dl_ln2, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C)
    residual_backward(dresidual, dl_attproj, dl_residual2, B*T*C)
    matmul_backward(dl_atty, dl_attprojw, dl_attprojb, dl_attproj, l_atty, l_attprojw, B, T, C, C)
    attention_backward(dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T, C, NH)
    matmul_backward(dl_ln1, dl_qkvw, dl_qkvb, dl_qkv, l_ln1, l_qkvw, B, T, C, 3*C)
    layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C)
  encoder_backward(grads.wte, grads.wpe, grads_acts.encoded, model.inputs, B, T, C)


proc gpt2_update(model: var GPT2, learning_rate, beta1, beta2, eps, weight_decay: float32, t: int32) =
  # reference: https:#pytorch.org/docs/stable/generated/torch.optim.AdamW.html

  # lazily allocate the memory for m_memory and v_memory
  if model.m_memory == nil:
    model.m_memory = toMView[float32](alloc_shared0(model.num_parameters.int * sizeof(float32)))
    model.v_memory = toMView[float32](alloc_shared0(model.num_parameters.int * sizeof(float32)))

  for i in 0 ..< model.num_parameters:
    let param = model.params_memory[i]
    let grad = model.grads_memory[i]

    # update the first moment (momentum)
    let m = beta1 * model.m_memory[i] + (1.0f - beta1) * grad
    # update the second moment (RMSprop)
    let v = beta2 * model.v_memory[i] + (1.0f - beta2) * grad * grad
    # bias-correct both moments
    let m_hat = m / (1.0f - pow(beta1, t.float32))
    let v_hat = v / (1.0f - pow(beta2, t.float32))

    # update
    model.m_memory[i] = m
    model.v_memory[i] = v
    model.params_memory[i] -= learning_rate * (m_hat / (sqrt(v_hat) + eps) + weight_decay * param)

# ----------------------------------------------------------------------------
# data loader lite
# returns random batches of data from a file of integers

type
  DataLoader = object
    # hyperparameters
    B: int32
    T: int32
    # input handling and its state
    tokens_file: File
    file_size: clong
    current_position: clong
    # output memory
    batch: MView[int32]
    inputs: MView[int32]
    targets: MView[int32]
    # convenience variables
    num_batches: int

proc `=copy`(a: var DataLoader, b: DataLoader) {.error: "DataLoader cannot be copied.".}
proc `=destroy`(loader: DataLoader) =
  close(loader.tokens_file)
  dealloc_shared(loader.batch)

proc init(_: typedesc[DataLoader], filename: string, B: int32, T: int32): DataLoader =
  result.B = B
  result.T = T

  # open the input file for reading
  result.tokens_file = open(filename, fmRead)
  if result.tokens_file == nil:
    echo "Error opening tokens file"
    quit(1)

  # determine the file size
  setFilePos(result.tokens_file, 0, fspEnd)
  result.file_size = getFilePos(result.tokens_file)
  setFilePos(result.tokens_file, 0, fspSet)
  if result.file_size < (B * T + 1) * sizeof(int32):
    echo "Error: file size is too small for the batch size and sequence length"
    quit(1)
  result.current_position = 0 # start at the beginning

  # allocate space for B*T + 1 integers to store the inputs and targets
  result.batch = toMView[int32](alloc_shared0((B * T + 1) * sizeof(int32)))
  result.inputs = result.batch
  result.targets = result.batch{1} # targets are shifted by one
  result.num_batches = result.file_size div (B * T * sizeof(int32))

proc reset(loader: var DataLoader) =
  loader.current_position = 0

proc next_batch(loader: var DataLoader) =
  let B = loader.B
  let T = loader.T
  # if we are at the end of the file, loop back to the beginning
  if loader.current_position + (B*T+1) * sizeof(int32) > loader.file_size:
      loader.current_position = 0
  # read the B*T+1 integers from the file into batch
  setFilePos(loader.tokens_file, loader.current_position, fspSet)
  discard loader.tokens_file.readBuffer(loader.batch, sizeof(int32) * B*T+1)
  # advance the current position by B*T integers
  loader.current_position += B*T * sizeof(int32)

# ----------------------------------------------------------------------------
# sampler

const GPT2_EOT = 50256

proc random_u32(state: var uint64): uint32 =
  # xorshift rng: https:#en.wikipedia.org/wiki/Xorshift#xorshift.2A
  state = state xor (state shr 12)
  state = state xor (state shl 25)
  state = state xor (state shr 27)
  result = ((state * 0x2545F4914F6CDD1D'u64) shr 32).uint32

proc random_f32(state: var uint64): float32 = # random float32 in [0,1)
  result = (random_u32(state) shr 8).float32 / 16777216.0f

proc sample_mult(probabilities: MView[float32], n: int32, coin: float32): int32 =
  # sample index from probabilities (they must sum to 1!)
  # coin is a random number in [0, 1), usually from random_f32()
  var cdf = 0.0'f32
  for i in 0 ..< n:
    cdf += probabilities[i]
    if coin < cdf:
      return i
  result = n - 1 # in case of rounding errors

# ----------------------------------------------------------------------------
# main training loop
proc main() =
  # build the GPT-2 model from a checkpoint
  var model: GPT2
  gpt2_build_from_checkpoint(model, "gpt2_124M.bin")

  # build the DataLoaders from tokens files. for now use tiny_shakespeare if available, else tiny_stories
  const tiny_stories_train = "data/TinyStories_train.bin"
  const tiny_stories_val = "data/TinyStories_val.bin"
  const tiny_shakespeare_train = "data/tiny_shakespeare_train.bin"
  const tiny_shakespeare_val = "data/tiny_shakespeare_val.bin"
  let train_tokens = if fileExists(tiny_shakespeare_train): tiny_shakespeare_train else: tiny_stories_train
  let val_tokens = if fileExists(tiny_shakespeare_val): tiny_shakespeare_val else: tiny_stories_val
  let B = 4'i32
  let T = 64'i32
  var train_loader = DataLoader.init(train_tokens, B, T)
  echo "train dataset num_batches: ", train_loader.num_batches
  var val_loader = DataLoader.init(val_tokens, B, T)
  echo "val dataset num_batches: ", val_loader.num_batches
  const val_num_batches = 10

  # some memory for generating samples from the model
  var rng_state = 1337'u64
  const gen_max_length = 64
  var gen_tokens: array[gen_max_length, int32]

  # train
  for step in 0 .. 40:
    # once in a while estimate the validation loss
    if step mod 10 == 0:
      var val_loss = 0.0'f32
      reset(val_loader)
      for i in 0 ..< val_num_batches:
          next_batch(val_loader)
          gpt2_forward(model, val_loader.inputs, val_loader.targets, B, T)
          val_loss += model.mean_loss
      val_loss /= val_num_batches
      echo "val loss ", val_loss

    # once in a while do model inference to print generated text
    if step > 0 and step mod 20 == 0:
      gen_tokens[0] = GPT2_EOT # the GPT-2 EOT token kicks off the generation
      for t in 1 ..< gen_max_length:
        # note that inference is wasteful here because
        # for each t, we re-compute all activations between 0 and t
        # leaving this alone because you want separate code for inference anyway
        # the inference here is just for sanity checking purposes
        gpt2_forward(model, toMView[int32](gen_tokens.addr), nil, 1, t.int32)
        let probs = model.acts.probs{(t-1) * model.config.vocab_size}
        let coin = random_f32(rng_state)
        let next_token = sample_mult(probs, model.config.vocab_size, coin)
        gen_tokens[t] = next_token
      stdout.write "generated: "
      for t in 0 ..< gen_max_length:
          stdout.write gen_tokens[t], " "
      echo ""

    # do a training step
    let start = getMonoTime()
    next_batch(train_loader)
    gpt2_forward(model, train_loader.inputs, train_loader.targets, B, T)
    gpt2_zero_grad(model)
    gpt2_backward(model)
    gpt2_update(model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, (step+1).int32)
    let stop = getMonoTime()
    let time_elapsed = stop - start
    echo &"step {step}: train loss {model.mean_loss} (took {time_elapsed.inMilliseconds} ms)"

main()

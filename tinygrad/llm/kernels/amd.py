from __future__ import annotations
import functools, math
from typing import Callable, cast
from tinygrad import Tensor, UOp, nn, Device, Context
from tinygrad.device import Buffer
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.helpers import prod, getenv
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, resolve

BLOCK_M, BLOCK_N, WARP_SIZE = 32, 32, 32
WMMA_M, WMMA_N, WMMA_K = 16, 16, 16
WAVES_M, WAVES_N, LANES_PER_WAVE_M, LANES_PER_WAVE_N = 2, 2, 2, 16
WMMA_ACC, THREADS_PER_BLOCK = WMMA_M // LANES_PER_WAVE_M, WARP_SIZE * WAVES_M * WAVES_N
LDS_PAD, WMMA_ARG, LOG2E = 4, ((WMMA_M, WMMA_N, WMMA_K), 'AMD', 32), math.log2(math.e)
Q4_K, Q5_K, Q6_K, IQ4_XS, GGML_BLOCK_SIZE, Q8_GROUP_SIZE, Q4_WORDS, Q5_WORDS, Q6_BYTES, IQ4_WORDS = 12, 13, 14, 23, 256, 32, 36, 44, 210, 34
Q6_PADDED, Q6_WORDS = 212, 53  # the 210-byte Q6 blocks are padded to 212 bytes so they are word-addressable
QUANT_SIZES = {Q4_K: Q4_WORDS*4, Q5_K: Q5_WORDS*4, Q6_K: Q6_BYTES, IQ4_XS: IQ4_WORDS*4}  # bytes per 256-weight block
AMD_PACKED_U32X4_LOAD = "__builtin_nontemporal_load((const unsigned_int4*){0})"
AMD_FMA_TO_F16 = "__builtin_amdgcn_fma_mixlo_f16({}, {}, {})"
AMD_PACKED_F16_MUL_TO_F16 = "__builtin_amdgcn_fma_mixlo_f16_packed({}, {}, {})"

def kernel_var(x:UOp) -> UOp:
  # a Variable is a 0-d ALU BUFFER in the tensor graph; inside kernels it takes the ALU PARAM form (same name keeps the value binding)
  return x.substitute({v: UOp.variable(v.expr, v.vmin, v.vmax, dtype=v.dtype, multiple_of=v.arg.multiple_of, param=True)
                       for v in x.toposort() if v.is_variable})

def _unbind(v:int|UOp) -> int|UOp: return kernel_var(v.unbind_all()[0]) if isinstance(v, UOp) else v

@functools.cache
def amd_custom_kernels_supported(device:str|tuple[str, ...]|None) -> bool:
  # the custom kernels are tuned for RDNA3 (gfx11): the WMMA register layouts don't match gfx12 (RDNA4)
  # or CDNA (MFMA-only, wave64), and the dp4a builtins and 32-lane wave ops aren't portable either.
  if isinstance(device, tuple): device = device[0]
  if device is None or device.split(":")[0] != "AMD": return False
  # @function contexts set ALLOW_DEVICE_USAGE=0 (scheduling must not open devices); the device is always open here
  with Context(ALLOW_DEVICE_USAGE=1):
    amd_device = Device[device]
    # The LLVM IR renderer cannot lower the lane-level CUSTOM operations used by these kernels.
    return (t:=getattr(amd_device, "target", None)) is not None and t[0] == 11 and amd_device.renderer.__class__.__name__ != "AMDLLVMRenderer"

@functools.cache
def amd_direct_isa(device:str|tuple[str, ...]|None) -> bool:
  if isinstance(device, tuple): device = device[0]
  if device is None or device.split(":")[0] != "AMD": return False
  with Context(ALLOW_DEVICE_USAGE=1): return Device[device].renderer.__class__.__name__ == "AMDRenderer"

def _swizzle_f32(val:UOp, offset:int) -> UOp:
  return UOp(Ops.CUSTOM, src=(val,), arg=
    (f"__builtin_bit_cast(float, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, {{0}}), {0x1f | offset<<10}))", dtypes.float))

def _peer16_f32(val:UOp) -> UOp:
  """Lane i ← lane i^16 via VALU permlanex16 (no ds_swizzle / lgkm wait)."""
  return UOp(Ops.CUSTOM, src=(val,), arg=(
    "__builtin_bit_cast(float, __builtin_amdgcn_permlanex16(__builtin_bit_cast(int, {0}), "
    "__builtin_bit_cast(int, {0}), 0x76543210, 0xfedcba98, true, false))", dtypes.float))

def warp_reduce(val:UOp, maximum:bool=False, full_wave:bool=False) -> UOp:
  for offset in ((16, 8, 4, 2, 1) if full_wave else (8, 4, 2, 1)):
    if val.op is Ops.INDEX and val.addrspace == AddrSpace.REG: val = val.load()
    other = _peer16_f32(val) if offset == 16 else _swizzle_f32(val, offset)
    val = val.maximum(other) if maximum else val + other
  return val

_reg_swizzle_slot = 64

def warp_reduce_many(vals:list[UOp], maximum:bool=False, full_wave:bool=False) -> list[UOp]:
  """Butterfly-reduce independent f32s; park same-stage swizzles in REG temps.

  Per-element REG stores keep all stage-N ds_swizzle ops ahead of butterfly adds so
  emit can issue SWIZZLE×N then one lgkm wait (HIP pattern). Unique slots per stage
  avoid clobbering live parks across keys.

  Offset 16 uses VALU permlanex16 (no lgkm) when full_wave — same cross-half exchange
  as ds_swizzle 0x1f|16<<10, cheaper on the flash_decode score reduce.
  """
  if not vals: return []
  if len(vals) == 1: return [warp_reduce(vals[0], maximum=maximum, full_wave=full_wave)]
  global _reg_swizzle_slot
  offsets = (16, 8, 4, 2, 1) if full_wave else (8, 4, 2, 1)
  chunk = list(vals)
  for i, v in enumerate(chunk):
    if v.op is Ops.INDEX and v.addrspace == AddrSpace.REG: chunk[i] = v.load()
  no_park = bool(getenv("AMD_SWIZZLE_NO_PARK", 0))
  for offset in offsets:
    raw = [_peer16_f32(v) if offset == 16 else _swizzle_f32(v, offset) for v in chunk]
    if no_park:
      # HIP-like: ADD reads swizzle/permlane dests directly (no REG park MOVs).
      # `_schedule_swizzle_mov_batches` still batches SW×N,ADD×N before regalloc.
      chunk = [v.maximum(o) if maximum else v + o for v, o in zip(chunk, raw)]
      continue
    tmp = UOp.placeholder((len(chunk),), dtypes.float, slot=_reg_swizzle_slot, addrspace=AddrSpace.REG)
    _reg_swizzle_slot += 1
    tmp = tmp.after(UOp.group(*[tmp[i].store(sw) for i, sw in enumerate(raw)]))
    others = [tmp[i].load() for i in range(len(chunk))]
    chunk = [v.maximum(o) if maximum else v + o for v, o in zip(chunk, others)]
  return chunk

def _reg(shape:tuple[int, ...], slot:int, value:float, dep:UOp|None=None) -> UOp:
  ret = UOp.placeholder(shape, dtypes.float, slot=slot, addrspace=AddrSpace.REG)
  return ret.after((ret if dep is None else ret.after(dep)).store(ret.const_like(value)))

# ******** quant linear: q8-activation kernels over packed ggml weights (Q4_K/Q5_K/Q6_K/IQ4_XS) ********

class Linear(nn.Linear):
  ggml_type:int|None = None
  use_custom_quant = True
  def __init__(self, in_features:int, out_features:int, bias=True):
    super().__init__(in_features, out_features, bias)
    self.in_features, self.out_features = in_features, out_features
  def set_quantized(self, decoded:Tensor):
    packed_sizes = {decoded.numel() // 256 * type_size:typ for typ,type_size in QUANT_SIZES.items()}
    graph = decoded.uop.toposort()
    raw = next((u for u in graph if u.op is Ops.SHRINK and u.dtype == dtypes.uint8 and prod(u.shape) in packed_sizes), None)
    if raw is None: return
    ggml_type = packed_sizes[prod(raw.shape)]
    # the packed byte rate alone can't distinguish same-rate formats (Q4_0 vs Q4_K, Q5_0 vs Q5_K, MXFP4 vs IQ4_XS).
    # the supported formats are 256-wide superblocks: their decode views the packed bytes at the superblock width
    # (ggml_data_to_tensor reshapes to (-1, QUANT_SIZES[type])), while same-rate 32-wide formats reshape to 17-22
    if not any(u.op is Ops.RESHAPE and u.shape[-1:] == (QUANT_SIZES[ggml_type],) for u in graph): return
    raw_offset = raw.contiguous_view_offset()
    assert raw_offset is not None and raw_offset % 4 == 0 and raw.buf_uop.dtype == dtypes.uint8
    self.ggml_type = ggml_type
    # store a typed buffer view: a lazy BITCAST is decomposed into byte-combining ALU before custom-kernel
    # scheduling and would copy the entire packed weight on every JIT graph
    if self.ggml_type == Q6_K:
      # Q6 blocks are 210 bytes, so consecutive blocks are only 2-byte aligned. pad each block to 212 bytes
      # the kernel can do all its reads as aligned u32 words
      nbytes, nblocks = raw.max_numel(), raw.max_numel() // Q6_BYTES
      byte_view = Tensor(UOp.from_buffer(cast(Buffer, raw.buf_uop.buffer).view(nbytes, dtypes.uint8, raw_offset)))
      padded = byte_view.reshape((nblocks, Q6_BYTES)).pad_to((nblocks, Q6_PADDED)).bitcast(dtypes.uint32)
      self.weight = padded.contiguous().reshape(nblocks * Q6_WORDS)
    else:
      self.weight = Tensor(UOp.from_buffer(cast(Buffer, raw.buf_uop.buffer)
        .view(raw.max_numel() * raw.dtype.itemsize // dtypes.uint32.itemsize, dtypes.uint32, raw_offset)))
  def __call__(self, x:Tensor) -> Tensor:
    supported = self.use_custom_quant and amd_custom_kernels_supported(self.weight.device)
    if self.ggml_type is None and supported:
      self.set_quantized(self.weight)
      if self.ggml_type is None:
        # tiny dense fp16 matmul (e.g. the ssm beta/alpha head rows): single fp16 gemv kernel instead of a
        # generic matmul schedule, and realize the densely packed weight once if it is still a lazy ggml view
        if self.weight.dtype in (dtypes.half, dtypes.float, dtypes.bfloat16) and self.out_features <= 2048 \
          and self.in_features % (WARP_SIZE*4) == 0:
          numel, max_shape = x.numel(), x.max_shape
          if isinstance(numel, int) or prod(max_shape) // self.in_features <= 32:
            out = f16_gemv(self, x if isinstance(numel, int) else x.pad_to(max_shape))
            return out if isinstance(numel, int) else out.shrink(tuple((0, s) for s in (*x.shape[:-1], self.out_features)))
        self.use_custom_quant = supported = False  # not a supported quant format
    if self.ggml_type in (Q4_K, Q5_K, Q6_K, IQ4_XS) and supported:
      if isinstance(x.numel(), int): return q8_linear(self, x)
      # symbolic token count: pad to the max chunk size so the kernels see static shapes, garbage rows are sliced off
      out = q8_linear(self, x.pad_to(x.max_shape))
      return out.shrink(tuple((0, s) for s in (*x.shape[:-1], self.out_features)))
    return super().__call__(x)

def _amd_dp4a(a:UOp, b:UOp, c:UOp) -> UOp:
  return UOp(Ops.CUSTOMI, src=(a.int(), b.int(), c),
             arg=("__builtin_amdgcn_sudot4(true, {}, true, {}, {}, false)", dtypes.int32))

def _amd_byte_perm(a:UOp, b:UOp, selectors:UOp) -> UOp:
  return UOp(Ops.CUSTOMI, src=tuple(x.cast(dtypes.uint32) for x in (a, b, selectors)), arg=("__builtin_amdgcn_perm({}, {}, {})", dtypes.uint32))

def _amd_fma_to_f16(a:UOp, b:UOp, c:UOp) -> UOp:
  return UOp(Ops.CUSTOMI, src=(a, b, c), arg=(AMD_FMA_TO_F16, dtypes.float16))

def _amd_packed_f16_mul_to_f16(a:UOp, b:UOp, high:int) -> UOp:
  if a.op is Ops.INDEX: a = a.load()
  return UOp(Ops.CUSTOMI, src=(a, b, UOp.const(high, dtypes.uint32)), arg=(AMD_PACKED_F16_MUL_TO_F16, dtypes.float16))

def _amd_load(ptr:UOp, lanes:int|None=None, packed_u32:bool=False) -> UOp:
  assert ptr.op is Ops.INDEX
  # nontemporal scalar load: streamed weights must not evict the activations/KV cache from L2
  if lanes is None: return ptr.load(arg="nontemporal")
  buf, coords = ptr.src[0], ptr.src[1:]
  idx = sum((coord*math.prod(buf.shape[i+1:]) for i,coord in enumerate(coords)), UOp.const(0))
  shrink = UOp(Ops.SHRINK, src=(buf.flatten(), idx, UOp.const(lanes)))
  if packed_u32:
    assert lanes == 4 and ptr.dtype is dtypes.uint32
    return UOp(Ops.CUSTOM, src=(shrink,), arg=(AMD_PACKED_U32X4_LOAD, dtypes.uint32))
  return shrink.load()

def _load_byte(raw:UOp, base:UOp, offset:UOp) -> UOp: return (raw[base + offset//4] >> ((offset&3)*8).cast(dtypes.uint32)) & 255
def _half(value:UOp) -> UOp:
  if value.dtype is not dtypes.uint16: value = value.cast(dtypes.uint16)
  return value.bitcast(dtypes.float16).float()

def _iq4_bytes(packed:UOp, shift:int) -> UOp:
  # the non-linear iq4nl table as a byte lookup: 3 byte_perms beat any arithmetic/select-tree form (~60% decode)
  selectors = (packed >> shift) & 0x0f0f0f0f
  low = _amd_byte_perm(UOp.const(0xf6eaddcf, dtypes.uint32), UOp.const(0xbfad9881, dtypes.uint32), selectors)
  high = _amd_byte_perm(UOp.const(0x71594535, dtypes.uint32), UOp.const(0x26190d01, dtypes.uint32), selectors & 0x07070707)
  return _amd_byte_perm(high, low, 0x03020100 | ((selectors & 0x08080808) >> 1))

def _q5_scales(raw:UOp, base:UOp, subgroup:UOp, header:UOp|None=None) -> tuple[UOp, UOp, UOp, UOp]:
  # scales/mins (6-bit each) live in block bytes 4-15: three words total, same for the whole super-block's lanes
  w0, w1, w2, w3 = (tuple(header[i] for i in range(4)) if header is not None else
                     (_amd_load(raw[base]), _amd_load(raw[base+1]), _amd_load(raw[base+2]), _amd_load(raw[base+3])))
  sb = (subgroup & 3) * 8  # byte within word
  byte1, byte2, byte3 = (w1 >> sb) & 255, (w2 >> sb) & 255, (w3 >> sb) & 255
  scale = (subgroup < 4).where(byte1 & 63, (byte3 & 15) | ((byte1 >> 6) << 4))
  minimum = (subgroup < 4).where(byte2 & 63, (byte3 >> 4) | ((byte2 >> 6) << 4))
  d, dmin = (w0 & 0xffff).cast(dtypes.uint16), (w0 >> 16).cast(dtypes.uint16)
  return _half(d), _half(dmin), scale.float(), minimum.float()

def _iq4_scales(raw:UOp, base:UOp, subgroup:UOp) -> tuple[UOp, UOp]:
  low = _load_byte(raw, base, 4 + subgroup//2)
  scale = ((low >> (4*(subgroup%2)).cast(dtypes.uint32)) & 15) | ((((raw[base] >> 16) >> (2*subgroup).cast(dtypes.uint32)) & 3) << 4)
  return _half(raw[base] & 0xffff), (scale.cast(dtypes.uint8).bitcast(dtypes.int8)-32).float()

@functools.cache
def iq4_half_lut(device:str) -> Tensor:
  from tinygrad.runtime.autogen.ggml_common import kvalues_iq4nl
  return Tensor([x for j in range(16) for i in range(16) for x in (kvalues_iq4nl[i], kvalues_iq4nl[j])],
                dtype=dtypes.float16, device=device).bitcast(dtypes.uint32).contiguous()

@functools.cache
def _q8_quantize_kernel(q:UOp, scale:UOp, xsum:UOp, x:UOp, tokens:int, in_features:int) -> UOp:
  groups = in_features//Q8_GROUP_SIZE
  token_group, lane = UOp.range(tokens*groups, 0, axis_type=AxisType.GLOBAL), UOp.range(32, 1, axis_type=AxisType.LOCAL)
  token, group = token_group//groups, token_group%groups
  x = x.reshape(tokens, groups, 32)
  group_scale = (warp_reduce(x[token, group, lane].float().abs(), maximum=True, full_wave=True) / 127).maximum(1e-8)
  word_lane = lane.minimum(7)
  xs = tuple(x[token, group, word_lane*4+i].float() for i in range(4))
  qs = tuple((v/group_scale).round().clip(-127, 127).cast(dtypes.int8) for v in xs)
  word = sum((v.cast(dtypes.uint8).cast(dtypes.uint32) << (i*8) for i, v in enumerate(qs)), UOp.const(0, dtypes.uint32))
  # per-16 sums of the quantized values (lanes 0-3 / 4-7): Q4_K/Q5_K need the 32-sum, Q6_K the 16-sums
  part = (lane < 8).where(sum((v.cast(dtypes.int32) for v in qs), UOp.const(0, dtypes.int32)), UOp.const(0, dtypes.int32))
  gsum = [warp_reduce(((lane & 4).eq(h*4)).where(part, UOp.const(0, dtypes.int32)), full_wave=True) for h in range(2)]
  store_half = (lane & 4) >> 2
  stores = (q[token, group, lane.valid(lane < 8)].store(word),
            UOp.group(scale[token, group.valid(lane.eq(0))].store(group_scale),
                      xsum[token, group, store_half.valid(lane.eq(0) | lane.eq(4))].store(
                        store_half.eq(0).where(gsum[0].float(), gsum[1].float()))))
  return UOp.group(*stores).end(token_group, lane).sink(arg=KernelInfo(name="q8_quantize", opts_to_apply=()))

def q8_quantize(x:Tensor, tokens:int, in_features:int) -> tuple[Tensor, Tensor, Tensor]:
  groups = in_features//Q8_GROUP_SIZE
  q = Tensor.empty(tokens, groups, 8, dtype=dtypes.uint32, device=x.device)
  scale = Tensor.empty(tokens, groups, dtype=dtypes.float32, device=x.device)
  xsum = Tensor.empty(tokens, groups, 2, dtype=dtypes.float32, device=x.device)
  q, scale, xsum = Tensor.custom_kernel(q, scale, xsum, x, fxn=functools.partial(_q8_quantize_kernel, tokens=tokens, in_features=in_features))[:3]
  return q, scale, xsum

def _decode_linear(out:UOp, out_features:int, group_count:int, group_dot, name:str) -> UOp:
  chunks = out.shape[2]
  # two-dim global grid instead of one flat grid: no div/mods needed to decompose the gid
  token_output = UOp.range(out.shape[0]*out_features, 0, axis_type=AxisType.GLOBAL)
  chunk, lane = UOp.range(chunks, 1, axis_type=AxisType.GLOBAL), UOp.range(32, 2, axis_type=AxisType.LOCAL)
  token, output = token_output // out_features, token_output % out_features
  group = (lane+chunk*32).minimum(group_count-1)
  value = group_dot(token, output, group) if chunks*32 == group_count else \
    (lane+chunk*32 < group_count).where(group_dot(token, output, group), UOp.const(0, dtypes.float32))
  total = warp_reduce(value, full_wave=True)
  return out[token, output, chunk.valid(lane.eq(0))].store(total.cast(out.dtype)).end(token_output, chunk, lane).sink(
    arg=KernelInfo(name=name, opts_to_apply=()))

@functools.cache
def _quant_decode_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, xs:UOp, out_features:int, in_features:int, ggml_type:int) -> UOp:
  group_count = in_features // Q8_GROUP_SIZE
  def load_xwords(token:UOp, group:UOp) -> tuple[UOp, ...]:
    # two u32×4 loads → B128 under direct ISA (u32×8 still scalarizes)
    halves = (_amd_load(xq[token, group, 0], 4), _amd_load(xq[token, group, 4], 4))
    return tuple(halves[i // 4][i % 4] for i in range(8))
  def group_dot(token:UOp, output:UOp, group:UOp) -> UOp:
    block, subgroup = group // 8, group % 8
    if ggml_type in (Q4_K, Q5_K):
      base = (output * in_features//GGML_BLOCK_SIZE + block) * (Q4_WORDS if ggml_type == Q4_K else Q5_WORDS)
      qs_base, dot = base + (4 if ggml_type == Q4_K else 12) + (subgroup//2)*8, UOp.const(0, dtypes.int32)
      # vectorize the 8 packed-weight words and (for Q5_K) the 32-byte high-bit bitmap
      qs_pair = (_amd_load(raw[qs_base], 4), _amd_load(raw[qs_base+4], 4))
      if ggml_type == Q5_K: qh_pair = (_amd_load(raw[base+4], 4), _amd_load(raw[base+8], 4))
      xwords = load_xwords(token, group)
      for word_idx in range(8):
        word = (qs_pair[word_idx//4][word_idx%4] >> ((subgroup&1)*4).cast(dtypes.uint32)) & 0x0f0f0f0f
        if ggml_type == Q5_K: word |= ((qh_pair[word_idx//4][word_idx%4] >> subgroup.cast(dtypes.uint32)) & 0x01010101) << 4
        dot = _amd_dp4a(word, xwords[word_idx], dot)
      d, dmin, scale, minimum = _q5_scales(raw, base, subgroup)
      gsum = xs[token, group, 0].load() + xs[token, group, 1].load()
      return (dot.float()*d*scale - gsum*dmin*minimum) * xd[token, group]
    if ggml_type == IQ4_XS:
      base = (output * in_features//GGML_BLOCK_SIZE + block) * IQ4_WORDS
      dot = UOp.const(0, dtypes.int32)
      xwords = load_xwords(token, group)
      for word_idx in range(8):
        packed = _amd_load(raw[base + 2 + subgroup*4 + word_idx%4])
        dot = _amd_dp4a(_iq4_bytes(packed, 4*(word_idx//4)), xwords[word_idx], dot)
      d, scale = _iq4_scales(raw, base, subgroup)
      return dot.float() * xd[token, group] * d * scale
    # the packed rows were padded to 212 bytes (53 words) per 256-block in set_quantized: everything is word-aligned
    base = (output*in_features//GGML_BLOCK_SIZE+block)*Q6_WORDS
    # Issue packed-weight VMEM first so the decode scheduler can burst raw reads before activations/scales.
    lows = tuple(_amd_load(raw[base + (subgroup//4)*16 + (subgroup%2)*8 + half*4], 4) for half in range(2))
    highs = tuple(_amd_load(raw[base + 32 + (subgroup//4)*8 + half*4], 4) for half in range(2))
    xwords = load_xwords(token, group)
    dots = [UOp.const(0, dtypes.int32)] * 2
    for word_idx in range(8):
      within = (subgroup*32 + word_idx*4)%128
      low = lows[word_idx//4][word_idx%4] >> ((within//64)*4).cast(dtypes.uint32)
      high = highs[word_idx//4][word_idx%4] >> ((within//32)*2).cast(dtypes.uint32)
      # 4 values per word: (low nibble) | (2 high bits << 4). values stay positive, so the int8-bitcast/-32 of the
      # naive dequant is skipped and the -32 offset is applied later via the per-16 sums of the quantized inputs
      word = (low & 0x0f0f0f0f) | ((high & 0x03030303) << 4)
      dots[word_idx//4] = _amd_dp4a(word, xwords[word_idx], dots[word_idx//4])
    scales = [((raw[base + 48 + (subgroup*2+i)//4] >> (((subgroup*2+i)%4)*8).cast(dtypes.uint32)) & 255)
              .cast(dtypes.uint8).bitcast(dtypes.int8).float() for i in range(2)]
    gsum = [xs[token, group, i].load() * 32 for i in range(2)]
    return ((dots[0].float() - gsum[0])*scales[0] + (dots[1].float() - gsum[1])*scales[1]) * xd[token, group] * _half(raw[base+52] & 0xffff)
  names = {Q4_K: "linear_q4_k", Q5_K: "linear_q5_k", IQ4_XS: "linear_iq4_xs", Q6_K: "linear_q6"}
  return _decode_linear(out, out_features, group_count, group_dot, names[ggml_type])

def _wmma_layout(out:UOp, out_features:int, token_tile:int, output_tiles:int):
  output_waves = 2 if out_features % (32*output_tiles) == 0 else 1
  token_block, output_block = UOp.range(out.shape[0]//token_tile, 0), UOp.range(out_features//(16*output_tiles*output_waves), 1)
  # lane is a hardware WARP range (like the flash kernel): the fragment math stays visible without being
  # range-split into nested loops, which would scramble the WMMA fragment layout
  lane, wave = UOp.range(WARP_SIZE, -1, axis_type=AxisType.WARP), UOp.range(output_waves, 3, axis_type=AxisType.LOCAL)
  col, half = lane % 16, lane // 16
  outputs = tuple((output_block*output_waves+wave)*(16*output_tiles) + tile*16 + col for tile in range(output_tiles))
  inputs = tuple(token_block*token_tile + tile*16 + col for tile in range(token_tile//16))
  tokens = tuple(tuple(token_block*token_tile + tile*16 + half*8 + i for i in range(8)) for tile in range(token_tile//16))
  return output_waves, token_block, output_block, lane, wave, half, outputs, inputs, tokens

def _wmma_stores(out, outputs, tokens, accs, update, half, lane, wave, output_waves, swizzle_stores:bool=False):
  # Accumulator fragment halves are exchanged between lane pairs (l, l^16).  HIP lowers the LDS
  # spill/reload to in-register ds_swizzle; direct ISA was paying 32 ds_load_b32 per WMMA tile.
  tt = len(tokens)
  if swizzle_stores:
    def values(acc:UOp) -> tuple[UOp, ...]:
      global _reg_swizzle_slot
      own = tuple(acc.after(update)[i].load() for i in range(8))
      # Park peers in REG so emit sees PERMLANEX×8 or SWIZZLE×8 consecutive.
      # AMD_WMMA_PEER_SWIZZLE=1: ds_swizzle ^16 (HIP-like) instead of permlanex16.
      peer_op = (lambda v: _swizzle_f32(v, 16)) if getenv("AMD_WMMA_PEER_SWIZZLE", 1) else _peer16_f32
      raw = [peer_op(own[i]) for i in range(8)]
      if getenv("AMD_WMMA_PEER_NOPARK", 0):
        peer = tuple(raw)
      else:
        tmp = UOp.placeholder((8,), dtypes.float, slot=_reg_swizzle_slot, addrspace=AddrSpace.REG)
        _reg_swizzle_slot += 1
        tmp = tmp.after(UOp.group(*[tmp[i].store(sw) for i, sw in enumerate(raw)]))
        peer = tuple(tmp[i].load() for i in range(8))
      low = half.eq(0)
      return tuple(low.where(own[i], peer[i+4]) if j == 0 else low.where(peer[i], own[i+4]) for i in range(4) for j in range(2))
    return [out[token, output].store(value) for ot,(output,output_accs) in enumerate(zip(outputs, accs))
            for tile,(tile_tokens,acc) in enumerate(zip(tokens, output_accs)) for token,value in zip(tile_tokens, values(acc))]
  flat_accs = [acc for output_accs in accs for acc in output_accs]
  lds = UOp.placeholder((output_waves, 32, len(flat_accs)*8), dtypes.float32, slot=33, addrspace=AddrSpace.LOCAL)
  stores = [lds[wave, lane, a*8+i].store(acc.after(update)[i].load()) for a,acc in enumerate(flat_accs) for i in range(8)]
  lds = lds.after(UOp.barrier(UOp.group(*stores)))
  def values(ai:int) -> tuple[UOp, ...]:
    own = tuple(lds[wave, lane, ai*8+i].load() for i in range(8))
    peer = tuple(lds[wave, lane ^ 16, ai*8+i].load() for i in range(8))
    low = half.eq(0)
    return tuple(low.where(own[i], peer[i+4]) if j == 0 else low.where(peer[i], own[i+4]) for i in range(4) for j in range(2))
  return [out[token, output].store(value) for ot,(output,output_accs) in enumerate(zip(outputs, accs))
          for tile,(tile_tokens,_acc) in enumerate(zip(tokens, output_accs)) for token,value in zip(tile_tokens, values(ot*tt+tile))]

def _quant_linear_wmma(out, x, out_features, in_features, type_words, layout, dequant, name, swizzle_stores:bool=False,
                       dequant_halves:Callable[[UOp, UOp], tuple[tuple[UOp, ...], tuple[UOp, ...]]]|None=None):
  x = x.reshape(out.shape[0], in_features)
  output_waves, token_block, output_block, lane, wave, physical_half, outputs, input_tokens, tokens = layout
  token_tile, output_tiles = len(tokens)*16, len(outputs)
  output_words = in_features // GGML_BLOCK_SIZE * type_words
  accs = tuple(tuple(UOp.placeholder((8,), dtypes.float32, slot=ot*(token_tile//16)+tile, addrspace=AddrSpace.REG)
                     for tile in range(token_tile // 16)) for ot in range(output_tiles))
  accs = tuple(tuple(acc.after(acc.store(acc.const_like(0))) for acc in output_accs) for output_accs in accs)
  group = UOp.range(in_features // Q8_GROUP_SIZE, 4, AxisType.REDUCE)
  block, subgroup = group // 8, group % 8
  wmma_accs = [list(output_accs) for output_accs in accs]
  for half in range(2):
    afrags = tuple(UOp.stack(*(x[input_token, group*32 + half*16 + i].cast(dtypes.float16) for i in range(16)))
                   for input_token in input_tokens)
    for output_tile,output in enumerate(outputs):
      base = output*output_words + block*type_words
      if dequant_halves is not None:
        if half == 0: both_halves = dequant_halves(base, subgroup)
        bfrag = UOp.stack(*both_halves[half])
      else:
        bfrag = UOp.stack(*dequant(base, subgroup, half))
      for tile,afrag in enumerate(afrags):
        previous = accs[output_tile][tile].after(group) if half == 0 else wmma_accs[output_tile][tile]
        wmma_accs[output_tile][tile] = UOp.wmma(afrag, bfrag, previous, *WMMA_ARG)
  update = UOp.group(*(acc.store(value) for output_accs,output_values in zip(accs, wmma_accs)
                       for acc,value in zip(output_accs, output_values))).end(group)
  stores = _wmma_stores(out, outputs, tokens, accs, update, physical_half, lane, wave, output_waves, swizzle_stores=swizzle_stores)
  return UOp.group(*stores).end(token_block, output_block, lane, wave).sink(arg=KernelInfo(name=name, opts_to_apply=()))

@functools.cache
def _q5_linear_f16_wmma_kernel(out:UOp, raw:UOp, x:UOp, out_features:int, in_features:int, ggml_type:int, direct_isa:bool=False) -> UOp:
  token_tile = 32 if out.shape[0] % 32 == 0 else 16
  output_tiles = 1
  def dequant(base:UOp, subgroup:UOp, half:int) -> tuple[UOp, ...]:
    d, dmin, scale, minimum = _q5_scales(raw, base, subgroup, _amd_load(raw[base], 4, packed_u32=direct_isa))
    qs_base = base + (4 if ggml_type == Q4_K else 12) + (subgroup // 2)*8 + half*4
    qwords = _amd_load(raw[qs_base], 4, packed_u32=direct_isa)
    hwords = _amd_load(raw[base+4+half*4], 4, packed_u32=direct_isa) if ggml_type == Q5_K else None
    words = tuple((qwords[i] >> ((subgroup&1)*4).cast(dtypes.uint32) & 0x0f0f0f0f) |
      (((hwords[i] >> subgroup.cast(dtypes.uint32) & 0x01010101) << 4) if hwords is not None else 0) for i in range(4))
    if direct_isa:
      scaled, bias = d*scale, dmin*minimum*-1
      return tuple(_amd_fma_to_f16((word >> (byte*8) & 255).float(), scaled, bias) for word in words for byte in range(4))
    return tuple(((word >> (byte*8) & 255).float()*d*scale-dmin*minimum).cast(dtypes.float16) for word in words for byte in range(4))
  return _quant_linear_wmma(out, x, out_features, in_features, Q4_WORDS if ggml_type == Q4_K else Q5_WORDS,
                            _wmma_layout(out, out_features, token_tile, output_tiles), dequant,
                            f"linear_q{4 if ggml_type == Q4_K else 5}_k_f16_wmma", swizzle_stores=direct_isa)

def _q6_scale_raw(raw:UOp, base:UOp, idx:UOp) -> UOp:
  return ((raw[base+48+idx//4] >> ((idx%4)*8).cast(dtypes.uint32)) & 255).cast(dtypes.uint8).bitcast(dtypes.int8).float()

def _q6_packed_word(low:UOp, high:UOp, lo_shift:UOp, hi_shift:UOp) -> UOp:
  # Match pre-isel LSHL_OR: OR(AND(low, mask), SHL(AND(high, mask2), 4))
  lo = (low >> lo_shift) & 0x0f0f0f0f
  hi = ((high >> hi_shift) & 0x03030303) << 4
  return lo | hi

def _q6_byte_f32(word:UOp, byte:int) -> UOp:
  # Pre-isel folds CAST(AND(SHR(word, shift), 255)) → V_CVT_F32_UBYTEn
  return ((word >> UOp.const(byte*8, dtypes.uint32)) & 255).cast(dtypes.float32)

def _q6_wmma_words(ql4:tuple[UOp, ...], qh4:tuple[UOp, ...], subgroup:UOp, half:int) -> tuple[UOp, ...]:
  # All four words in a half share lo/hi shifts (within steps by 4 do not cross 64/32 boundaries).
  within0 = (subgroup*32 + half*16) % 128
  lo_shift, hi_shift = (within0//64)*4, (within0//32)*2
  lo_s, hi_s = lo_shift.cast(dtypes.uint32), hi_shift.cast(dtypes.uint32)
  return tuple(_q6_packed_word(ql4[i], qh4[i], lo_s, hi_s) for i in range(4))

def _q6_wmma_frags(words:tuple[UOp, ...], scale:UOp, d:UOp, direct_isa:bool) -> tuple[UOp, ...]:
  if direct_isa:
    scaled, bias = scale*d, scale*d*-32
    return tuple(_amd_fma_to_f16(_q6_byte_f32(word, byte), scaled, bias) for word in words for byte in range(4))
  return tuple((((word >> (byte*8) & 255).float()-32)*scale*d).cast(dtypes.float16) for word in words for byte in range(4))

@functools.cache
def _q6_linear_f16_wmma_kernel(out:UOp, raw:UOp, x:UOp, out_features:int, in_features:int, direct_isa:bool=False) -> UOp:
  token_tile = 32 if out.shape[0] % 32 == 0 else 16
  def dequant(base:UOp, subgroup:UOp, half:int) -> tuple[UOp, ...]:
    scale_idx = subgroup*2 + half
    scale = ((raw[base+48+scale_idx//4] >> ((scale_idx%4)*8).cast(dtypes.uint32)) & 255).cast(dtypes.uint8).bitcast(dtypes.int8).float()
    d = _half(raw[base+52] & 0xffff)
    ql = _amd_load(raw[base+(subgroup//4)*16+(subgroup%2)*8+half*4], 4, packed_u32=direct_isa)
    qh = _amd_load(raw[base+32+(subgroup//4)*8+half*4], 4, packed_u32=direct_isa)
    return _q6_wmma_frags(_q6_wmma_words(ql, qh, subgroup, half), scale, d, direct_isa)
  def dequant_halves(base:UOp, subgroup:UOp) -> tuple[tuple[UOp, ...], tuple[UOp, ...]]:
    d = _half(raw[base+52] & 0xffff)
    scale0, scale1 = _q6_scale_raw(raw, base, subgroup*2), _q6_scale_raw(raw, base, subgroup*2+1)
    ql_off, qh_off = (subgroup//4)*16 + (subgroup%2)*8, 32 + (subgroup//4)*8
    ql0, ql1 = _amd_load(raw[base+ql_off], 4, packed_u32=True), _amd_load(raw[base+ql_off+4], 4, packed_u32=True)
    qh0, qh1 = _amd_load(raw[base+qh_off], 4, packed_u32=True), _amd_load(raw[base+qh_off+4], 4, packed_u32=True)
    return (_q6_wmma_frags(_q6_wmma_words(ql0, qh0, subgroup, 0), scale0, d, True),
            _q6_wmma_frags(_q6_wmma_words(ql1, qh1, subgroup, 1), scale1, d, True))
  return _quant_linear_wmma(out, x, out_features, in_features, Q6_WORDS, _wmma_layout(out, out_features, token_tile, 1),
                            dequant, "linear_q6_k_f16_wmma", swizzle_stores=direct_isa,
                            dequant_halves=dequant_halves if direct_isa else None)

@functools.cache
def _iq4_linear_f16_wmma_kernel(out:UOp, raw:UOp, x:UOp, lut:UOp, out_features:int, in_features:int, direct_isa:bool=False) -> UOp:
  token_tile = 32 if out_features <= 1024 and out.shape[0] % 32 == 0 else 64 if out.shape[0] % 64 == 0 and \
    (out_features <= 6144 or out_features == 5120 and in_features > 8192) else 128 if out.shape[0] % 128 == 0 else \
    32 if out.shape[0] % 32 == 0 else 16
  # Dual output tiles (old default for 1024<out<=6144) inflated LUT dequant VGPR live
  # ranges on some shapes. Single tile is default; AMD_IQ4_DUAL_OUT=1 restores.
  if (t:=getenv("AMD_IQ4_OUTPUT_TILES", 0)): output_tiles = t
  elif getenv("AMD_IQ4_DUAL_OUT", 0):
    output_tiles = 1 if out_features <= 1024 else 2 if out_features <= 6144 else 1
  else: output_tiles = 1
  layout = _wmma_layout(out, out_features, token_tile, output_tiles)
  output_waves, _, _, lane, wave, _, _, _, _ = layout
  local_lut = UOp.placeholder((256,), dtypes.uint32, slot=32, addrspace=AddrSpace.LOCAL)
  tid, lut_items = wave*32+lane, 256//(32*output_waves)
  # Packed global→LDS fill (HIP ds_store_b128): load u32×4 then scalar stores; emit folds to B128.
  if direct_isa and lut_items >= 4 and lut_items % 4 == 0:
    stores = []
    for i in range(0, lut_items, 4):
      packed = _amd_load(lut[tid*lut_items+i], 4, packed_u32=True)
      stores.extend(local_lut[tid*lut_items+i+j].store(packed[j]) for j in range(4))
    lut_fill = UOp.group(*stores)
  else:
    lut_fill = UOp.group(*(local_lut[tid*lut_items+i].store(lut[tid*lut_items+i]) for i in range(lut_items)))
  lut = local_lut.after(lut_fill.barrier())
  def dequant(base:UOp, subgroup:UOp, half:int) -> tuple[UOp, ...]:
    d, scale = _iq4_scales(raw, base, subgroup)
    scale = scale * d
    qwords = _amd_load(raw[base+2+subgroup*4], 4, packed_u32=direct_isa)
    # Packed-byte LUT gather for all shapes. The old >6144 nibble path inflated VGPR
    # live ranges (~102 vs HIP ~69) and lost ~10µs on 8192×2048 WMMA.
    pairs = tuple(lut[((qwords[word] >> (byte*8)) & 255).cast(dtypes.weakint)]
                  for word in range(4) for byte in range(4))
    if direct_isa: return tuple(_amd_packed_f16_mul_to_f16(pair, scale, half) for pair in pairs)
    values = tuple(_half((pair >> (half*16)) & 0xffff) for pair in pairs)
    return tuple((value*scale).cast(dtypes.float16) for value in values)
  return _quant_linear_wmma(out, x, out_features, in_features, IQ4_WORDS, layout, dequant, "linear_iq4_xs_f16_wmma", swizzle_stores=direct_isa)

def q8_linear(layer:Linear, x:Tensor) -> Tensor:
  assert layer.ggml_type in (Q4_K, Q5_K, Q6_K, IQ4_XS)
  tokens = int(x.numel()) // layer.in_features
  raw, out_features, in_features = layer.weight.uop, layer.out_features, layer.in_features
  def run(fxn:Callable[..., UOp], out:UOp, *srcs:UOp) -> Tensor:
    all_srcs = (out,)+srcs
    params = tuple(UOp.placeholder_like(src, slot=i) for i,src in enumerate(all_srcs))
    kernel = fxn(*params, out_features=out_features, in_features=in_features).call(*all_srcs)
    result = Tensor(out.after(kernel))
    if len(result.shape) == 3: result = result.sum(-1)
    result = result.reshape(*x.shape[:-1], out_features)
    return result if layer.bias is None else result + layer.bias
  out = Tensor.empty(tokens, out_features, dtype=dtypes.float32, device=x.device).uop
  if tokens % 16 == 0 and out_features % 16 == 0:
    fxn:Callable[..., UOp]
    if layer.ggml_type == IQ4_XS: fxn = functools.partial(_iq4_linear_f16_wmma_kernel, direct_isa=amd_direct_isa(x.device))
    elif layer.ggml_type == Q6_K: fxn = functools.partial(_q6_linear_f16_wmma_kernel, direct_isa=amd_direct_isa(x.device))
    else: fxn = functools.partial(_q5_linear_f16_wmma_kernel, ggml_type=layer.ggml_type, direct_isa=amd_direct_isa(x.device))
    extra = (iq4_half_lut(str(x.device)).uop,) if layer.ggml_type == IQ4_XS else ()
    return run(fxn, out, raw, x.cast(dtypes.float16).contiguous().uop, *extra)
  xq_, xd, xs = q8_quantize(x, tokens, in_features)
  decode = functools.partial(_quant_decode_kernel, ggml_type=layer.ggml_type)
  out = Tensor.empty(tokens, out_features, (in_features+1023)//1024, dtype=dtypes.float32, device=x.device).uop
  return run(decode, out, raw, xq_.uop, xd.uop, xs.uop)

# ******** tiny dense fp16 gemv ********

@functools.cache
def _amd_f16_gemv_kernel(out:UOp, w:UOp, x:UOp, *rest:UOp, in_features:int, out_features:int, tokens:int) -> UOp:
  bias: UOp|None = rest[0] if rest else None
  # one block per (token, output row), 32 lanes accumulate 4-wide chunks of the row
  lanes, val_chunk = WARP_SIZE, 4
  token, out_row = UOp.range(tokens, 0, AxisType.GLOBAL), UOp.range(out_features, 1, AxisType.GLOBAL)
  lane = UOp.range(lanes, 2, axis_type=AxisType.LOCAL)
  per = in_features // (lanes * val_chunk)
  assert per * lanes * val_chunk == in_features
  w = w.reshape((out_features, per, lanes*val_chunk))
  x = x.reshape((tokens, per, lanes*val_chunk))
  acc = UOp.const(0, dtypes.float32)
  for i in range(per):
    for j in range(val_chunk):
      acc = acc + w[out_row, i, lane*val_chunk + j].load().float() * x[token, i, lane*val_chunk + j].load().float()
  total = warp_reduce(acc, full_wave=True)
  if bias is not None: total = total + bias[token, out_row].load().float()
  return out[token, out_row.valid(lane.eq(0))].store(total).end(token, out_row, lane).sink(arg=KernelInfo(name="linear_f16_gemv", opts_to_apply=()))

def _view_back(t:Tensor) -> Tensor:
  """strip top-of-chain CAST(s) from a lazy weight: reading the raw file bytes in the kernel instead of
  materializing the cast into a fresh buffer every step"""
  uop = t.uop
  while uop.op is Ops.CAST: uop = uop.src[0]
  return Tensor(uop).reshape(t.shape)

def f16_gemv(layer:Linear, x:Tensor) -> Tensor:
  tokens = prod(x.shape[:-1])
  assert isinstance(tokens, int)
  weight = _view_back(layer.weight)
  x = x.contiguous() if x.dtype == dtypes.half else x.cast(dtypes.half).contiguous()
  out = Tensor.empty(tokens, layer.out_features, dtype=dtypes.float32, device=x.device)
  fxn = functools.partial(_amd_f16_gemv_kernel, in_features=layer.in_features, out_features=layer.out_features, tokens=tokens)
  srcs = (out, weight.reshape(-1), x.reshape(tokens, layer.in_features)) + (() if layer.bias is None else (_view_back(layer.bias),))
  return Tensor.custom_kernel(*srcs, fxn=fxn)[0].reshape(*x.shape[:-1], layer.out_features)

# ******** flash attention on the KV cache ********

def _vec_load(ptr:UOp, lanes:int) -> tuple[UOp, ...]:
  if lanes == 1: return (ptr.load().float(),)
  vec = _amd_load(ptr, lanes)
  return tuple(vec[i].float() for i in range(lanes))

@functools.cache
def _amd_flash_attention_decode_partial(out, stats, q, cache_kv, valid_kv_len, max_kv_len, block_n, waves=4):
  valid_kv_len = _unbind(valid_kv_len)
  _, B, H_KV, N, D = cast(tuple[int, int, int, int, int], cache_kv.shape)
  _, H, M, _ = cast(tuple[int, int, int, int], q.shape)
  assert M == 1 and H % H_KV == 0 and D % WARP_SIZE == 0 and max_kv_len <= N and max_kv_len % block_n == 0
  G, CHUNK, DPL, WAVES = H // H_KV, block_n, D // WARP_SIZE, waves
  assert CHUNK % WAVES == 0
  SEC = CHUNK // WAVES  # keys each wave scans independently
  live_chunks = (valid_kv_len+CHUNK-1)//CHUNK
  live_chunks = min(live_chunks, out.shape[2]) if isinstance(live_chunks, int) else live_chunks.minimum(out.shape[2])
  block_bhkv, block_chunk = UOp.range(B*H_KV, 0, AxisType.GLOBAL), UOp.range(live_chunks, 1, AxisType.GLOBAL)
  lane, wave = UOp.range(WARP_SIZE, -1, axis_type=AxisType.WARP), UOp.range(WAVES, 3, axis_type=AxisType.LOCAL)
  b, kv_head = block_bhkv // H_KV, block_bhkv % H_KV
  # per-lane query fragments for every GQA head, kept packed in registers; unpacked at use
  qf = tuple(_vec_load(q[b, kv_head*G+h, 0, lane*DPL], DPL) for h in range(G))
  zerof = UOp.const(0, dtypes.float)
  valids: list[UOp] = []
  scores: list[list[UOp]] = [[zerof]*G for _ in range(SEC)]
  vfrags: list[tuple[UOp, ...]] = [()]*SEC
  # Batch keys into one warp_reduce_many so swizzle/permlane stages share lgkm waits.
  # Default 8 = full SEC at waves=8 (one reduce for the wave's keys). Smaller batches
  # were for VGPR headroom; measured same vgpr with better wait sharing at 8.
  score_batch = getenv("AMD_FLASH_SCORE_BATCH", 16)
  for j0 in range(0, SEC, score_batch):
    js = range(j0, min(j0 + score_batch, SEC))
    dots: list[UOp] = []
    for j in js:
      key = block_chunk*CHUNK + wave*SEC + j
      valid = key < valid_kv_len
      valids.append(valid)
      kfrag = _vec_load(cache_kv[0, b, kv_head, key, lane*DPL], DPL)
      # V is prefetched in the score pass so both streams are in flight together
      vfrags[j] = _vec_load(cache_kv[1, b, kv_head, key, lane*DPL], DPL)
      dots.extend([sum((qf[h][i]*kfrag[i] for i in range(DPL)), zerof) for h in range(G)])
    for t, s in enumerate(warp_reduce_many(dots, full_wave=True)):
      j, h = j0 + t // G, t % G
      scores[j][h] = valids[j].where(s * (1/math.sqrt(D)), UOp.const(-math.inf, dtypes.float))
  ninf = UOp.const(-math.inf, dtypes.float)
  row_max = [functools.reduce(UOp.maximum, (scores[j][h] for j in range(SEC)), ninf) for h in range(G)]
  accs:list[list[UOp]] = [[UOp.const(0, dtypes.float)] * DPL for _ in range(G)]
  row_sums:list[UOp] = [UOp.const(0, dtypes.float) for _ in range(G)]
  for j in range(SEC):
    for h in range(G):
      beta = valids[j].where(((scores[j][h]-row_max[h])*LOG2E).exp2(), UOp.const(0, dtypes.float))
      # Invalid physical-cache rows are still speculatively loaded. Mask before the multiply so an
      # uninitialized NaN cannot leak through 0*NaN into the partial accumulator.
      accs[h] = [a + beta*valids[j].where(v, UOp.const(0, dtypes.float)) for a, v in zip(accs[h], vfrags[j])]
      row_sums[h] = row_sums[h] + beta
  # exchange across the block's waves through LDS (fp16 halves LDS so more blocks fit per CU)
  acc_lds = UOp.placeholder((WAVES, G, D), dtypes.half, slot=0, addrspace=AddrSpace.LOCAL)
  ml_lds = UOp.placeholder((WAVES, G, 2), dtypes.float, slot=1, addrspace=AddrSpace.LOCAL)
  lds_acc = acc_lds.reshape(WAVES, G, WARP_SIZE, DPL)
  stores = [lds_acc[wave, h, lane].store(UOp.stack(*accs[h]).cast(dtypes.half)) for h in range(G)]
  # NOTE: duplicate stores of the same value from every lane are harmless here
  stores += [ml_lds[wave, h, i].store(x) for h in range(G) for i, x in enumerate((row_max[h], row_sums[h]))]
  barrier = UOp.barrier(UOp.group(*stores))
  acc_lds, ml_lds = acc_lds.after(barrier), ml_lds.after(barrier)
  tid = wave*WARP_SIZE + lane
  final_stores:list[UOp] = []
  for i in range(-(-G*D//(WAVES*WARP_SIZE))):
    flat = tid + i*WAVES*WARP_SIZE
    h, d = flat // D, flat % D
    M = functools.reduce(UOp.maximum, (ml_lds[w, h, 0].load() for w in range(WAVES)), ninf)
    val = sum((((ml_lds[w, h, 0].load()-M)*LOG2E).exp2() * acc_lds[w, h, d].load().float() for w in range(WAVES)), UOp.const(0, dtypes.float))
    oidx = out[b, kv_head*G + h, block_chunk, d]
    if G*D % (WAVES*WARP_SIZE): oidx = out[b, (kv_head*G + h).valid(flat < G*D), block_chunk, d]
    final_stores.append(oidx.store(val))
  hstat = tid
  M = functools.reduce(UOp.maximum, (ml_lds[w, hstat, 0].load() for w in range(WAVES)), ninf)
  L = sum((((ml_lds[w, hstat, 0].load()-M)*LOG2E).exp2() * ml_lds[w, hstat, 1].load() for w in range(WAVES)), UOp.const(0, dtypes.float))
  q_head = (kv_head*G + hstat).valid(hstat < G) if WAVES*WARP_SIZE > G else kv_head*G + hstat
  final_stores += [stats[b, q_head, block_chunk, 0].store(M), stats[b, q_head, block_chunk, 1].store(L)]
  return UOp.group(*final_stores).end(lane, wave, block_chunk, block_bhkv).sink(arg=KernelInfo(name="flash_decode_partial", opts_to_apply=()))

@functools.cache
def _amd_flash_decode_combine(o:UOp, partial:UOp, stats:UOp, live:int|UOp) -> UOp:
  # one wave per (batch, head, 64-dim tile): every lane redundantly weights its chunks; no cross-lane traffic
  live = _unbind(live)
  B, H, C, D = cast(tuple[int, int, int, int], partial.shape)
  DT = 64 if D % 64 == 0 else WARP_SIZE  # dims per block
  assert D % DT == 0
  block_bh, block_dt = UOp.range(B*H, 0, AxisType.GLOBAL), UOp.range(D//DT, 1, AxisType.GLOBAL)
  lane = UOp.range(WARP_SIZE, 2, axis_type=AxisType.LOCAL)
  b, h = block_bh // H, block_bh % H
  NPD = DT // WARP_SIZE  # output dims per lane
  dims = tuple(block_dt*DT + lane*NPD + i for i in range(NPD))
  # Unroll short combines. live=32 (kv_len=2048) is ~138 VGPRs but closes the HIP decode gap
  # (~54→51µs). Cap via AMD_COMBINE_UNROLL (larger live stays looped).
  if isinstance(live, int) and live <= getenv("AMD_COMBINE_UNROLL", 32):
    chunk_max = functools.reduce(lambda m, ci: UOp.maximum(m, stats[b, h, ci, 0].load()),
                                 range(live), UOp.const(-math.inf, dtypes.float))
    acc = [UOp.const(0, dtypes.float) for _ in range(NPD)]
    weight_sum = UOp.const(0, dtypes.float)
    for ci in range(live):
      w = ((stats[b, h, ci, 0].load() - chunk_max) * LOG2E).exp2()
      acc = [a + w * partial[b, h, ci, d].load() for a, d in zip(acc, dims)]
      weight_sum = weight_sum + w * stats[b, h, ci, 1].load()
    inv = 1 / weight_sum
    return UOp.group(*[o[b, h, 0, d].store(a * inv) for a, d in zip(acc, dims)]) \
      .end(lane, block_dt, block_bh).sink(arg=KernelInfo(name="flash_decode_combine", opts_to_apply=()))
  chunk = UOp.range(live, 100, AxisType.REDUCE)
  def iloop(ph, val): return ph.store(ph.const_like(val))
  chunk_max = UOp.placeholder((1,), dtypes.float, slot=0, addrspace=AddrSpace.REG)
  chunk_max_i = chunk_max.after(iloop(chunk_max, -math.inf))
  update0 = chunk_max_i.store(chunk_max_i.after(chunk).maximum(stats[b, h, chunk, 0].load())).end(chunk)
  chunk_max = chunk_max_i.after(update0)
  chunk2 = UOp.range(live, 101, AxisType.REDUCE)
  acc = UOp.placeholder((NPD,), dtypes.float, slot=1, addrspace=AddrSpace.REG)
  weight_sum = UOp.placeholder((1,), dtypes.float, slot=2, addrspace=AddrSpace.REG)
  acc_i, weight_sum_i = acc.after(iloop(acc, 0)), weight_sum.after(iloop(weight_sum, 0))
  w = ((stats[b, h, chunk2, 0].load()-chunk_max)*LOG2E).exp2()
  update1 = UOp.group(*[acc_i[i].store(acc_i.after(chunk2)[i].load() + w*partial[b, h, chunk2, d].load()) for i, d in enumerate(dims)],
                      weight_sum_i[0].store(weight_sum_i.after(chunk2)[0].load() + w*stats[b, h, chunk2, 1].load())).end(chunk2)
  acc, weight_sum = acc_i.after(update1), weight_sum_i.after(update1)
  inv = 1 / weight_sum[0].load()
  return UOp.group(*[o[b, h, 0, d].store(acc[i].load() * inv) for i, d in enumerate(dims)]) \
    .end(lane, block_dt, block_bh).sink(arg=KernelInfo(name="flash_decode_combine", opts_to_apply=()))

def amd_flash_attention_decode(q:Tensor, cache_kv:Tensor, valid_kv_len:int|UOp, max_kv_len:int) -> Tensor:
  B, H, D = cache_kv.shape[1], q.shape[1], cache_kv.shape[4]
  chunks = min(256, max_kv_len // 64)
  partial = Tensor.empty(B, H, chunks, D, dtype="float32", device=q.device)
  stats = Tensor.empty(B, H, chunks, 2, dtype="float32", device=q.device)
  # Eight waves keeps the LDS exchange mapping correct for both direct ISA and HIP.
  # AMD_FLASH_WAVES overrides (must divide block_n=64).
  waves = getenv("AMD_FLASH_WAVES", 8)
  assert 64 % waves == 0, f"AMD_FLASH_WAVES={waves} must divide block_n=64"
  fxn = functools.partial(_amd_flash_attention_decode_partial, valid_kv_len=valid_kv_len, max_kv_len=max_kv_len, block_n=64, waves=waves)
  partial, stats = Tensor.custom_kernel(partial, stats, q, cache_kv, fxn=fxn)[:2]
  live = (valid_kv_len+63)//64
  live = min(live, chunks) if isinstance(live, int) else live.minimum(chunks)
  out = Tensor.empty(B, H, 1, D, dtype="float32", device=q.device)
  fxn = functools.partial(_amd_flash_decode_combine, live=live)
  return Tensor.custom_kernel(out, partial, stats, fxn=fxn)[0]

@functools.cache
def _amd_flash_attention(o:UOp, q:UOp, cache:UOp, valid_kv_len:int|UOp, q_start:int|UOp|None=None,
                         acc_small:bool=False) -> UOp:
  valid_kv_len, q_start = _unbind(valid_kv_len), _unbind(q_start) if q_start is not None else None
  BH, M, D = q.shape
  _, B, H_KV, physical_n, cache_dim = cache.shape
  k, v = cache[0].reshape(B*H_KV, physical_n, cache_dim), cache[1].reshape(B*H_KV, physical_n, cache_dim)
  assert k.shape == v.shape and BH % k.shape[0] == 0 and k.shape[2] == D
  gqa_group = BH // k.shape[0]
  if isinstance(M, int) and isinstance(valid_kv_len, int): assert M % BLOCK_M == 0 and valid_kv_len % BLOCK_N == 0
  assert isinstance(D, int) and D % WMMA_K == 0 and D % LANES_PER_WAVE_N == 0
  TM, TN, TD, SCALE = BLOCK_M//(WAVES_M*LANES_PER_WAVE_M), BLOCK_N//LANES_PER_WAVE_N, D//(WAVES_N*LANES_PER_WAVE_N), 1/math.sqrt(D)
  # query row 0 sits at sequence position q_base (the queries may be padded beyond valid_kv_len - q_base rows)
  q_base = valid_kv_len - M if q_start is None else q_start
  block_bh, block_m = UOp.range(BH, 0, AxisType.GLOBAL), UOp.range(M // BLOCK_M, 1, AxisType.GLOBAL)
  kv_head = block_bh // gqa_group
  q, o = (x.reshape(BH, M//BLOCK_M, BLOCK_M, D)[block_bh, block_m] for x in (q, o))
  k, v = k[kv_head], v[kv_head]
  wave_m, wave_n, lane = UOp.range(WAVES_M, 2, AxisType.LOCAL), UOp.range(WAVES_N, 3, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
  tid, lane_m, lane_n = (wave_m * WAVES_N + wave_n) * WARP_SIZE + lane, lane // LANES_PER_WAVE_N, lane % LANES_PER_WAVE_N
  Q_ELEMS_PER_THREAD, KV_ELEMS_PER_THREAD = BLOCK_M * D // THREADS_PER_BLOCK, BLOCK_N * D // THREADS_PER_BLOCK
  QP_lds = UOp.placeholder((BLOCK_M, D + LDS_PAD), dtypes.half, slot=0, addrspace=AddrSpace.LOCAL)
  KV_lds = UOp.placeholder((BLOCK_N, D + LDS_PAD), dtypes.half, slot=1, addrspace=AddrSpace.LOCAL)[:, :D]
  acc, m_i, l_i = _reg((TM, TD), 2, 0), _reg((TM,), 3, -math.inf), _reg((TM,), 4, 0)
  n_tile = UOp.range((q_base + (block_m + 1) * BLOCK_M + BLOCK_N - 1) // BLOCK_N, 100, AxisType.REDUCE)
  Q_lds = QP_lds[:, :D]
  Q_store = Q_lds.after(n_tile).reshape(THREADS_PER_BLOCK, Q_ELEMS_PER_THREAD)[tid].store(q.reshape(THREADS_PER_BLOCK, Q_ELEMS_PER_THREAD)[tid])
  load_k = UOp.range(KV_ELEMS_PER_THREAD, 90)
  kval = k.reshape(physical_n*D)[n_tile*BLOCK_N*D + tid*KV_ELEMS_PER_THREAD + load_k].float()
  K_store = KV_lds.reshape(THREADS_PER_BLOCK, KV_ELEMS_PER_THREAD)[tid, load_k].store(kval).end(load_k)
  qk_load_barrier = UOp.barrier(UOp.group(Q_store, K_store))
  Q_lds, KV_lds_k = Q_lds.after(qk_load_barrier), KV_lds.after(qk_load_barrier)
  S_reg = _reg((TM, TN), 6, 0, n_tile)
  # AMD_FLASH_UNROLL: 0=off; 1 or 6=soft+corr python-unroll (const REG slots). WMMA stays ranged —
  # frag.after(store) cin was wrong; const_like(0) chains break direct-ISA emit (half2 LDS).
  # AMD_FLASH_ACC_SEP: keep WMMA C in ACC; copy out to a const-indexed soft buffer before softmax.
  # Default on: after soft vscnt + scratch addr CSE this is faster (~2.1ms vs ~2.7ms prefill).
  _acc_sep = bool(getenv("AMD_FLASH_ACC_SEP", 1))
  # Park one ACC pack per column (python-unrolled WMMA). Flash-only — AMD_WMMA_ACC_SMALL
  # also parks ≤64 quant tiles and breaks Q4/Q6; prefer AMD_FLASH_ACC_SMALL / acc_small arg.
  _acc_small = bool(acc_small or getenv("AMD_WMMA_ACC_SMALL", 0) or getenv("AMD_FLASH_ACC_SMALL", 0))
  _fu = getenv("AMD_FLASH_UNROLL", 0)
  if _fu == 1: _fu = 6
  if _acc_sep: _fu |= 6  # soft/corr need const REG indices on the working buffer
  S_view = S_reg.reshape(TM // WMMA_ACC, WMMA_ACC, TN).permute(0, 2, 1)
  Q_view = Q_lds.reshape(WAVES_M, TM // WMMA_ACC, WMMA_M, D // WMMA_K, WMMA_K)
  K_view = KV_lds_k.reshape(TN, WMMA_N, D // WMMA_K, WMMA_K)
  # AMD_FLASH_K_UNROLL: chain K WMMA in Python (more static WMMA like HIP). Opt-in.
  _k_unroll = bool(_acc_small and getenv("AMD_FLASH_K_UNROLL", 0))
  if _acc_small and _k_unroll:
    qk_stores = []
    for tn_i in range(TN):
      for tm_i in range(TM // WMMA_ACC):
        S_frag = S_view[tm_i, tn_i]
        acc = S_frag  # zero-init / parked ACC
        for k_i in range(D // WMMA_K):
          q_frag = Q_view[wave_m, tm_i, lane_n, k_i]
          k_frag = K_view[tn_i, lane_n, k_i]
          acc = UOp.wmma(q_frag, k_frag, acc, *WMMA_ARG)
        qk_stores.append(S_frag.store(acc))
    qk_done = UOp.group(*qk_stores)
  elif _acc_small:
    k_qk = UOp.range(D//WMMA_K, 101, AxisType.REDUCE)
    qk_stores = []
    for tn_i in range(TN):
      for tm_i in range(TM // WMMA_ACC):
        S_frag = S_view[tm_i, tn_i]
        q_frag = Q_view[wave_m, tm_i, lane_n, k_qk]
        k_frag = K_view[tn_i, lane_n, k_qk]
        qk_stores.append(S_frag.store(UOp.wmma(q_frag, k_frag, S_frag.after(k_qk), *WMMA_ARG)))
    qk_done = UOp.group(*qk_stores).end(k_qk)
  else:
    k_qk, tm1, tn1 = UOp.range(D//WMMA_K, 101, AxisType.REDUCE), UOp.range(TM//WMMA_ACC, 200), UOp.range(TN, 201)
    S_frag = S_view[tm1, tn1]
    q_frag = Q_view[wave_m, tm1, lane_n, k_qk]
    k_frag = K_view[tn1, lane_n, k_qk]
    qk_done = S_frag.store(UOp.wmma(q_frag, k_frag, S_frag.after(k_qk), *WMMA_ARG)).end(tm1, tn1).end(k_qk)
  # Scale+mask: default on WMMA/scratch buffer first. Predicated where-stores into a
  # REG-promoted const-index soft buffer were dropped (broke masked SDPA).
  # AMD_FLASH_SOFT_SCALE=1: copy raw WMMA→soft first, then scale+mask on soft (VGPR).
  _soft_scale = bool(_acc_sep and getenv("AMD_FLASH_SOFT_SCALE", 1))
  if not _soft_scale:
    S_reg = S_reg.after(qk_done, S_reg.store(S_reg * SCALE))
    if _fu & 2:
      mask_stores = []
      for rm_i in range(TM):
        for rn_i in range(TN):
          q_idx = q_base + block_m * BLOCK_M + wave_m * WMMA_M + rm_i * LANES_PER_WAVE_M + lane_m
          k_idx = n_tile * BLOCK_N + rn_i * LANES_PER_WAVE_N + lane_n
          mask_stores.append(S_reg[rm_i, rn_i].store((k_idx <= q_idx).where(S_reg[rm_i, rn_i], S_reg[rm_i, rn_i].const_like(-math.inf))))
      S_masked = S_reg.after(UOp.group(*mask_stores))
    else:
      rm, rn = UOp.range(TM, 250), UOp.range(TN, 251)
      q_idx = q_base + block_m * BLOCK_M + wave_m * WMMA_M + rm * LANES_PER_WAVE_M + lane_m
      k_idx = n_tile * BLOCK_N + rn * LANES_PER_WAVE_N + lane_n
      S_masked = S_reg.after(S_reg[rm, rn].store((k_idx <= q_idx).where(S_reg[rm, rn], S_reg[rm, rn].const_like(-math.inf))).end(rm, rn))
  else:
    S_masked = S_reg.after(qk_done)
  if _acc_sep:
    # Copy into a const-index soft buffer (REG-promotable) for softmax.
    # Skip zero-init: every slot is overwritten by the copy before any read.
    S_soft = UOp.placeholder((TM, TN), dtypes.float, slot=16, addrspace=AddrSpace.REG)
    if _soft_scale and getenv("AMD_FLASH_SOFT_FUSE", 1):
      # Scale+mask while copying ACC→soft (one SSTORE wave; was copy then rewrite).
      sm_stores = []
      for rm_i in range(TM):
        for rn_i in range(TN):
          q_idx = q_base + block_m * BLOCK_M + wave_m * WMMA_M + rm_i * LANES_PER_WAVE_M + lane_m
          k_idx = n_tile * BLOCK_N + rn_i * LANES_PER_WAVE_N + lane_n
          scaled = S_masked[rm_i, rn_i] * SCALE
          sm_stores.append(S_soft[rm_i, rn_i].store((k_idx <= q_idx).where(scaled, scaled.const_like(-math.inf))))
      S_reg = S_soft.after(UOp.group(*sm_stores))
    elif getenv("AMD_FLASH_VEC_COPY", 0) and (TM * TN) % 4 == 0:
      # float4 shrink copies → vec SLOAD/SSTORE → scratch_*_b128 on the S_reg side.
      src, dst = S_masked.reshape(TM * TN), S_soft.reshape(TM * TN)
      S_reg = S_soft.after(UOp.group(*[dst[i:i+4].store(src[i:i+4]) for i in range(0, TM * TN, 4)]))
      if _soft_scale:
        sm_stores = []
        for rm_i in range(TM):
          for rn_i in range(TN):
            q_idx = q_base + block_m * BLOCK_M + wave_m * WMMA_M + rm_i * LANES_PER_WAVE_M + lane_m
            k_idx = n_tile * BLOCK_N + rn_i * LANES_PER_WAVE_N + lane_n
            scaled = S_reg[rm_i, rn_i] * SCALE
            sm_stores.append(S_reg[rm_i, rn_i].store((k_idx <= q_idx).where(scaled, scaled.const_like(-math.inf))))
        S_reg = S_reg.after(UOp.group(*sm_stores))
    else:
      S_reg = S_soft.after(UOp.group(*[S_soft[ri, rj].store(S_masked[ri, rj]) for ri in range(TM) for rj in range(TN)]))
      if _soft_scale:
        # Per-element scale+mask so REG promote sees scalar SSTOREs (bulk store desyncs slots).
        sm_stores = []
        for rm_i in range(TM):
          for rn_i in range(TN):
            q_idx = q_base + block_m * BLOCK_M + wave_m * WMMA_M + rm_i * LANES_PER_WAVE_M + lane_m
            k_idx = n_tile * BLOCK_N + rn_i * LANES_PER_WAVE_N + lane_n
            scaled = S_reg[rm_i, rn_i] * SCALE
            sm_stores.append(S_reg[rm_i, rn_i].store((k_idx <= q_idx).where(scaled, scaled.const_like(-math.inf))))
        S_reg = S_reg.after(UOp.group(*sm_stores))
  else:
    S_reg = S_masked
  if _fu & 2:
    m_ij = _reg((TM,), 7, -math.inf, n_tile)
    for rn_i in range(TN):
      m_ij = m_ij.after(m_ij.store(m_ij.maximum(S_reg[:, rn_i])))
    m_ij = m_ij.after(UOp.group(*[m_ij[ri].store(warp_reduce(m_ij[ri], maximum=True)) for ri in range(TM)]))
    tile_max = m_ij.reshape(TM, 1).expand(TM, TN).maximum(-1e30)
    S_reg = S_reg.after(S_reg.store(((S_reg - tile_max) * LOG2E).exp2()))
    p_local = _reg((TM,), 8, 0, n_tile)
    p_sum = p_local.after(UOp.group(*[p_local[ri].store(sum((warp_reduce(S_reg[ri, rn_i]) for rn_i in range(TN)), S_reg.const_like(0)))
                                      for ri in range(TM)]))
  else:
    m_ij, rm2 = _reg((TM,), 7, -math.inf, n_tile), UOp.range(TN, 261, AxisType.REDUCE)
    m_ij = m_ij.after(m_ij.store(m_ij.after(rm2).maximum(S_reg[:, rm2])).end(rm2))
    ri_w = UOp.range(TM, 270)
    m_ij = m_ij.after(m_ij[ri_w].store(warp_reduce(m_ij[ri_w], maximum=True)).end(ri_w))
    tile_max = m_ij.reshape(TM, 1).expand(TM, TN).maximum(-1e30)
    S_reg = S_reg.after(S_reg.store(((S_reg - tile_max) * LOG2E).exp2()))
    p_local, ri_ws = _reg((TM,), 8, 0, n_tile), UOp.range(TM, 295)
    p_sum = p_local.after(p_local[ri_ws].store(sum((warp_reduce(S_reg[ri_ws, rn]) for rn in range(TN)), S_reg.const_like(0))).end(ri_ws))
  P_lds = QP_lds.flatten()[:WAVES_N * BLOCK_M * BLOCK_N].reshape(WAVES_N, BLOCK_M, BLOCK_N)
  P_write = P_lds.reshape(WAVES_N, WAVES_M, TM, LANES_PER_WAVE_M, 1, TN, LANES_PER_WAVE_N, 1).permute((1, 0, 3, 6, 2, 4, 5, 7)) \
    .reshape(THREADS_PER_BLOCK, TM, TN)
  P_store = P_write[tid].store(S_reg.cast(dtypes.half))
  beta_i = UOp.placeholder((TM,), dtypes.float, slot=9, addrspace=AddrSpace.REG)
  if _fu & 4:
    acc_c, l_c, m_c = acc.after(n_tile), l_i.after(n_tile), m_i.after(n_tile)
    corr = []
    for ri in range(TM):
      m_new = m_c[ri].maximum(m_ij[ri])
      alpha_val, beta_val = ((m_c[ri] - m_new) * LOG2E).exp2(), ((m_ij[ri] - m_new) * LOG2E).exp2()
      corr.append(UOp.group(*[acc_c[ri, rj].store(alpha_val * acc_c[ri, rj]) for rj in range(TD)],
                           l_c[ri].store(alpha_val * l_c[ri] + beta_val * p_sum[ri]),
                           m_c[ri].store(m_new), beta_i[ri].store(beta_val)))
    correction = UOp.group(*corr)
    acc, l_i, m_i, beta_i = acc.after(correction), l_i.after(correction), m_i.after(correction), beta_i.after(correction)
  else:
    ri4, rj4 = UOp.range(TM, 330), UOp.range(TD, 331)
    m_new = m_i[ri4].maximum(m_ij[ri4])
    alpha_val, beta_val = ((m_i[ri4] - m_new) * LOG2E).exp2(), ((m_ij[ri4] - m_new) * LOG2E).exp2()
    correction = UOp.group(acc[ri4, rj4].store(alpha_val * acc[ri4, rj4]).end(rj4),
                           l_i[ri4].store(alpha_val * l_i[ri4] + beta_val * p_sum[ri4]),
                           m_i[ri4].store(m_new), beta_i[ri4].store(beta_val)).end(ri4)
    acc, l_i, m_i, beta_i = acc.after(correction), l_i.after(correction), m_i.after(correction), beta_i.after(correction)
  V_lds = UOp.placeholder((D, BLOCK_N + LDS_PAD), dtypes.half, slot=1, addrspace=AddrSpace.LOCAL)[:, :BLOCK_N]
  V_copy, load_v = V_lds.after(qk_done).permute(1, 0), UOp.range(KV_ELEMS_PER_THREAD, 390)
  vval = v.reshape(physical_n*D)[n_tile*BLOCK_N*D + tid*KV_ELEMS_PER_THREAD + load_v].float()
  V_store = V_copy.reshape(THREADS_PER_BLOCK, KV_ELEMS_PER_THREAD)[tid, load_v].store(vval).end(load_v)
  pv_barrier = UOp.barrier(UOp.group(P_store, V_store))
  P_lds, V_lds = P_lds.after(pv_barrier), V_lds.after(pv_barrier)
  pv_acc = _reg((TM, TD), 10, 0, n_tile).after(pv_barrier)
  pv_view = pv_acc.reshape(TM // WMMA_ACC, WMMA_ACC, TD).permute(0, 2, 1)
  P_view = P_lds[wave_n].reshape(WAVES_M, TM // WMMA_ACC, WMMA_M, BLOCK_N // WMMA_K, WMMA_K)
  V_view = V_lds.reshape(WAVES_N, TD, WMMA_N, BLOCK_N // WMMA_K, WMMA_K)
  if _acc_small and _k_unroll:
    pv_stores = []
    for td_i in range(TD):
      for tm_i in range(TM // WMMA_ACC):
        pv_frag = pv_view[tm_i, td_i]
        acc = pv_frag
        for k_i in range(BLOCK_N // WMMA_K):
          p_frag = P_view[wave_m, tm_i, lane_n, k_i]
          v_frag = V_view[wave_n, td_i, lane_n, k_i]
          acc = UOp.wmma(p_frag, v_frag, acc, *WMMA_ARG)
        pv_stores.append(pv_frag.store(acc))
    pv_done = UOp.group(*pv_stores)
  elif _acc_small:
    k_pv = UOp.range(BLOCK_N//WMMA_K, 400, AxisType.REDUCE)
    pv_stores = []
    for td_i in range(TD):
      for tm_i in range(TM // WMMA_ACC):
        pv_frag = pv_view[tm_i, td_i]
        p_frag = P_view[wave_m, tm_i, lane_n, k_pv]
        v_frag = V_view[wave_n, td_i, lane_n, k_pv]
        pv_stores.append(pv_frag.store(UOp.wmma(p_frag, v_frag, pv_frag.after(k_pv), *WMMA_ARG)))
    pv_done = UOp.group(*pv_stores).end(k_pv)
  else:
    k_pv, tm2, tn2 = UOp.range(BLOCK_N//WMMA_K, 400, AxisType.REDUCE), UOp.range(TM//WMMA_ACC, 401), UOp.range(TD, 402)
    pv_frag = pv_view[tm2, tn2]
    p_frag = P_view[wave_m, tm2, lane_n, k_pv]
    v_frag = V_view[wave_n, tn2, lane_n, k_pv]
    pv_done = pv_frag.store(UOp.wmma(p_frag, v_frag, pv_frag.after(k_pv), *WMMA_ARG)).end(tm2, tn2).end(k_pv)
  pv_acc = pv_acc.after(pv_done)
  if _acc_sep:
    # Fully overwritten before read — skip REG zero-fill (same as S_soft).
    pv_soft = UOp.placeholder((TM, TD), dtypes.float, slot=17, addrspace=AddrSpace.REG)
    if getenv("AMD_FLASH_VEC_COPY", 0) and (TM * TD) % 4 == 0:
      src, dst = pv_acc.reshape(TM * TD), pv_soft.reshape(TM * TD)
      pv_acc = pv_soft.after(UOp.group(*[dst[i:i+4].store(src[i:i+4]) for i in range(0, TM * TD, 4)]))
    else:
      pv_acc = pv_soft.after(UOp.group(*[pv_soft[ri, rj].store(pv_acc[ri, rj]) for ri in range(TM) for rj in range(TD)]))
  ri5, rj5 = UOp.range(TM, 410), UOp.range(TD, 411)
  if _fu & 4 and getenv("AMD_FLASH_ACC_UNROLL", 1):
    # Const-index acc+= (better addressing). Slots 2/3/4 stay unpromoted via
    # AMD_REG_PROMOTE_SKIP_SLOTS so REDUCE-carried values remain correct.
    acc_c = acc.after(n_tile)
    acc_up = UOp.group(*[acc_c[ri, rj].store(acc_c[ri, rj] + beta_i[ri] * pv_acc[ri, rj])
                         for ri in range(TM) for rj in range(TD)])
    n_tile_end = acc_up.barrier().end(n_tile)
  else:
    n_tile_end = acc[ri5, rj5].store(acc[ri5, rj5] + beta_i[ri5] * pv_acc[ri5, rj5]).end(ri5, rj5).barrier().end(n_tile)
  acc, l_i, m_i = acc.after(n_tile_end), l_i.after(n_tile_end), m_i.after(n_tile_end)
  acc = acc.after(acc.store(acc * (1 / l_i).reshape(TM, 1).expand(TM, TD)))
  o = o.reshape(WAVES_M, TM, LANES_PER_WAVE_M, 1, WAVES_N, TD, LANES_PER_WAVE_N, 1) \
    .permute((0, 4, 2, 6, 1, 3, 5, 7)).reshape(THREADS_PER_BLOCK, TM, TD)
  return o[tid].store(acc).end(wave_m, wave_n, lane).end(block_m, block_bh).sink(arg=KernelInfo(opts_to_apply=()))

def flash_attention(q:Tensor, assigned_kv:Tensor, valid_end:int|UOp) -> Tensor:
  # cached flash attention on the half KV cache (already written through assigned_kv); valid_end stays bound at the graph level
  T_real, q_start = q.shape[2], None
  if resolve(T_real == 1): return amd_flash_attention_decode(q.half(), assigned_kv, valid_end, cast(int, assigned_kv.shape[3]))
  # Direct ISA can render the hand WMMA prefill correctly, but still trails HIP and
  # often SDPA. With AMD_FLASH_ACC_SMALL (default on for DIRECT): ~670µs vs ~1030µs
  # scratch ACC; HIP ~268µs. SDPA fallback ~306µs remains the non-DIRECT default.
  # Do NOT set AMD_WMMA_ACC_SMALL globally — parks ≤64 quant tiles and breaks Q4/Q6.
  if amd_direct_isa(q.device) and not getenv("AMD_FLASH_DIRECT", 0):
    k, v = assigned_kv[0, :, :, 0:valid_end, :], assigned_kv[1, :, :, 0:valid_end, :]
    mask = Tensor.full((1, 1, T_real, valid_end), float("-inf"), dtype=q.dtype, device=q.device, buffer=False).triu(valid_end-T_real+1)
    return q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)
  # Flash-only ACC residency via python-unrolled WMMA (≥2 packs → renderer auto-park).
  # AMD_FLASH_ACC_SMALL=0 disables; AMD_WMMA_ACC_SMALL=1 forces (unsafe for quant).
  use_acc_small = bool(getenv("AMD_WMMA_ACC_SMALL", 0) or getenv("AMD_FLASH_ACC_SMALL", 1))
  if isinstance(T_real, UOp):
    # symbolic chunk: pad the queries to the static tile; garbage rows are sliced off
    T_pad = q.max_shape[2]
    assert T_pad % BLOCK_M == 0, "chunk_size must be a multiple of 32"
    q, q_start = q.pad_to((*q.shape[:2], T_pad, q.shape[3])), valid_end - T_real
  B, H, T, D = q.shape
  out = Tensor.empty(B*H, T, D, dtype="float32", device=q.device)
  fxn = functools.partial(_amd_flash_attention, valid_kv_len=valid_end, q_start=q_start, acc_small=use_acc_small)
  out = Tensor.custom_kernel(out, q.half().reshape(B*H, T, D), assigned_kv, fxn=fxn)[0].reshape(B, H, T, D)
  return out if q_start is None else out[:, :, :T_real]

# ******** gated delta net: fused recurrent scan ********

@functools.cache
def _gated_delta_prefill_kernel(core:UOp, q:UOp, k:UOp, v:UOp, beta:UOp, alpha:UOp, state:UOp, kq:UOp, start_pos:UOp|None=None) -> UOp:
  batch, heads, tokens, value_dim, row_tile = *core.shape, 4
  key_dim, alpha_dim = q.shape[-1], alpha.shape[-1] if len(alpha.shape) == 4 else 1
  assert all(isinstance(x, int) for x in (batch, heads, tokens, value_dim, key_dim)) and \
    (key_dim <= 32 or key_dim % 32 == 0) and value_dim % row_tile == 0
  batch, heads, tokens, value_dim, key_dim = cast(tuple[int, int, int, int, int], (batch, heads, tokens, value_dim, key_dim))
  key_chunks = (key_dim + 31) // 32
  core, v = (x.reshape(batch*heads, tokens, value_dim) for x in (core, v))
  q, k = (x.reshape(batch*heads, tokens, key_dim) for x in (q, k))
  beta, kq = (x.reshape(batch*heads, tokens) for x in (beta, kq))
  alpha, state = alpha.reshape(batch*heads, tokens, alpha_dim), state.reshape(batch*heads, value_dim, key_dim)
  bh_row, lane = UOp.range(batch*heads*value_dim//row_tile, 0), UOp.range(32, 1, axis_type=AxisType.LOCAL)
  bh, row_base = bh_row // (value_dim//row_tile), (bh_row % (value_dim//row_tile))*row_tile
  rows = tuple(row_base+i for i in range(row_tile))
  raw_cols = tuple(lane+i*32 for i in range(key_chunks))
  col_valids = tuple(col < key_dim for col in raw_cols)
  cols = tuple(col.valid(valid) for col,valid in zip(raw_cols, col_valids))
  current = UOp.placeholder((row_tile*key_chunks,), dtypes.float32, slot=0, addrspace=AddrSpace.REG)
  initial = None if start_pos is None else start_pos.eq(0)
  current = current.after(current.store(UOp.stack(*(valid.where(
    state[bh, row, col].float() if initial is None else initial.where(0, state[bh, row, col].float()), UOp.const(0, dtypes.float32))
    for row in rows for col,valid in zip(cols, col_valids)))))
  token = UOp.range(tokens, 2, AxisType.REDUCE)
  keys = tuple(valid.where(k[bh, token, col].load(), UOp.const(0, dtypes.float32)) for col,valid in zip(cols, col_valids))
  queries = tuple(valid.where(q[bh, token, col].load(), UOp.const(0, dtypes.float32)) for col,valid in zip(cols, col_valids))
  updates, stores = [], []
  for row_idx,row in enumerate(rows):
    previous = tuple(current.after(token)[row_idx*key_chunks+i].load() for i in range(key_chunks))
    av, bv = alpha[bh, token, row if alpha_dim > 1 else 0].load(), beta[bh, token].load()
    state_k = warp_reduce(sum((x*y for x,y in zip(previous, keys)), UOp.const(0, dtypes.float32)), full_wave=True)
    state_q = warp_reduce(sum((x*y for x,y in zip(previous, queries)), UOp.const(0, dtypes.float32)), full_wave=True)
    delta = (v[bh, token, row].load() - state_k*av) * bv
    updates += [x*av + delta*y for x,y in zip(previous, keys)]
    stores.append(core[bh, token, row.valid(lane.eq(0))].store(state_q*av + delta*kq[bh, token]))
  step = UOp.group(*stores, current.store(UOp.stack(*updates))).end(token)
  state_stores = (state[bh, row, col].store(current.after(step)[row_idx*key_chunks+i].load().cast(state.dtype))
                  for row_idx,row in enumerate(rows) for i,col in enumerate(cols))
  return UOp.group(*state_stores).end(lane, bh_row).sink(arg=KernelInfo(name="gated_delta_prefill", opts_to_apply=()))

def gated_delta_prefill(q:Tensor, k:Tensor, v:Tensor, beta:Tensor, alpha:Tensor, state:Tensor, start_pos:Tensor|None=None) -> Tensor:
  batch, heads, tokens, key_dim = q.shape
  value_dim = v.shape[-1]
  assert q.shape == k.shape and v.shape[:3] == beta.shape == (batch, heads, tokens) and state.shape == (batch, heads, value_dim, key_dim)
  assert alpha.shape[:3] == (batch, heads, tokens) and (len(alpha.shape) == 3 or alpha.shape[-1] in (1, value_dim))
  assert (key_dim <= 32 or key_dim % 32 == 0) and value_dim % 4 == 0
  core, kq = Tensor.empty_like(v), (q*k).sum(-1).contiguous()
  srcs = (core, q.contiguous(), k.contiguous(), v.contiguous(), beta.contiguous(), alpha.contiguous(), state, kq)
  if start_pos is None: return Tensor.custom_kernel(*srcs, fxn=_gated_delta_prefill_kernel)[0]
  contig = tuple(x.uop if x.uop.op is Ops.AFTER else x.uop.contiguous() for x in srcs)
  params = tuple(UOp.placeholder_like(x, slot=i) for i,x in enumerate(contig))
  assert start_pos.uop.is_bound_var
  # the bound start_pos reaches the graph through the state AFTER chain, like the flash kernels' valid_end
  call = _gated_delta_prefill_kernel(*params, kernel_var(start_pos.uop.src[0])).call(*contig)
  return Tensor(contig[0].after(call))

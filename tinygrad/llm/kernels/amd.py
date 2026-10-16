from __future__ import annotations
import contextlib, functools, math, os
from typing import Callable, cast
from tinygrad import Tensor, UOp, nn, Device, Context
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.helpers import prod, getenv, ALLOW_DEVICE_USAGE
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, resolve
from tinygrad.renderer.cstyle import HIPRenderer

BLOCK_M, BLOCK_N, WARP_SIZE = 32, 32, 32
WMMA_M, WMMA_N, WMMA_K = 16, 16, 16
WAVES_M, WAVES_N, LANES_PER_WAVE_M, LANES_PER_WAVE_N = 2, 2, 2, 16
WMMA_ACC, THREADS_PER_BLOCK = WMMA_M // LANES_PER_WAVE_M, WARP_SIZE * WAVES_M * WAVES_N
LDS_PAD, WMMA_ARG, LOG2E = 4, ((WMMA_M, WMMA_N, WMMA_K), 32), math.log2(math.e)

def flash_tile_dims(D:int=128) -> tuple[int, int, int]:
  """Per-thread tile locals for flash prefill: (TM, TN, TD). D=128 → (8, 2, 4)."""
  return (BLOCK_M // (WAVES_M * LANES_PER_WAVE_M),
          BLOCK_N // LANES_PER_WAVE_N,
          D // (WAVES_N * LANES_PER_WAVE_N))
GGML_BLOCK_SIZE, Q8_GROUP_SIZE = 256, 32
Q2_K, Q3_K, Q4_K, Q5_K, Q6_K = 10, 11, 12, 13, 14
IQ2_XS, IQ3_XXS, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS = 17, 18, 20, 21, 22, 23
# Legacy padded Q6 word count kept for unused DIRECT Q6-WMMA helpers; decode uses Q6_BYTES u16 views.
Q4_WORDS, Q5_WORDS, Q6_BYTES, IQ4_WORDS, Q6_WORDS = 36, 44, 210, 34, 53
QUANT_SIZES = {Q2_K: 84, Q3_K: 110, Q4_K: 144, Q5_K: 176, Q6_K: 210, IQ2_XS: 74,
               IQ3_XXS: 98, IQ4_NL: 144, IQ3_S: 110, IQ2_S: 82, IQ4_XS: 136}  # bytes per 256 weights
HALFWORD_QUANTS = (Q6_K, Q2_K, Q3_K, IQ2_XS, IQ3_XXS, IQ4_NL, IQ3_S, IQ2_S)
QUANT_NAMES = {Q2_K: "q2_k", Q3_K: "q3_k", Q4_K: "q4_k", Q5_K: "q5_k", Q6_K: "q6", IQ4_XS: "iq4_xs",
               IQ2_XS: "iq2_xs", IQ3_XXS: "iq3_xxs", IQ4_NL: "iq4_nl", IQ3_S: "iq3_s", IQ2_S: "iq2_s"}
AMD_PACKED_U32X4_LOAD = "__builtin_nontemporal_load((const unsigned_int4*){0})"
AMD_FMA_TO_F16 = "__builtin_amdgcn_fma_mixlo_f16({}, {}, {})"
AMD_PACKED_F16_MUL_TO_F16 = "__builtin_amdgcn_fma_mixlo_f16_packed({}, {}, {})"

def _unbind(v:int|UOp) -> int|UOp: return v.unbind_all()[0] if isinstance(v, UOp) else v

def _bound_int(v:int|UOp) -> int|None:
  if isinstance(v, int): return v
  if v.op is Ops.CONST and isinstance(v.arg, int): return v.arg
  if v.op is Ops.PARAM and isinstance((val:=getattr(v.arg, "val", None)), int): return val
  return None

@functools.cache
def amd_custom_kernels_supported(device:str|tuple[str, ...]|None) -> bool:
  if getenv("DISABLE_AMD_KERNELS"): return False
  # Wave32 and WMMA kernels support RDNA3/4. CDNA's wave64/MFMA is not supported.
  if isinstance(device, tuple): device = device[0]
  if device is None or device.split(":")[0] != "AMD": return False
  # @function contexts set ALLOW_DEVICE_USAGE=0 (scheduling must not open devices); the device is always open here
  with Context(ALLOW_DEVICE_USAGE=1):
    d = Device[device]
    t = getattr(d, "target", None)
    if t is None or t[0] not in (11, 12): return False
    # HIP C-style and DIRECT AMDRenderer both lower the amdgcn CUSTOM/CUSTOMI intrinsics.
    r = d.renderer
    return isinstance(r, HIPRenderer) or r.__class__.__name__ == "AMDRenderer"

@functools.cache
def amd_direct_isa(device:str|tuple[str, ...]|None) -> bool:
  if isinstance(device, tuple): device = device[0]
  if device is None or device.split(":")[0] != "AMD": return False
  with Context(ALLOW_DEVICE_USAGE=1): return Device[device].renderer.__class__.__name__ == "AMDRenderer"

@functools.cache
def _wmma_rdna4(device:str|tuple[str, ...]) -> bool:
  if isinstance(device, tuple): device = device[0]
  with Context(ALLOW_DEVICE_USAGE=1): return getattr(Device[device], "target")[0] == 12

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
  for offset in offsets:
    raw = [_peer16_f32(v) if offset == 16 else _swizzle_f32(v, offset) for v in chunk]
    # Always REG-park peers (NO_PARK* removed — NaN / decode regress).
    tmp = UOp.placeholder((len(chunk),), dtypes.float, slot=_reg_swizzle_slot, addrspace=AddrSpace.REG)
    _reg_swizzle_slot += 1
    tmp = tmp.after(UOp.group(*[tmp[i].store(sw) for i, sw in enumerate(raw)]))
    others = [tmp[i].load() for i in range(len(chunk))]
    chunk = [v.maximum(o) if maximum else v + o for v, o in zip(chunk, others)]
  return chunk

def _reg(shape:tuple[int, ...], slot:int, value:float, dep:UOp|None=None) -> UOp:
  ret = UOp.placeholder(shape, dtypes.float, slot=slot, addrspace=AddrSpace.REG)
  return ret.after((ret if dep is None else ret.after(dep)).store(ret.const_like(value)))

# ******** quant linear: q8-activation kernels over packed ggml weights ********

class Linear(nn.Linear):
  ggml_type:int|None = None
  use_custom_quant = True
  def __init__(self, in_features:int, out_features:int, bias=True):
    super().__init__(in_features, out_features, bias)
    self.in_features, self.out_features = in_features, out_features
  def set_quantized(self, decoded:Tensor):
    if self.in_features % GGML_BLOCK_SIZE: return
    packed_sizes = {typ: decoded.numel() // 256 * type_size for typ,type_size in QUANT_SIZES.items()}
    graph = decoded.uop.toposort()
    raw = next((u for u in graph if u.op is Ops.SHRINK and u.dtype == dtypes.uint8 and prod(u.shape) in packed_sizes.values()), None)
    # Full-span slices can canonicalize away; then the original packed uint8 buffer is the raw view.
    if raw is None: raw = next((u for u in graph if u.op is Ops.BUFFER and u.dtype == dtypes.uint8 and prod(u.shape) in packed_sizes.values()), None)
    if raw is None: return
    # Only unwrap storage/order-preserving views, then require the exact dequantization expression.
    # This rejects subsequent arithmetic and permutations, including RoPE's concatenated query weights.
    def unwrapped(u:UOp) -> UOp:
      while u.op in (Ops.RESHAPE, Ops.STAGE) or (u.op is Ops.CAST and dtypes.is_float(u.dtype) and dtypes.is_float(u.src[0].dtype)):
        u = u.src[0]
      return u
    # Several formats have the same byte count (Q3_K/IQ3_S and Q4_K/IQ4_NL). Match the expression, not just the size.
    for ggml_type, size in packed_sizes.items():
      if size != prod(raw.shape): continue
      expected = ggml_data_to_tensor(Tensor(raw), self.in_features * self.out_features, ggml_type)
      if unwrapped(decoded.uop).key == unwrapped(expected.uop).key: break
    else: return
    # Some blocks are only halfword-aligned; keep all formats as zero-copy views of the GGUF storage.
    word_dtype = dtypes.uint16 if ggml_type in HALFWORD_QUANTS else dtypes.uint32
    raw_offset = raw.contiguous_view_offset()
    if raw_offset is None or raw_offset % word_dtype.itemsize or raw.buf_uop.dtype != dtypes.uint8: return
    self.ggml_type = ggml_type
    self.weight = Tensor(raw).bitcast(word_dtype).contiguous()
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
    if self.ggml_type in QUANT_SIZES and supported:
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

def _amd_load(ptr:UOp, lanes:int|None=None, packed_u32:bool=False, stream:bool=False) -> UOp:
  assert ptr.op is Ops.INDEX
  # nontemporal scalar load: streamed weights must not evict the activations/KV cache from L2
  if lanes is None: return ptr.load(arg="nontemporal")
  buf, coords = ptr.src[0], ptr.src[1:]
  idx = sum((coord*math.prod(buf.shape[i+1:]) for i,coord in enumerate(coords)), UOp.const(0))
  shrink = UOp(Ops.SHRINK, src=(buf.flatten(), idx, UOp.const(lanes)))
  if packed_u32:
    assert lanes == 4 and ptr.dtype is dtypes.uint32
    return UOp(Ops.CUSTOM, src=(shrink,), arg=(AMD_PACKED_U32X4_LOAD, dtypes.uint32))
  return shrink.load(arg="nontemporal" if stream else None)

def _load_byte(raw:UOp, base:UOp, offset:int|UOp) -> UOp:
  size = raw.dtype.itemsize
  return (raw[base + offset//size].cast(dtypes.uint32) >> ((offset%size)*8)) & 255
def _load_u32(raw:UOp, base:UOp, offset:int|UOp, stream:bool=False) -> UOp:
  # Halfword-aligned formats cannot load u32 directly.
  lo, hi = (raw[base+offset//2+i] for i in range(2))
  if stream: lo, hi = _amd_load(lo), _amd_load(hi)
  return lo.cast(dtypes.uint32) | (hi.cast(dtypes.uint32) << 16)

def _half(value:UOp) -> UOp: return value.cast(dtypes.uint16).bitcast(dtypes.float16).float()

def _iq4_bytes(packed:UOp, shift:int|UOp) -> UOp:
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

def iq4_half_lut(device:str) -> Tensor:
  from tinygrad.runtime.autogen.ggml_common import kvalues_iq4nl
  return Tensor.const(tuple(x for j in range(16) for i in range(16) for x in (kvalues_iq4nl[i], kvalues_iq4nl[j])),
                      dtypes.float16).to(device, force=True).bitcast(dtypes.uint32)

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

def _iq_grid(device:str, ggml_type:int) -> Tensor:
  from tinygrad.runtime.autogen import ggml_common as ggml
  grid, words = {IQ2_XS: (ggml.iq2xs_grid, 2), IQ2_S: (ggml.iq2s_grid, 2),
                 IQ3_XXS: (ggml.iq3xxs_grid, 1), IQ3_S: (ggml.iq3s_grid, 1)}[ggml_type]
  return Tensor.const(tuple((v >> (32*i)) & 0xffffffff for v in grid for i in range(words)),
                      dtypes.uint32).to(device, force=True)

def _iq_even_signs(signs:UOp) -> UOp:
  parity = signs ^ (signs >> 4)
  parity ^= parity >> 2
  parity ^= parity >> 1
  return signs | ((parity & 1) << 7)

def _iq_signed_word(word:UOp, signs:UOp) -> UOp:
  mask = sum(((signs >> i) & 1) * (255 << (8*i)) for i in range(4))
  # Grid magnitudes are nonzero and <128, so each byte can be negated without a carry into its neighbor.
  return (word ^ mask) + (mask & 0x01010101)

def _quant_word(raw:UOp, base:UOp, subgroup:UOp, i:int|UOp, ggml_type:int, grid:UOp|None) -> UOp:
  # Four packed weight bytes, shared by integer-dot decode and FP16 WMMA prefill.
  def byte(offset): return _load_byte(raw, base, offset)
  def word(offset): return _load_u32(raw, base, offset, stream=ggml_type == Q6_K)
  if ggml_type == Q6_K:
    low = word((subgroup//4)*64 + (subgroup%2)*32 + i*4) >> ((subgroup%4//2)*4).cast(dtypes.uint32)
    high = word(128 + (subgroup//4)*32 + i*4) >> ((subgroup%4)*2).cast(dtypes.uint32)
    return (low & 0x0f0f0f0f) | ((high & 0x03030303) << 4)
  if ggml_type in (Q2_K, Q3_K):
    offset = (16 if ggml_type == Q2_K else 32) + (subgroup//4)*32 + i*4
    weights = (word(offset) >> ((subgroup%4)*2)) & 0x03030303
    if ggml_type == Q3_K: weights |= ((word(i*4) >> subgroup) & 0x01010101) << 2
    return weights
  if ggml_type in (IQ4_NL, IQ4_XS):
    packed = word(2+(i%4)*4) if ggml_type == IQ4_NL else _amd_load(raw[base+2+subgroup*4+i%4])
    return _iq4_bytes(packed, 4*(i//4))
  assert grid is not None
  grid_words = 2 if ggml_type in (IQ2_XS, IQ2_S) else 1
  if ggml_type in (IQ3_S, IQ3_XXS):
    index = byte(2+subgroup*8+i)
    if ggml_type == IQ3_S:
      index |= ((byte(66+subgroup) >> i) & 1) << 8
      signs = byte(74+subgroup*4+i//2)
    else: signs = _iq_even_signs((word(66+subgroup*4) >> (7*(i//2))) & 127)
  elif ggml_type == IQ2_XS:
    packed = raw[base+1+subgroup*4+i//2].cast(dtypes.uint32)
    index, signs = packed & 511, _iq_even_signs(packed >> 9)
  else:
    index = byte(2+subgroup*4+i//2) | (((byte(66+subgroup) >> (2*(i//2))) & 3) << 8)
    signs = byte(34+subgroup*4+i//2)
  return _iq_signed_word(grid[index*grid_words+i%grid_words], signs >> (4*(i%2)))

@functools.cache
def _quant_decode_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, xs:UOp, *grids:UOp,
                         out_features:int, in_features:int, ggml_type:int) -> UOp:
  def group_dot(token:UOp, output:UOp, group:UOp) -> UOp:
    block, subgroup = group//8, group%8
    base = (output*in_features//256 + block) * (QUANT_SIZES[ggml_type]//raw.dtype.itemsize)
    if ggml_type == IQ4_NL: base = (output*in_features//32 + group)*9
    def byte(offset): return _load_byte(raw, base, offset)
    def word(offset): return _load_u32(raw, base, offset)
    if ggml_type in (Q4_K, Q5_K):
      qs_base = base + (4 if ggml_type == Q4_K else 12) + (subgroup//2)*8
      # Keep the vector loads for these formats; scalarizing them hurts decode bandwidth.
      qs_pair = (_amd_load(raw[qs_base], 4), _amd_load(raw[qs_base+4], 4))
      if ggml_type == Q5_K: qh_pair = (_amd_load(raw[base+4], 4), _amd_load(raw[base+8], 4))
    xwords = _amd_load(xq[token, group, 0], 8)
    # One accumulator per scale group: 32 weights for Q4/Q5/IQ4_XS, two groups of 16 otherwise.
    dots = [UOp.const(0, dtypes.int32)] * (1 if ggml_type in (Q4_K, Q5_K, IQ4_XS) else 2)
    for i in range(8):
      if ggml_type in (Q4_K, Q5_K):
        weights = (qs_pair[i//4][i%4] >> ((subgroup&1)*4).cast(dtypes.uint32)) & 0x0f0f0f0f
        if ggml_type == Q5_K: weights |= ((qh_pair[i//4][i%4] >> subgroup.cast(dtypes.uint32)) & 0x01010101) << 4
      else: weights = _quant_word(raw, base, subgroup, i, ggml_type, grids[0] if grids else None)
      acc = i//(8//len(dots))
      dots[acc] = _amd_dp4a(weights, xwords[i], dots[acc])
    if ggml_type in (Q4_K, Q5_K):
      d, dmin, scale, minimum = _q5_scales(raw, base, subgroup)
      total = dots[0].float()*d*scale - (xs[token, group, 0].load()+xs[token, group, 1].load())*dmin*minimum
    elif ggml_type == IQ4_XS:
      d, scale = _iq4_scales(raw, base, subgroup)
      return dots[0].float() * xd[token, group] * d * scale
    elif ggml_type == Q6_K:
      # Subtract the quant offset via the activation sums instead of unpacking signed bytes.
      scales = [(raw[base+96+subgroup] >> (h*8)).cast(dtypes.uint8).bitcast(dtypes.int8).float() for h in range(2)]
      total = sum((dots[h].float()-32*xs[token, group, h].load())*scales[h] for h in range(2))
      return total * xd[token, group] * _half(raw[base+104])
    elif ggml_type in (Q2_K, Q3_K):
      total = UOp.const(0, dtypes.float32)
      for half in range(2):
        j = subgroup*2+half
        if ggml_type == Q2_K:
          scale = byte(j)
          total += dots[half].float() * (scale & 15).float() * _half(raw[base+40]) - \
                   xs[token, group, half] * (scale >> 4).float() * _half(raw[base+41])
        else:
          scale = ((byte(96+j%8) >> ((j//8)*4)) & 15) | (((byte(104+j%4) >> ((j//4)*2)) & 3) << 4)
          total += (dots[half].float() - 4*xs[token, group, half]) * (scale.cast(dtypes.int32)-32).float() * _half(raw[base+54])
    elif ggml_type in (IQ2_XS, IQ2_S):
      scales = byte((66 if ggml_type == IQ2_XS else 74)+subgroup)
      total = sum(dots[h].float() * (((scales >> (h*4)) & 15).float()+0.5) for h in range(2)) * 0.25 * _half(raw[base])
    else:
      total = (dots[0]+dots[1]).float() * _half(raw[base])
      if ggml_type == IQ3_S: total *= (1+2*((byte(106+subgroup//2) >> ((subgroup%2)*4)) & 15)).float()
      if ggml_type == IQ3_XXS: total *= ((word(66+subgroup*4) >> 28).float()+0.5)*0.5
    return total * xd[token, group]
  return _decode_linear(out, out_features, in_features//32, group_dot, "linear_"+QUANT_NAMES[ggml_type])

def _wmma_layout(out:UOp, out_features:int, token_tile:int, output_tiles:int):
  if out_features % (16*output_tiles): output_tiles = 1
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

def _wmma_stores(out, outputs, tokens, accs, update, half, lane, wave, output_waves, swizzle_stores:bool=False, rdna4:bool=False):
  # Accumulator fragment halves are exchanged between lane pairs (l, l^16).  HIP lowers the LDS
  # spill/reload to in-register ds_swizzle; direct ISA was paying 32 ds_load_b32 per WMMA tile.
  tt = len(tokens)
  # RDNA4 owns eight consecutive rows per half-wave, matching tokens directly.
  if rdna4:
    return [out[token, output].store(acc.after(update)[i].load()) for output,output_accs in zip(outputs, accs)
            for tile_tokens,acc in zip(tokens, output_accs) for i,token in enumerate(tile_tokens)]
  if swizzle_stores:
    def values(acc:UOp) -> tuple[UOp, ...]:
      global _reg_swizzle_slot
      own = tuple(acc.after(update)[i].load() for i in range(8))
      # Park peers in REG so emit sees PERMLANEX×8 or SWIZZLE×8 consecutive.
      # ds_swizzle ^16 (HIP-like); permlanex / NOPARK lab knobs removed.
      def peer_op(v:UOp) -> UOp: return _swizzle_f32(v, 16)
      raw = [peer_op(own[i]) for i in range(8)]
      tmp = UOp.placeholder((8,), dtypes.float, slot=_reg_swizzle_slot, addrspace=AddrSpace.REG)
      _reg_swizzle_slot += 1
      tmp = tmp.after(UOp.group(*[tmp[i].store(sw) for i, sw in enumerate(raw)]))
      peer = tuple(tmp[i].load() for i in range(8))
      low = half.eq(0)
      return tuple(low.where(own[i], peer[i+4]) if j == 0 else low.where(peer[i], own[i+4]) for i in range(4) for j in range(2))
    return [out[token, output].store(value) for ot,(output,output_accs) in enumerate(zip(outputs, accs))
            for tile,(tile_tokens,acc) in enumerate(zip(tokens, output_accs)) for token,value in zip(tile_tokens, values(acc))]
  # the accumulator fragment halves are exchanged between lane pairs (l, l^16) through LDS (a ds_swizzle without CUSTOM)
  flat_accs = [acc for output_accs in accs for acc in output_accs]
  lds = UOp.alloc((output_waves, 32, len(flat_accs)*8), dtypes.float32, addrspace=AddrSpace.LOCAL)
  stores = [lds[wave, lane, a*8+i].store(acc.after(update)[i].load()) for a,acc in enumerate(flat_accs) for i in range(8)]
  lds = lds.after(UOp.barrier(UOp.group(*stores)))
  def lds_values(ai:int) -> tuple[UOp, ...]:
    own = tuple(lds[wave, lane, ai*8+i].load() for i in range(8))
    peer = tuple(lds[wave, lane ^ 16, ai*8+i].load() for i in range(8))
    low = half.eq(0)
    return tuple(low.where(own[i], peer[i+4]) if j == 0 else low.where(peer[i], own[i+4]) for i in range(4) for j in range(2))
  return [out[token, output].store(value) for ot,(output,output_accs) in enumerate(zip(outputs, accs))
          for tile,(tile_tokens,_acc) in enumerate(zip(tokens, output_accs)) for token,value in zip(tile_tokens, lds_values(ot*tt+tile))]

def _quant_linear_wmma(out, x, out_features, in_features, type_words, layout, dequant, name, swizzle_stores:bool=False,
                       dequant_halves:Callable[[UOp, UOp], tuple[tuple[UOp, ...], tuple[UOp, ...]]]|None=None, rdna4:bool=False,
                       interleave_wmma:bool=False):
  x = x.reshape(out.shape[0], in_features)
  output_waves, token_block, output_block, lane, wave, physical_half, outputs, input_tokens, tokens = layout
  token_tile, output_tiles = len(tokens)*16, len(outputs)
  output_words = in_features // GGML_BLOCK_SIZE * type_words
  accs = tuple(tuple(UOp.alloc((8,), dtypes.float32, addrspace=AddrSpace.REG)
                     for tile in range(token_tile // 16)) for ot in range(output_tiles))
  accs = tuple(tuple(acc.after(acc.store(acc.const_like(0))) for acc in output_accs) for output_accs in accs)
  group = UOp.range(in_features // Q8_GROUP_SIZE, 4, AxisType.LOOP)
  block, subgroup = group // 8, group % 8
  wmma_accs = [list(output_accs) for output_accs in accs]
  # gfx12 moves k2 from the element index to lane bit 4: each lane supplies two groups of four.
  ks = tuple(i%4 + physical_half*4 + (i//4)*8 for i in range(8)) if rdna4 else tuple(range(16))
  for half in range(2):
    afrags = tuple(UOp.stack(*(x[input_token, group*32 + half*16 + i].cast(dtypes.float16) for i in ks))
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
  stores = _wmma_stores(out, outputs, tokens, accs, update, physical_half, lane, wave, output_waves,
                        swizzle_stores=swizzle_stores, rdna4=rdna4)
  return UOp.group(*stores).end(token_block, output_block, lane, wave).sink(arg=KernelInfo(name=name, opts_to_apply=(),
    schedule_hints=("wmma_interleave",) if interleave_wmma else ()))

@functools.cache
def _q5_linear_f16_wmma_kernel(out:UOp, raw:UOp, x:UOp, out_features:int, in_features:int, ggml_type:int,
                               direct_isa:bool=False, rdna4:bool=False) -> UOp:
  # The direct RDNA3 renderer miscompiles the wider token/output tilings for Q4/Q5 WMMA.
  # Keep the ISA path on the validated 32x1 tile until those layouts are fixed.
  token_tile, output_tiles = ((32 if out.shape[0] % 32 == 0 else 16), 1) if direct_isa else \
    ((64, 1 if out_features <= 1024 else 2) if out.shape[0] % 64 == 0 else (32 if out.shape[0] % 32 == 0 else 16, 2))
  layout = _wmma_layout(out, out_features, token_tile, output_tiles)
  word_indices = (layout[5], layout[5]+2) if rdna4 else tuple(range(4))
  def dequant(base:UOp, subgroup:UOp, half:int) -> tuple[UOp, ...]:
    d, dmin, scale, minimum = _q5_scales(raw, base, subgroup, _amd_load(raw[base], 4, packed_u32=direct_isa))
    qs_base = base + (4 if ggml_type == Q4_K else 12) + (subgroup // 2)*8 + half*4
    qwords = _amd_load(raw[qs_base], 4, packed_u32=direct_isa)
    hwords = _amd_load(raw[base+4+half*4], 4, packed_u32=direct_isa) if ggml_type == Q5_K else None
    words = tuple((qwords[i] >> ((subgroup&1)*4).cast(dtypes.uint32) & 0x0f0f0f0f) |
      (((hwords[i] >> subgroup.cast(dtypes.uint32) & 0x01010101) << 4) if hwords is not None else 0) for i in word_indices)
    if direct_isa:
      scaled, bias = d*scale, dmin*minimum*-1
      return tuple(_amd_fma_to_f16((word >> (byte*8) & 255).float(), scaled, bias) for word in words for byte in range(4))
    return tuple(((word >> (byte*8) & 255).float()*d*scale-dmin*minimum).cast(dtypes.float16) for word in words for byte in range(4))
  return _quant_linear_wmma(out, x, out_features, in_features, QUANT_SIZES[ggml_type]//4,
                            layout, dequant, f"linear_q{4 if ggml_type == Q4_K else 5}_k_f16_wmma",
                            swizzle_stores=direct_isa, rdna4=rdna4)

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

def _q6_wmma_words(ql4:UOp, qh4:UOp, subgroup:UOp, half:int) -> tuple[UOp, ...]:
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
def _q6_linear_f16_wmma_kernel(out:UOp, raw:UOp, x:UOp, out_features:int, in_features:int, direct_isa:bool=False, rdna4:bool=False) -> UOp:
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
                            dequant_halves=dequant_halves if direct_isa else None, rdna4=rdna4)

@functools.cache
def _iq4_linear_f16_wmma_kernel(out:UOp, raw:UOp, x:UOp, lut:UOp, out_features:int, in_features:int,
                                direct_isa:bool=False, rdna4:bool=False) -> UOp:
  token_tile = 32 if out_features <= 1024 and out.shape[0] % 32 == 0 else 64 if out.shape[0] % 64 == 0 and \
    out_features <= 6144 else 128 if out.shape[0] % 128 == 0 else \
    32 if out.shape[0] % 32 == 0 else 16
  # Dual output tiles inflated LUT dequant VGPR live ranges — single tile only.
  output_tiles = 1
  layout = _wmma_layout(out, out_features, token_tile, output_tiles)
  output_waves, _, _, lane, wave, half, _, _, _ = layout
  local_lut = UOp.alloc((256,), dtypes.uint32, addrspace=AddrSpace.LOCAL)
  tid, lut_items = wave*32+lane, 256//(32*output_waves)
  # Packed global→LDS fill (HIP ds_store_b128): load u32×4 then scalar stores; emit folds to B128.
  if direct_isa and lut_items >= 4 and lut_items % 4 == 0:
    stores: list[UOp] = []
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
  def dequant_halves(base:UOp, subgroup:UOp) -> tuple[tuple[UOp, ...], tuple[UOp, ...]]:
    # Share qwords+LUT pairs across both WMMA halves (only mixlo/hi differs).
    d, scale = _iq4_scales(raw, base, subgroup)
    scale = scale * d
    qwords = _amd_load(raw[base+2+subgroup*4], 4, packed_u32=True)
    pairs = tuple(lut[((qwords[word] >> (byte*8)) & 255).cast(dtypes.weakint)]
                  for word in range(4) for byte in range(4))
    return (tuple(_amd_packed_f16_mul_to_f16(pair, scale, 0) for pair in pairs),
            tuple(_amd_packed_f16_mul_to_f16(pair, scale, 1) for pair in pairs))
  return _quant_linear_wmma(out, x, out_features, in_features, QUANT_SIZES[IQ4_XS]//4, layout, dequant, "linear_iq4_xs_f16_wmma",
                            swizzle_stores=direct_isa, dequant_halves=dequant_halves if direct_isa else None, rdna4=rdna4)

@functools.cache
def _quant_linear_f16_wmma_kernel(out:UOp, raw:UOp, x:UOp, *grids:UOp,
                                  out_features:int, in_features:int, ggml_type:int, direct_isa:bool=False,
                                  rdna4:bool=False) -> UOp:
  token_tile = (32 if out.shape[0] % 32 == 0 else 16) if ggml_type == Q3_K else \
    64 if out.shape[0] % 64 == 0 else 32 if out.shape[0] % 32 == 0 else 16
  # On RDNA3, a single Q6 output tile is faster than two for multi-token WMMA.
  output_tiles = 1 if direct_isa and (out_features % 32 or ggml_type in (IQ2_S, IQ3_XXS, IQ4_NL, IQ3_S) or
                                      ggml_type == Q6_K and not rdna4) else 2
  layout = _wmma_layout(out, out_features, token_tile, output_tiles)
  output_waves, _, _, lane, wave, physical_half, _, _, _ = layout
  word_indices = (physical_half, physical_half+2) if rdna4 else tuple(range(4))
  grid = None
  if grids:
    grid = UOp.alloc((int(grids[0].numel()),), dtypes.uint32, addrspace=AddrSpace.LOCAL)
    tid, threads = wave*32+lane, output_waves*32
    grid = grid.after(*(grid[tid+i*threads].store(grids[0][tid+i*threads]) for i in range(int(grid.numel())//threads)))
  def dequant(base:UOp, subgroup:UOp, half:int) -> tuple[UOp, ...]:
    if ggml_type == IQ4_NL: base += subgroup*9  # eight independent 18-byte blocks per 256 weights
    def byte(offset): return _load_byte(raw, base, offset)
    zero, minimum = 0, UOp.const(0, dtypes.float)
    if ggml_type == Q2_K:
      sc = byte(subgroup*2+half)
      scale, minimum = _half(raw[base+40])*(sc & 15).float(), _half(raw[base+41])*(sc >> 4).float()
    elif ggml_type == Q3_K:
      j = subgroup*2+half
      sc = ((byte(96+j%8) >> ((j//8)*4)) & 15) | (((byte(104+j%4) >> ((j//4)*2)) & 3) << 4)
      scale, zero = _half(raw[base+54])*(sc.cast(dtypes.int32)-32).float(), 4
    elif ggml_type == Q6_K:
      scale, zero = _half(raw[base+104])*byte(192+subgroup*2+half).cast(dtypes.uint8).bitcast(dtypes.int8).float(), 32
    else:
      scale = _half(raw[base])
      if ggml_type in (IQ2_XS, IQ2_S):
        sc = byte((66 if ggml_type == IQ2_XS else 74)+subgroup)
        scale *= (((sc >> (half*4)) & 15).float()+0.5)*0.25
      elif ggml_type == IQ3_S: scale *= (1+2*((byte(106+subgroup//2) >> ((subgroup%2)*4)) & 15)).float()
      elif ggml_type == IQ3_XXS: scale *= ((_load_u32(raw, base, 66+subgroup*4) >> 28).float()+0.5)*0.5
    words = tuple(_quant_word(raw, base, subgroup, half*4+i, ggml_type, grid) for i in word_indices)
    return tuple((((word >> (b*8)).cast(dtypes.uint8).bitcast(dtypes.int8).float()-zero)*scale-minimum).cast(dtypes.half)
                 for word in words for b in range(4))
  return _quant_linear_wmma(out, x, out_features, in_features, QUANT_SIZES[ggml_type]//raw.dtype.itemsize,
                            layout, dequant, f"linear_{QUANT_NAMES[ggml_type]}_f16_wmma", swizzle_stores=direct_isa,
                            rdna4=rdna4, interleave_wmma=direct_isa and rdna4 and ggml_type == Q6_K)

def q8_linear(layer:Linear, x:Tensor) -> Tensor:
  assert layer.ggml_type in QUANT_SIZES
  tokens = int(x.numel()) // layer.in_features
  out_features, in_features = layer.out_features, layer.in_features
  out_shape:tuple[int, ...] = (tokens, out_features)
  fxn:Callable[..., UOp]
  extra = (_iq_grid(str(x.device), layer.ggml_type),) if layer.ggml_type in (IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S) else ()
  direct = amd_direct_isa(x.device)
  # The direct IQ4_XS WMMA path returns zero on RDNA3; keep this format correct via quantized decode.
  if tokens % 16 == 0 and out_features % 16 == 0 and not (direct and layer.ggml_type == IQ4_XS):
    rdna4 = _wmma_rdna4(x.device)
    if layer.ggml_type == IQ4_XS:
      fxn = functools.partial(_iq4_linear_f16_wmma_kernel, direct_isa=direct, rdna4=rdna4)
      extra = (iq4_half_lut(str(x.device)),)
    elif layer.ggml_type in (Q4_K, Q5_K):
      fxn = functools.partial(_q5_linear_f16_wmma_kernel, ggml_type=layer.ggml_type, direct_isa=direct, rdna4=rdna4)
    else:
      fxn = functools.partial(_quant_linear_f16_wmma_kernel, ggml_type=layer.ggml_type, direct_isa=direct, rdna4=rdna4)
    srcs = (x.cast(dtypes.float16).contiguous(), *extra)
  else:
    fxn = functools.partial(_quant_decode_kernel, ggml_type=layer.ggml_type)
    srcs = (*q8_quantize(x, tokens, in_features), *extra)
    out_shape += ((in_features+1023)//1024,)
  out = Tensor.empty(out_shape, dtype=dtypes.float32, device=x.device)
  result = Tensor.custom_kernel(out, layer.weight, *srcs, fxn=functools.partial(fxn, out_features=out_features, in_features=in_features))[0]
  if len(result.shape) == 3: result = result.sum(-1)
  result = result.reshape(*x.shape[:-1], out_features)
  return result if layer.bias is None else result + layer.bias

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
  if bias is not None: total = total + bias[out_row].load().float()
  return out[token, out_row.valid(lane.eq(0))].store(total).end(token, out_row, lane).sink(arg=KernelInfo(name="linear_f16_gemv", opts_to_apply=()))

def _view_back(t:Tensor) -> Tensor:
  """Strip only float-to-float casts from a lazy weight, which can be applied by the kernel load."""
  uop = t.uop
  while uop.op is Ops.CAST and dtypes.is_float(uop.dtype) and dtypes.is_float(uop.src[0].dtype): uop = uop.src[0]
  return Tensor(uop).reshape(t.shape)

def f16_gemv(layer:Linear, x:Tensor) -> Tensor:
  tokens = prod(x.shape[:-1])
  assert isinstance(tokens, int)
  weight = _view_back(layer.weight)
  x = x.contiguous() if x.dtype in (dtypes.half, dtypes.bfloat16, dtypes.float) else x.cast(dtypes.half).contiguous()
  out = Tensor.empty(tokens, layer.out_features, dtype=dtypes.float32, device=x.device)
  fxn = functools.partial(_amd_f16_gemv_kernel, in_features=layer.in_features, out_features=layer.out_features, tokens=tokens)
  srcs = (out, weight.reshape(-1), x.reshape(tokens, layer.in_features)) + (() if layer.bias is None else (_view_back(layer.bias),))
  return Tensor.custom_kernel(*srcs, fxn=fxn)[0].reshape(*x.shape[:-1], layer.out_features)

# ******** flash attention on the KV cache ********

def _vec_load(ptr:UOp, lanes:int) -> tuple[UOp, ...]:
  if lanes == 1: return (ptr.load().float(),)
  vec = _amd_load(ptr, lanes)
  return tuple(vec[i].float() for i in range(lanes))

def _vec_load_half(ptr:UOp, lanes:int) -> tuple[UOp, ...]:
  """Keep V as half for v_fma_mix_f32 (HIP flash_decode); cast only at FMAC."""
  if lanes == 1: return (ptr.load().cast(dtypes.half),) if ptr.dtype is not dtypes.half else (ptr.load(),)
  vec = _amd_load(ptr, lanes)
  return tuple(vec[i] if vec[i].dtype is dtypes.half else vec[i].cast(dtypes.half) for i in range(lanes))

@functools.cache
def _amd_flash_attention_decode_partial(out, stats, q, cache_kv, valid_kv_len, max_kv_len, block_n, waves=4):
  valid_kv_len = _unbind(valid_kv_len)
  _, B, H_KV, N, D = cast(tuple[int, int, int, int, int], cache_kv.shape)
  _, H, M, _ = cast(tuple[int, int, int, int], q.shape)
  assert M == 1 and H % H_KV == 0 and D % WARP_SIZE == 0 and max_kv_len <= N and max_kv_len % block_n == 0
  G, CHUNK, DPL, WAVES, PARTIALS = H // H_KV, block_n, D // WARP_SIZE, waves, out.shape[2]
  assert CHUNK % WAVES == 0
  SEC = CHUNK // WAVES  # keys each wave scans independently
  total_chunks = (valid_kv_len+CHUNK-1)//CHUNK
  live_chunks = min(total_chunks, PARTIALS) if isinstance(total_chunks, int) else total_chunks.minimum(PARTIALS)
  block_bhkv, block_chunk = UOp.range(B*H_KV, 0, AxisType.GLOBAL), UOp.range(live_chunks, 1, AxisType.GLOBAL)
  lane, wave = UOp.range(WARP_SIZE, -1, axis_type=AxisType.WARP), UOp.range(WAVES, 3, axis_type=AxisType.LOCAL)
  b, kv_head = block_bhkv // H_KV, block_bhkv % H_KV
  zerof = UOp.const(0, dtypes.float)
  zeroh = UOp.const(0, dtypes.half)
  # Each block scans every PARTIALS-th chunk, keeping an online softmax across rounds.
  chunk_round = UOp.range((total_chunks-1-block_chunk)//PARTIALS+1, 4, AxisType.LOOP)
  chunk_id = block_chunk + chunk_round*PARTIALS
  # Batch four keys so independent reductions share swizzle/permlane stages without over-expanding live state.
  score_batch = 4
  # A finite initial max keeps fully masked waves from computing exp(-inf - -inf).
  acc_reg, max_reg, sum_reg = _reg((G, DPL), 2, 0), _reg((G,), 3, -1e30), _reg((G,), 4, 0)
  prev_acc, prev_max, prev_sum = acc_reg.after(chunk_round), max_reg.after(chunk_round), sum_reg.after(chunk_round)
  # Precompute key validity once (shared across heads).
  valids: list[UOp] = [(chunk_id*CHUNK + wave*SEC + j) < valid_kv_len for j in range(SEC)]
  qf = tuple(_vec_load(q[b, kv_head*G+h, 0, lane*DPL], DPL) for h in range(G))
  scale = 1 / math.sqrt(D)
  scores: list[list[UOp]] = [[zerof]*G for _ in range(SEC)]
  vfrags: list[tuple[UOp, ...]] = [()]*SEC
  for j0 in range(0, SEC, score_batch):
    js = range(j0, min(j0 + score_batch, SEC))
    dots = []
    for j in js:
      key = chunk_id*CHUNK + wave*SEC + j
      kfrag = _vec_load(cache_kv[0, b, kv_head, key, lane*DPL], DPL)
      vfrags[j] = tuple(valids[j].where(v, zeroh) for v in _vec_load_half(cache_kv[1, b, kv_head, key, lane*DPL], DPL))
      dots.extend([sum((qf[h][i]*kfrag[i] for i in range(DPL)), zerof) for h in range(G)])
    for t, s in enumerate(warp_reduce_many(dots, full_wave=True)):
      j, h = j0 + t // G, t % G
      scores[j][h] = valids[j].where(s * scale, UOp.const(-1e30, dtypes.float))
  row_max = [functools.reduce(UOp.maximum, (scores[j][h] for j in range(SEC)), prev_max[h].load()) for h in range(G)]
  alpha = [((prev_max[h].load()-row_max[h])*LOG2E).exp2() for h in range(G)]
  accs = [[alpha[h]*prev_acc[h, i].load() for i in range(DPL)] for h in range(G)]
  row_sums = [alpha[h]*prev_sum[h].load() for h in range(G)]
  for j in range(SEC):
    for h in range(G):
      beta = valids[j].where(((scores[j][h]-row_max[h])*LOG2E).exp2(), zerof)
      accs[h] = [a + beta * v.cast(dtypes.float32) for a, v in zip(accs[h], vfrags[j])]
      row_sums[h] = row_sums[h] + beta
  update = UOp.group(acc_reg.store(UOp.stack(*(x for acc in accs for x in acc)).reshape(G, DPL)),
                     max_reg.store(UOp.stack(*row_max)), sum_reg.store(UOp.stack(*row_sums))).end(chunk_round)
  acc_reg, max_reg, sum_reg = acc_reg.after(update), max_reg.after(update), sum_reg.after(update)
  # exchange across the block's waves through LDS (fp16 halves LDS so more blocks fit per CU)
  # Matching cache/LDS strides can reuse a loop-local cache index outside the loop. Pad that layout.
  # Always pad when G==SEC (WAVES=16 @ block_n=64) — unpadded corrupts wave combine.
  # Note: G==SEC sum drifts ~0.05 vs waves=8 from fp16 LDS×16-wave tree (maxabs ~1e-4), not alias.
  acc_lds = UOp.placeholder((WAVES, G, D + (LDS_PAD if G == SEC else 0)), dtypes.half, slot=0, addrspace=AddrSpace.LOCAL)[:, :, :D]
  # (WAVES, 2, G): avoid G==SEC stride alias that corrupts stats L on AMDRenderer.
  ml_lds = UOp.placeholder((WAVES, 2, G), dtypes.float, slot=1, addrspace=AddrSpace.LOCAL)
  lds_acc = acc_lds.reshape(WAVES, G, WARP_SIZE, DPL)
  # Normalize before fp16 to avoid overflow. Nonempty waves have sum >= 1; empty waves keep their zero accumulator.
  stores = [lds_acc[wave, h, lane].store((acc_reg[h].load() / sum_reg[h].load().maximum(1)).cast(dtypes.half)) for h in range(G)]
  # NOTE: duplicate stores of the same value from every lane are harmless here
  stores += [ml_lds[wave, i, h].store(x) for h in range(G) for i, x in enumerate((max_reg[h].load(), sum_reg[h].load()))]
  barrier = UOp.barrier(UOp.group(*stores))
  acc_lds, ml_lds = acc_lds.after(barrier), ml_lds.after(barrier)
  tid = wave*WARP_SIZE + lane
  final_stores:list[UOp] = []
  for i in range(-(-G*D//(WAVES*WARP_SIZE))):
    flat = tid + i*WAVES*WARP_SIZE
    h, d = flat // D, flat % D
    M = functools.reduce(UOp.maximum, (ml_lds[w, 0, h].load() for w in range(WAVES)))
    # LDS holds normalized values; restore each wave's sum before combining.
    val = sum((((ml_lds[w, 0, h].load()-M)*LOG2E).exp2() * ml_lds[w, 1, h].load() * acc_lds[w, h, d].load().float()
               for w in range(WAVES)), zerof)
    oidx = out[b, kv_head*G + h, block_chunk, d]
    if G*D % (WAVES*WARP_SIZE): oidx = out[b, (kv_head*G + h).valid(flat < G*D), block_chunk, d]
    final_stores.append(oidx.store(val))
  hstat = tid
  M = functools.reduce(UOp.maximum, (ml_lds[w, 0, hstat].load() for w in range(WAVES)))
  L = sum((((ml_lds[w, 0, hstat].load()-M)*LOG2E).exp2() * ml_lds[w, 1, hstat].load() for w in range(WAVES)), zerof)
  q_head = (kv_head*G + hstat).valid(hstat < G) if WAVES*WARP_SIZE > G else kv_head*G + hstat
  final_stores += [stats[b, q_head, block_chunk, 0].store(M), stats[b, q_head, block_chunk, 1].store(L)]
  return UOp.group(*final_stores).end(lane, wave, block_chunk, block_bhkv).sink(arg=KernelInfo(name="flash_decode_partial", opts_to_apply=()))

@functools.cache
def _amd_flash_decode_combine(o:UOp, partial:UOp, stats:UOp, live:int|UOp) -> UOp:
  # one wave per (batch, head, 64-dim tile): every lane redundantly weights its chunks; no cross-lane traffic
  live = _unbind(live)
  B, H, C, D = cast(tuple[int, int, int, int], partial.shape)
  # Dims per block: 64 when D%64==0 else WARP_SIZE (COMBINE_DT removed).
  DT = 64 if D % 64 == 0 else WARP_SIZE
  assert D % DT == 0 and DT % WARP_SIZE == 0
  block_bh, block_dt = UOp.range(B*H, 0, AxisType.GLOBAL), UOp.range(D//DT, 1, AxisType.GLOBAL)
  lane = UOp.range(WARP_SIZE, 2, axis_type=AxisType.LOCAL)
  b, h = block_bh // H, block_bh % H
  NPD = DT // WARP_SIZE  # output dims per lane
  dims = tuple(block_dt*DT + lane*NPD + i for i in range(NPD))
  def _pload(ci, d):
    v = partial[b, h, ci, d].load()
    return v.cast(dtypes.float) if partial.dtype == dtypes.half else v
  # Unroll short combines to avoid a loop over the small set of partials.
  if isinstance(live, int) and live <= 32:
    zerof = UOp.const(0, dtypes.float)
    # Rematerialize stats M loads in the weight loop (keeping ms[] live → vgpr~169).
    chunk_max = functools.reduce(lambda m, ci: UOp.maximum(m, stats[b, h, ci, 0].load()),
                                 range(live), UOp.const(-math.inf, dtypes.float))
    acc = [zerof for _ in range(NPD)]
    weight_sum = zerof
    for ci in range(live):
      w = ((stats[b, h, ci, 0].load() - chunk_max) * LOG2E).exp2()
      acc = [a + w * _pload(ci, d) for a, d in zip(acc, dims)]
      weight_sum = weight_sum + w * stats[b, h, ci, 1].load()
    inv = 1 / weight_sum
    return UOp.group(*[o[b, h, 0, d].store(a * inv) for a, d in zip(acc, dims)]) \
      .end(lane, block_dt, block_bh).sink(arg=KernelInfo(name="flash_decode_combine", opts_to_apply=()))
  chunk = UOp.range(live, 100, AxisType.LOOP)
  def iloop(ph, val): return ph.store(ph.const_like(val))
  chunk_max = UOp.alloc((1,), dtypes.float, addrspace=AddrSpace.REG)
  chunk_max_i = chunk_max.after(iloop(chunk_max, -math.inf))
  update0 = chunk_max_i.store(chunk_max_i.after(chunk).maximum(stats[b, h, chunk, 0].load())).end(chunk)
  chunk_max = chunk_max_i.after(update0)
  chunk2 = UOp.range(live, 101, AxisType.LOOP)
  acc = UOp.alloc((NPD,), dtypes.float, addrspace=AddrSpace.REG)
  weight_sum = UOp.alloc((1,), dtypes.float, addrspace=AddrSpace.REG)
  acc_i, weight_sum_i = acc.after(iloop(acc, 0)), weight_sum.after(iloop(weight_sum, 0))
  w = ((stats[b, h, chunk2, 0].load()-chunk_max)*LOG2E).exp2()
  update1 = UOp.group(*[acc_i[i].store(acc_i.after(chunk2)[i].load() + w*_pload(chunk2, d)) for i, d in enumerate(dims)],
                      weight_sum_i[0].store(weight_sum_i.after(chunk2)[0].load() + w*stats[b, h, chunk2, 1].load())).end(chunk2)
  acc, weight_sum = acc_i.after(update1), weight_sum_i.after(update1)
  inv = 1 / weight_sum[0].load()
  return UOp.group(*[o[b, h, 0, d].store(acc[i].load() * inv) for i, d in enumerate(dims)]) \
    .end(lane, block_dt, block_bh).sink(arg=KernelInfo(name="flash_decode_combine", opts_to_apply=()))

# Proven flash DIRECT compile policy (HW-locked 2026-09). Apply on each build+realize
# scope; prefill clears decode-only keys. Unspecified AMD_* overrides remain available for
# lab A/B. No KEEP_* / opt-in forest in this path.
_FLASH_SHARED = {
  "TC_LDS_AB": "0",              # TC_LDS_AB NaNs hand flash
  "AMD_WMMA_ACC_BASE": "145",    # soft partition (matmul uses 121/128)
  "AMD_WMMA_ACC_N": "48",
  # AMD_LLOAD_POOL default 135 in rdna3_defs (matches ship)
  # AMD_VOPD_PAIR_CONST hardcoded ON in renderer (const/±inf dual_mov bias)
  # AMD_SWIZZLE_BATCH_MAX hardcoded 16 in schedule
}
_FLASH_PREFILL = {
  **_FLASH_SHARED,
  "AMD_FLASH_ACC_SMALL": "1",
  "AMD_REMAT_ADDR": "1",
  "ALLOW_UPCAST16": "0",
  "AMD_WMMA_INTERLEAVE": "1", # Interleave later WMMA loads; OFF regresses the multi-tile DIRECT check.
  # REMAT_CONST_MOV INF-only hardcoded in renderer; PACK_* / REG_PROMOTE_SKIP → ACC_SMALL
  # AMD_WMMA_DELAY hardcoded ON in emit; PACK_LLOAD_ALIGN8 defaults OFF under ACC_SMALL
}
_FLASH_DECODE = {
  **_FLASH_SHARED,
  "ALLOW_UPCAST16": "1",
  # SCHEDULE_VMEM_ALU / REG_PROMOTE_SKIP / PACK_* folded into VOPD_FMAC / ACC_SMALL
  # MUL/FMAC_DELAY + MOV_ALIAS_SWIZZLE + FMA_MIX_ALL + SWIZZLE_VALU_GAP → VOPD_FMAC
  # SCHEDULE_FMAC_ADD → VOPD_FMAC_ALU
  "AMD_VOPD_FMAC": "1",
  "AMD_VOPD_FMAC_ALU": "1",
}
_FLASH_DECODE_ONLY = (
  "AMD_VOPD_FMAC", "AMD_VOPD_FMAC_ALU",
)

def _flash_apply_env(policy:dict[str, str]) -> None:
  for k, v in policy.items():
    if os.environ.get(k) is None: os.environ[k] = v

def _flash_clear_decode_policy() -> None:
  # Decode scheduling is process-global; don't let it leak into generic SDPA fallback kernels.
  for k in _FLASH_DECODE_ONLY: os.environ.pop(k, None)
  getenv.cache_clear()  # type: ignore[attr-defined]

@contextlib.contextmanager
def _flash_direct_compile_env(*, flash_acc_small:bool=True):
  """Apply ship flash DIRECT compile policy for build+realize."""
  prev = os.environ.get("AMD_FLASH_ACC_SMALL")
  prev_remat = os.environ.get("AMD_REMAT_ADDR")
  prev_interleave = os.environ.get("AMD_WMMA_INTERLEAVE")
  prev_decode = {k:os.environ.get(k) for k in _FLASH_DECODE_ONLY}
  if flash_acc_small:
    for k in _FLASH_DECODE_ONLY: os.environ.pop(k, None)
    _flash_apply_env(_FLASH_PREFILL)
    # Force path flips (must overwrite sticky decode / matmul / prior ACC_SMALL).
    os.environ.update({
      "TC_LDS_AB": "0", "AMD_FLASH_ACC_SMALL": "1",
      "AMD_WMMA_ACC_BASE": "145", "AMD_WMMA_ACC_N": "48",
      "ALLOW_UPCAST16": "0",  # flash prefill ship (ALLOW_UPCAST16 meta knobs removed)
    })
  else:
    _flash_apply_env(_FLASH_DECODE)
    os.environ.update({
      "TC_LDS_AB": "0",
      "AMD_WMMA_ACC_BASE": "145", "AMD_WMMA_ACC_N": "48",
      "ALLOW_UPCAST16": "1",  # flash decode ship
    })
  getenv.cache_clear()  # type: ignore[attr-defined]
  try:
    yield
  finally:
    if flash_acc_small:
      if prev is None: os.environ.pop("AMD_FLASH_ACC_SMALL", None)
      else: os.environ["AMD_FLASH_ACC_SMALL"] = prev
      if prev_remat is None: os.environ.pop("AMD_REMAT_ADDR", None)
      else: os.environ["AMD_REMAT_ADDR"] = prev_remat
      if prev_interleave is None: os.environ.pop("AMD_WMMA_INTERLEAVE", None)
      else: os.environ["AMD_WMMA_INTERLEAVE"] = prev_interleave
    for k, v in prev_decode.items():
      if v is None: os.environ.pop(k, None)
      else: os.environ[k] = v
    getenv.cache_clear()  # type: ignore[attr-defined]

def _flash_direct_realize(out:Tensor, *, flash_acc_small:bool=True) -> Tensor:
  """Realize under flash compile envs (ACC_SMALL remat). Prefer wrapping build+realize."""
  with _flash_direct_compile_env(flash_acc_small=flash_acc_small):
    return out.realize()

def amd_flash_attention_decode(q:Tensor, cache_kv:Tensor, valid_kv_len:int|UOp, max_kv_len:int) -> Tensor:
  # Carry length bindings even when the cache was populated independently of this call.
  if isinstance(valid_kv_len, UOp): cache_kv = Tensor(cache_kv.uop.after(valid_kv_len))
  B, H, D = cache_kv.shape[1], q.shape[1], cache_kv.shape[4]
  group = H // cache_kv.shape[2]
  # Large GQA groups overflow the direct kernel's per-head score/softmax state. Split heads
  # into independent launches; this keeps the common G<=4 path fused and correct.
  if amd_direct_isa(q.device) and (group >= 8 or (D >= 256 and group >= 4 and isinstance(valid_kv_len, UOp))):
    first = amd_flash_attention_decode(q[:, :1], cache_kv[:, :, :1], valid_kv_len, max_kv_len)
    return first.cat(*(amd_flash_attention_decode(q[:, h:h+1], cache_kv[:, :, h//group:h//group+1],
                         valid_kv_len, max_kv_len) for h in range(1, H)), dim=1)
  # Split key/value context into 64-token chunks.
  block_n = 64  # BLOCK_N hardcoded
  assert max_kv_len % block_n == 0, f"max_kv_len={max_kv_len} must divide block_n={block_n}"
  chunks = max_kv_len // block_n
  # PARTIAL_HALF: store normalized partials as f16 (less combine BW); cast up in combine.
  pdt = "float32"
  partial = Tensor.empty(B, H, chunks, D, dtype=pdt, device=q.device)
  stats = Tensor.empty(B, H, chunks, 2, dtype="float32", device=q.device)
  # Pad acc LDS rows when G==SEC to avoid aliasing stats LDS on AMDRenderer.
  group = H // cache_kv.shape[2]
  # 16 waves give incorrect long-context graph replays for the common 4:1 GQA case.
  waves = 8
  assert block_n % waves == 0
  while waves * group * ((D+LDS_PAD)*2 + 8) > 65536: waves //= 2
  assert waves > 0, "attention head group exceeds shared memory capacity"
  fxn = functools.partial(_amd_flash_attention_decode_partial, valid_kv_len=valid_kv_len, max_kv_len=max_kv_len,
                          block_n=block_n, waves=waves)
  partial, stats = Tensor.custom_kernel(partial, stats, q, cache_kv, fxn=fxn)[:2]
  live = (valid_kv_len + block_n - 1) // block_n
  live = min(live, chunks) if isinstance(live, int) else live.minimum(chunks)
  out = Tensor.empty(B, H, 1, D, dtype="float32", device=q.device)
  fxn = functools.partial(_amd_flash_decode_combine, live=live)
  return Tensor.custom_kernel(out, partial, stats, fxn=fxn)[0]

def _wmma_fragment(fragment:UOp, lane:UOp, rdna4:bool) -> UOp:
  return fragment.reshape(2, 2, 4)[:, lane//16, :].reshape(8) if rdna4 else fragment

@functools.cache
def _amd_flash_attention(o:UOp, q:UOp, cache:UOp, valid_kv_len:int|UOp, q_start:int|UOp|None=None,
                         acc_small:bool=False, k_unroll:int=0, use_acc_work:bool=False,
                         k_hip_scope:str="all", k_midstore:int|None=None,
                         acc_vgpr:bool=False, hoist_q:bool=False,
                         phase_dumps: tuple[tuple[str, UOp], ...]=()) -> UOp:
  """Hand flash-attn. Optional phase_dumps=(("qk", buf), ...) for diagnostics.

  Dump buffers use *logical* tile coords so each slot has one writer:
    q_lds:   (BH, M//BLOCK_M, Ntiles, BLOCK_M, D) — Q LDS after qk_load_barrier; unique (qrow,d) writers
    k_lds:   (BH, M//BLOCK_M, Ntiles, BLOCK_N, D) — K LDS after qk_load_barrier; unique (krow,d) writers
    qk_wmma:(BH, M//BLOCK_M, Ntiles, WAVES_N, BLOCK_M, BLOCK_N) — raw QK after qk_done (slot 6 / per wave)
    s_soft:  (BH, M//BLOCK_M, Ntiles, WAVES_N, BLOCK_M, BLOCK_N) — after ACC→S_soft (+scale/mask); before exp
    qk:      (BH, M//BLOCK_M, Ntiles, BLOCK_M, BLOCK_N) — scaled+masked S; writer wave_n==0 (legacy)
    soft_m:  (BH, M//BLOCK_M, Ntiles, BLOCK_M) — softmax max; writer wave_n==0 & lane_n==0
    soft_l:  (BH, M//BLOCK_M, Ntiles, BLOCK_M) — softmax sum; writer wave_n==0 & lane_n==0
    p_reg:   (BH, M//BLOCK_M, Ntiles, WAVES_N, BLOCK_M, BLOCK_N) — after softmax exp (per wave)
    p_lds:   (BH, M//BLOCK_M, Ntiles, WAVES_N, BLOCK_M, BLOCK_N) — P after pv_barrier (per wave_n slice)
    v_lds:   (BH, M//BLOCK_M, Ntiles, D, BLOCK_N) — V after pv_barrier; unique (d,n) writers
    pv_wmma: (BH, M//BLOCK_M, Ntiles, BLOCK_M, D) — PV *before* ACC→pv_soft copy (slot 10)
    pv:      (BH, M//BLOCK_M, Ntiles, BLOCK_M, D) — PV *after* copy (slot 17 / current reader)
    acc:     (BH, M//BLOCK_M, BLOCK_M, D) — pre-norm acc; all threads unique (qrow,dcol)
  Ntiles = ceil(S/BLOCK_N) when S is int. Probe wave_n=1 with qk_wmma → s_soft → p_reg.
  Shared Q/K LDS dumps rule out tile data before blaming wave-local ACC/VGPR/spill.
  """
  dumps = dict(phase_dumps)
  valid_kv_len, q_start = _unbind(valid_kv_len), _unbind(q_start) if q_start is not None else None
  BH, M, D = q.shape
  _, B, H_KV, physical_n, cache_dim = cache.shape
  k, v = cache[0].reshape(B*H_KV, physical_n, cache_dim), cache[1].reshape(B*H_KV, physical_n, cache_dim)
  assert k.shape == v.shape and BH % k.shape[0] == 0 and k.shape[2] == D
  gqa_group = BH // k.shape[0]
  if isinstance(M, int): assert M % BLOCK_M == 0
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
  Q_lds = UOp.placeholder((BLOCK_M, D + LDS_PAD), dtypes.half, slot=0, addrspace=AddrSpace.LOCAL)[:, :D]
  KV_lds = UOp.placeholder((BLOCK_N, D + LDS_PAD), dtypes.half, slot=1, addrspace=AddrSpace.LOCAL)[:, :D]
  # ACC_VGPR: park REDUCE ACC in promotable slot 19 (no scratch / no ACC_WORK).
  # ACC_LDS removed (never shipped; REG slot 2 + SKIP=2 is the work path).
  if acc_vgpr:
    acc = _reg((TM, TD), 19, 0)
  else:
    acc = _reg((TM, TD), 2, 0)
  m_i, l_i = _reg((TM,), 3, -math.inf), _reg((TM,), 4, 0)
  # AMD_FLASH_HOIST_Q: Q LDS (slot 0) is never overwritten by K/V/P — load once before
  # the n_tile REDUCE instead of reloading every KV tile (~64× for S=2048).
  _hoist_q = bool(hoist_q)
  Q_hoisted = None
  if _hoist_q:
    Q_hoisted = Q_lds.reshape(THREADS_PER_BLOCK, Q_ELEMS_PER_THREAD)[tid].store(
      q.reshape(THREADS_PER_BLOCK, Q_ELEMS_PER_THREAD)[tid])
    Q_lds = Q_lds.after(UOp.barrier(Q_hoisted))
  n_tile = UOp.range((q_base + (block_m + 1) * BLOCK_M + BLOCK_N - 1) // BLOCK_N, 100, AxisType.LOOP)
  if _hoist_q:
    assert Q_hoisted is not None
    # Unroll K global→LDS (HIP-like static gload count). Opt out: AMD_FLASH_UNROLL_KV=0.
    k_flat = k.reshape(physical_n * D)
    k_dst = KV_lds.after(n_tile).reshape(THREADS_PER_BLOCK, KV_ELEMS_PER_THREAD)
    k_stores = []
    for ei in range(KV_ELEMS_PER_THREAD):
      flat = tid * KV_ELEMS_PER_THREAD + ei
      kval = k_flat[n_tile * BLOCK_N * D + flat]
      kval = (n_tile * BLOCK_N + flat // D < valid_kv_len).where(kval, kval.const_like(0))
      k_stores.append(k_dst[tid, ei].store(kval))
    K_store = UOp.group(*k_stores)
    qk_load_barrier = UOp.barrier(K_store)
    Q_lds, KV_lds_k = Q_lds.after(qk_load_barrier), KV_lds.after(qk_load_barrier)
  else:
    Q_store = Q_lds.after(n_tile).reshape(THREADS_PER_BLOCK, Q_ELEMS_PER_THREAD)[tid].store(q.reshape(THREADS_PER_BLOCK, Q_ELEMS_PER_THREAD)[tid])
    k_flat = k.reshape(physical_n * D)
    k_dst = KV_lds.reshape(THREADS_PER_BLOCK, KV_ELEMS_PER_THREAD)
    k_stores = []
    for ei in range(KV_ELEMS_PER_THREAD):
      flat = tid * KV_ELEMS_PER_THREAD + ei
      kval = k_flat[n_tile * BLOCK_N * D + flat]
      kval = (n_tile * BLOCK_N + flat // D < valid_kv_len).where(kval, kval.const_like(0))
      k_stores.append(k_dst[tid, ei].store(kval))
    K_store = UOp.group(*k_stores)
    qk_load_barrier = UOp.barrier(UOp.group(Q_store, K_store))
    Q_lds, KV_lds_k = Q_lds.after(qk_load_barrier), KV_lds.after(qk_load_barrier)
  # Diagnostic: shared Q/K LDS after barrier (pre-WMMA inputs; expect exact across launches).
  if dumps and "q_lds" in dumps:
    dq = dumps["q_lds"]  # (BH, Nm, Ntiles, BLOCK_M, D) — unique (qrow, dcol) per thread
    q_lds_stores = []
    for ei in range(Q_ELEMS_PER_THREAD):
      flat = tid * Q_ELEMS_PER_THREAD + ei
      qrow, dcol = flat // D, flat % D
      q_lds_stores.append(dq[block_bh, block_m, n_tile, qrow, dcol].store(Q_lds[qrow, dcol].float()))
    Q_lds = Q_lds.after(UOp.group(*q_lds_stores))
  if dumps and "k_lds" in dumps:
    dk = dumps["k_lds"]  # (BH, Nm, Ntiles, BLOCK_N, D) — unique (krow, dcol) per thread
    k_lds_stores = []
    for ei in range(KV_ELEMS_PER_THREAD):
      flat = tid * KV_ELEMS_PER_THREAD + ei
      krow, dcol = flat // D, flat % D
      k_lds_stores.append(dk[block_bh, block_m, n_tile, krow, dcol].store(KV_lds_k[krow, dcol].float()))
    KV_lds_k = KV_lds_k.after(UOp.group(*k_lds_stores))
  S_reg = _reg((TM, TN), 6, 0, n_tile)
  # AMD_FLASH_UNROLL: 0=off; 1 or 6=soft+corr python-unroll (const REG slots). WMMA stays ranged —
  # frag.after(store) cin was wrong; const_like(0) chains break direct-ISA emit (half2 LDS).
  # ACC_SEP hardcoded: keep WMMA C in ACC; copy out to const-indexed soft before softmax.
  # Park one ACC pack per column (python-unrolled WMMA). Flash-only — AMD_WMMA_ACC_SMALL
  # also parks ≤64 quant tiles and breaks Q4/Q6; prefer AMD_FLASH_ACC_SMALL / acc_small arg.
  _acc_small = bool(acc_small or getenv("AMD_WMMA_ACC_SMALL", 0) or getenv("AMD_FLASH_ACC_SMALL", 0))
  _fu = 6  # FLASH_UNROLL hardcoded: soft+corr const REG indices
  S_view = S_reg.reshape(TM // WMMA_ACC, WMMA_ACC, TN).permute(0, 2, 1)
  Q_view = Q_lds.reshape(WAVES_M, TM // WMMA_ACC, WMMA_M, D // WMMA_K, WMMA_K)
  K_view = KV_lds_k.reshape(TN, WMMA_N, D // WMMA_K, WMMA_K)
  # AMD_FLASH_K_UNROLL: 0=ranged; N>=2 chained (default 2 w/ ACC_SMALL+UPCAST16=0);
  # 1=full chain (MMU); -1=HIP-style store-each-K; 8 MMUs. Scope via k_hip_scope.
  # AMD_FLASH_K_MIDSTORE=N: in full/hip QK|PV chains, store ACC every N WMMA (breaks MMU at N=4).
  _k_factor = int(k_unroll) if _acc_small else 0
  _k_hip = _k_factor == -1
  _hip_scope = (k_hip_scope or "all").lower() if _k_hip or _k_factor >= 2 else ""
  _k_hip_qk = _k_hip and _hip_scope in ("all", "qk", "q")
  _k_hip_pv = _k_hip and _hip_scope in ("all", "pv", "v")
  _k_unroll_full = _k_factor == 1
  _k_unroll_factor = _k_factor if _k_factor >= 2 and (D // WMMA_K) % _k_factor == 0 else 0
  # Mid-store every 4 WMMA. MIDSTORE=8 is unstable on Navi31 multi-tile prefill.
  _mid_default = 4 if (_k_unroll_full or _k_hip or _k_factor >= 8) else 0
  if not _acc_small: _mid_every = 0
  elif k_midstore is None: _mid_every = _mid_default
  else: _mid_every = k_midstore
  # Factor unroll can be scoped like hip: qk|pv|all (default all when scope unset).
  _factor_scope = (k_hip_scope or "all").lower() if _k_unroll_factor else ""
  _factor_qk = bool(_k_unroll_factor) and _factor_scope in ("all", "qk", "q", "")
  _factor_pv = bool(_k_unroll_factor) and _factor_scope in ("all", "pv", "v", "")
  if not _k_unroll_factor: _factor_qk = _factor_pv = False
  if _factor_scope in ("qk", "q"): _factor_pv = False
  if _factor_scope in ("pv", "v"): _factor_qk = False
  # Factor-8 with outer range of 1 is a full WMMA chain (MMU). Prefer hip-style + midstore.
  if _mid_every and _k_unroll_factor >= 8:
    if _factor_qk: _k_hip_qk = True
    if _factor_pv: _k_hip_pv = True
    _k_unroll_factor = 0
    _factor_qk = _factor_pv = False
  if _acc_small and _k_hip_qk:
    # HIP-style: chain all K in ACC, one store per (tm,tn) — not store-reload each K.
    qk_stores = []
    for tn_i in range(TN):
      for tm_i in range(TM // WMMA_ACC):
        S_frag = S_view[tm_i, tn_i]
        wmma_c = S_frag
        for k_i in range(D // WMMA_K):
          q_frag = Q_view[wave_m, tm_i, lane_n, k_i]
          k_frag = K_view[tn_i, lane_n, k_i]
          wmma_c = UOp.wmma(q_frag, k_frag, wmma_c, *WMMA_ARG)
          if _mid_every and (k_i + 1) % _mid_every == 0 and k_i + 1 < D // WMMA_K:
            wmma_c = S_frag.after(S_frag.store(wmma_c))
        qk_stores.append(S_frag.store(wmma_c))
    qk_done = UOp.group(*qk_stores)
  elif _acc_small and _k_unroll_full:
    qk_stores = []
    for tn_i in range(TN):
      for tm_i in range(TM // WMMA_ACC):
        S_frag = S_view[tm_i, tn_i]
        wmma_c = S_frag  # parked ACC cin (do not shadow output `acc`)
        for k_i in range(D // WMMA_K):
          q_frag = Q_view[wave_m, tm_i, lane_n, k_i]
          k_frag = K_view[tn_i, lane_n, k_i]
          wmma_c = UOp.wmma(q_frag, k_frag, wmma_c, *WMMA_ARG)
          if _mid_every and (k_i + 1) % _mid_every == 0 and k_i + 1 < D // WMMA_K:
            wmma_c = S_frag.after(S_frag.store(wmma_c))
        qk_stores.append(S_frag.store(wmma_c))
    qk_done = UOp.group(*qk_stores)
  elif _acc_small and _factor_qk:
    k_outer = UOp.range((D // WMMA_K) // _k_unroll_factor, 101, AxisType.LOOP)
    qk_stores = []
    for tn_i in range(TN):
      for tm_i in range(TM // WMMA_ACC):
        S_frag = S_view[tm_i, tn_i]
        wmma_c = S_frag.after(k_outer)
        for k_off in range(_k_unroll_factor):
          k_i = k_outer * _k_unroll_factor + k_off
          q_frag = Q_view[wave_m, tm_i, lane_n, k_i]
          k_frag = K_view[tn_i, lane_n, k_i]
          wmma_c = UOp.wmma(q_frag, k_frag, wmma_c, *WMMA_ARG)
        qk_stores.append(S_frag.store(wmma_c))
    qk_done = UOp.group(*qk_stores).end(k_outer)
  elif _acc_small:
    k_qk = UOp.range(D//WMMA_K, 101, AxisType.LOOP)
    qk_stores = []
    for tn_i in range(TN):
      for tm_i in range(TM // WMMA_ACC):
        S_frag = S_view[tm_i, tn_i]
        q_frag = Q_view[wave_m, tm_i, lane_n, k_qk]
        k_frag = K_view[tn_i, lane_n, k_qk]
        qk_stores.append(S_frag.store(UOp.wmma(q_frag, k_frag, S_frag.after(k_qk), *WMMA_ARG)))
    qk_done = UOp.group(*qk_stores).end(k_qk)
  else:
    k_qk, tm1, tn1 = UOp.range(D//WMMA_K, 101, AxisType.LOOP), UOp.range(TM//WMMA_ACC, 200), UOp.range(TN, 201)
    S_frag = S_view[tm1, tn1]
    q_frag = Q_view[wave_m, tm1, lane_n, k_qk]
    k_frag = K_view[tn1, lane_n, k_qk]
    qk_done = S_frag.store(UOp.wmma(q_frag, k_frag, S_frag.after(k_qk), *WMMA_ARG)).end(tm1, tn1).end(k_qk)
  # Scale+mask: ACC_SEP + SOFT_SCALE + SOFT_FUSE hardcoded on (ship).
  S_masked = S_reg.after(qk_done)
  # Diagnostic: raw QK WMMA C (slot 6) per wave_n, immediately after qk_done (before ACC→soft).
  if dumps and "qk_wmma" in dumps:
    dq = dumps["qk_wmma"]  # (BH, Nm, Ntiles, WAVES_N, BLOCK_M, BLOCK_N)
    qk_wmma_stores = []
    for ri in range(TM):
      for rj in range(TN):
        qrow = wave_m * WMMA_M + ri * LANES_PER_WAVE_M + lane_m
        kcol = rj * LANES_PER_WAVE_N + lane_n
        qk_wmma_stores.append(dq[block_bh, block_m, n_tile, wave_n, qrow, kcol].store(S_masked[ri, rj]))
    S_masked = S_masked.after(UOp.group(*qk_wmma_stores))
  # Copy into const-index soft buffer; scale+mask in one SSTORE wave (SOFT_FUSE).
  S_soft = UOp.placeholder((TM, TN), dtypes.float, slot=16, addrspace=AddrSpace.REG)
  sm_stores = []
  for rm_i in range(TM):
    for rn_i in range(TN):
      q_idx = q_base + block_m * BLOCK_M + wave_m * WMMA_M + rm_i * LANES_PER_WAVE_M + lane_m
      k_idx = n_tile * BLOCK_N + rn_i * LANES_PER_WAVE_N + lane_n
      scaled = S_masked[rm_i, rn_i] * SCALE
      sm_stores.append(S_soft[rm_i, rn_i].store((k_idx <= q_idx).where(scaled, scaled.const_like(-math.inf))))
  S_reg = S_soft.after(UOp.group(*sm_stores))
  # Diagnostic: S_soft (slot 16) after ACC→soft (+scale/mask); before max/sum/exp. Per wave_n.
  if dumps and "s_soft" in dumps:
    ds = dumps["s_soft"]  # (BH, Nm, Ntiles, WAVES_N, BLOCK_M, BLOCK_N)
    s_soft_stores = []
    for ri in range(TM):
      for rj in range(TN):
        qrow = wave_m * WMMA_M + ri * LANES_PER_WAVE_M + lane_m
        kcol = rj * LANES_PER_WAVE_N + lane_n
        s_soft_stores.append(ds[block_bh, block_m, n_tile, wave_n, qrow, kcol].store(S_reg[ri, rj]))
    S_reg = S_reg.after(UOp.group(*s_soft_stores))
  # Legacy: scaled+masked QK, wave_n==0 only (does not prove wave_n=1).
  if dumps and "qk" in dumps:
    dq = dumps["qk"]  # (BH, Nm, Ntiles, BLOCK_M, BLOCK_N)
    qk_stores = []
    for ri in range(TM):
      for rj in range(TN):
        qrow = wave_m * WMMA_M + ri * LANES_PER_WAVE_M + lane_m
        kcol = rj * LANES_PER_WAVE_N + lane_n
        # Gate on an index axis (not INDEX.valid) so STORE stays INDEX+value.
        qk_stores.append(dq[block_bh, block_m, n_tile, qrow, kcol.valid(wave_n.eq(0))].store(S_reg[ri, rj]))
    S_reg = S_reg.after(UOp.group(*qk_stores))
  if _fu & 2:
    m_ij = _reg((TM,), 7, -math.inf, n_tile)
    for rn_i in range(TN):
      m_ij = m_ij.after(m_ij.store(m_ij.maximum(S_reg[:, rn_i])))
    # Batch TM keys so same-stage ds_swizzle share one lgkm wait (HIP pattern).
    _m_red = warp_reduce_many([m_ij[ri] for ri in range(TM)], maximum=True)
    m_ij = m_ij.after(UOp.group(*[m_ij[ri].store(_m_red[ri]) for ri in range(TM)]))
    tile_max = m_ij.reshape(TM, 1).expand(TM, TN).maximum(-1e30)
    S_reg = S_reg.after(S_reg.store(((S_reg - tile_max) * LOG2E).exp2()))
    p_local = _reg((TM,), 8, 0, n_tile)
    _p_vals = [S_reg[ri, rn_i] for ri in range(TM) for rn_i in range(TN)]
    _p_red = warp_reduce_many(_p_vals)
    p_sum = p_local.after(UOp.group(*[
      p_local[ri].store(sum((_p_red[ri * TN + rn_i] for rn_i in range(TN)), S_reg.const_like(0)))
      for ri in range(TM)]))
  else:
    m_ij, rm2 = _reg((TM,), 7, -math.inf, n_tile), UOp.range(TN, 261, AxisType.LOOP)
    m_ij = m_ij.after(m_ij.store(m_ij.after(rm2).maximum(S_reg[:, rm2])).end(rm2))
    # Ranged soft path: still batch the TM max reduces when the range is unrolled away.
    _m_red = warp_reduce_many([m_ij[ri] for ri in range(TM)], maximum=True)
    m_ij = m_ij.after(UOp.group(*[m_ij[ri].store(_m_red[ri]) for ri in range(TM)]))
    tile_max = m_ij.reshape(TM, 1).expand(TM, TN).maximum(-1e30)
    S_reg = S_reg.after(S_reg.store(((S_reg - tile_max) * LOG2E).exp2()))
    p_local = _reg((TM,), 8, 0, n_tile)
    _p_vals = [S_reg[ri, rn] for ri in range(TM) for rn in range(TN)]
    _p_red = warp_reduce_many(_p_vals)
    p_sum = p_local.after(UOp.group(*[
      p_local[ri].store(sum((_p_red[ri * TN + rn] for rn in range(TN)), S_reg.const_like(0)))
      for ri in range(TM)]))
  # Diagnostic: softmax max/sum → logical BLOCK_M rows. One writer per row (wave_n=0, lane_n=0).
  if dumps and any(k in dumps for k in ("soft_m", "soft_l")):
    soft_stores: list[UOp] = []
    writer = wave_n.eq(0) & lane_n.eq(0)
    for ri in range(TM):
      qrow = wave_m * WMMA_M + ri * LANES_PER_WAVE_M + lane_m
      if "soft_m" in dumps:
        soft_stores.append(dumps["soft_m"][block_bh, block_m, n_tile, qrow.valid(writer)].store(m_ij[ri]))
      if "soft_l" in dumps:
        soft_stores.append(dumps["soft_l"][block_bh, block_m, n_tile, qrow.valid(writer)].store(p_sum[ri]))
    gate = UOp.group(*soft_stores)
    S_reg, m_ij, p_sum = S_reg.after(gate), m_ij.after(gate), p_sum.after(gate)
  # Diagnostic: softmax P in REG before LDS write (per wave_n — waves should match).
  if dumps and "p_reg" in dumps:
    dp = dumps["p_reg"]  # (BH, Nm, Ntiles, WAVES_N, BLOCK_M, BLOCK_N)
    p_reg_stores = []
    for ri in range(TM):
      for rj in range(TN):
        qrow = wave_m * WMMA_M + ri * LANES_PER_WAVE_M + lane_m
        kcol = rj * LANES_PER_WAVE_N + lane_n
        p_reg_stores.append(dp[block_bh, block_m, n_tile, wave_n, qrow, kcol].store(S_reg[ri, rj]))
    S_reg = S_reg.after(UOp.group(*p_reg_stores))
  # QK reads Q from Q_lds and K from KV_lds. P used to reuse Q's LDS buffer and V reuses
  # KV_lds (slot 1). after(qk_done) on V was per-wave only — early waves could V-store over K
  # while peers were still in the K-loop. Dedicated P removes Q/P aliasing; workgroup
  # barrier on qk_done gates V until every wave finishes QK.
  lds_reuse_barrier = UOp.barrier(qk_done)
  P_lds = UOp.placeholder((WAVES_N, BLOCK_M, BLOCK_N), dtypes.half, slot=5, addrspace=AddrSpace.LOCAL)
  P_write = P_lds.reshape(WAVES_N, WAVES_M, TM, LANES_PER_WAVE_M, 1, TN, LANES_PER_WAVE_N, 1).permute((1, 0, 3, 6, 2, 4, 5, 7)) \
    .reshape(THREADS_PER_BLOCK, TM, TN)
  P_store = P_write[tid].store(S_reg.cast(dtypes.half))
  beta_i = UOp.placeholder((TM,), dtypes.float, slot=9, addrspace=AddrSpace.REG)
  # Tile-local promotable working copy of REDUCE-carried acc when slot 2 stays scratch
  # (or ACC is in LDS). Never combine REG-promoted ACC with the work copy — wrong checksums.
  # One scratch/LDS load at tile start + one store at tile end; alpha*acc still runs before
  # V load (keeps latency hiding). Soft/stats buffers already promote on their own.
  # EARLY_V hardcoded on with ACC_SMALL (ship prefill).
  _early_v = bool(_acc_small)
  _acc_work = bool(use_acc_work and (_fu & 4) and _acc_small)
  acc_work = UOp.placeholder((TM, TD), dtypes.float, slot=19, addrspace=AddrSpace.REG) if _acc_work else None
  # Slot 1 reuses K's LDS after QK (OVERLAP_V dedicated slot-7 removed).
  V_lds = UOp.placeholder((D, BLOCK_N + LDS_PAD), dtypes.half, slot=1,
                          addrspace=AddrSpace.LOCAL)[:, :BLOCK_N]
  V_store = None
  def _kv_ld(idx:UOp) -> UOp:
    return idx  # half→half LDS (KV_FLOAT_COPY removed)
  def _kv_zero(sample:UOp) -> UOp:
    return sample.const_like(0)
  def _v_store_unrolled(v_lds_view:UOp) -> UOp:
    v_flat = v.reshape(physical_n * D)
    v_dst = v_lds_view.permute(1, 0).reshape(THREADS_PER_BLOCK, KV_ELEMS_PER_THREAD)
    stores = []
    for ei in range(KV_ELEMS_PER_THREAD):
      flat = tid * KV_ELEMS_PER_THREAD + ei
      v_pos = n_tile * BLOCK_N + flat // D
      vraw = v_flat[n_tile * BLOCK_N * D + flat]
      vval = (v_pos < valid_kv_len).where(_kv_ld(vraw), _kv_zero(vraw))
      stores.append(v_dst[tid, ei].store(vval))
      if dumps and "v_global" in dumps:
        stores.append(dumps["v_global"][block_bh, block_m, n_tile, flat % D, flat // D].store(vval.float()))
    return UOp.group(*stores)
  if _early_v:
    V_store = _v_store_unrolled(V_lds.after(lds_reuse_barrier))
  if _fu & 4:
    acc_c, l_c, m_c = acc.after(n_tile), l_i.after(n_tile), m_i.after(n_tile)
    if _acc_work:
      assert acc_work is not None
      # float4 copy: fewer scratch ops than 32 scalar SLOAD/SSTORE (promoted dst still VGPR).
      if (TM * TD) % 4 == 0:
        src, dst = acc_c.reshape(TM * TD), acc_work.reshape(TM * TD)
        acc_work = acc_work.after(UOp.group(*[dst[i:i+4].store(src[i:i+4]) for i in range(0, TM * TD, 4)]))
      else:
        acc_work = acc_work.after(UOp.group(*[acc_work[ri, rj].store(acc_c[ri, rj]) for ri in range(TM) for rj in range(TD)]))
      work = acc_work
    else:
      work = acc_c
    corr = []
    for ri in range(TM):
      m_new = m_c[ri].maximum(m_ij[ri])
      alpha_val, beta_val = ((m_c[ri] - m_new) * LOG2E).exp2(), ((m_ij[ri] - m_new) * LOG2E).exp2()
      corr.append(UOp.group(*[work[ri, rj].store(alpha_val * work[ri, rj]) for rj in range(TD)],
                           l_c[ri].store(alpha_val * l_c[ri] + beta_val * p_sum[ri]),
                           m_c[ri].store(m_new), beta_i[ri].store(beta_val)))
    correction = UOp.group(*corr)
    if _acc_work:
      acc_work = work.after(correction)
      acc, l_i, m_i, beta_i = acc.after(correction), l_i.after(correction), m_i.after(correction), beta_i.after(correction)
    else:
      acc, l_i, m_i, beta_i = acc.after(correction), l_i.after(correction), m_i.after(correction), beta_i.after(correction)
  else:
    ri4, rj4 = UOp.range(TM, 330), UOp.range(TD, 331)
    m_new = m_i[ri4].maximum(m_ij[ri4])
    alpha_val, beta_val = ((m_i[ri4] - m_new) * LOG2E).exp2(), ((m_ij[ri4] - m_new) * LOG2E).exp2()
    correction = UOp.group(acc[ri4, rj4].store(alpha_val * acc[ri4, rj4]).end(rj4),
                           l_i[ri4].store(alpha_val * l_i[ri4] + beta_val * p_sum[ri4]),
                           m_i[ri4].store(m_new), beta_i[ri4].store(beta_val)).end(ri4)
    acc, l_i, m_i, beta_i = acc.after(correction), l_i.after(correction), m_i.after(correction), beta_i.after(correction)
  if not _early_v:
    # after(qk_done) is per-wave only; lds_reuse_barrier syncs the workgroup so K is not
    # overwritten by V while another wave is still in QK.
    V_store = _v_store_unrolled(V_lds.after(lds_reuse_barrier))
  assert V_store is not None
  pv_barrier = UOp.barrier(UOp.group(P_store, V_store))
  P_lds, V_lds = P_lds.after(pv_barrier), V_lds.after(pv_barrier)
  # Diagnostic: P/V LDS after barrier (pre-WMMA inputs).
  if dumps and "p_lds" in dumps:
    dp = dumps["p_lds"]  # (BH, Nm, Ntiles, WAVES_N, BLOCK_M, BLOCK_N)
    p_lds_stores = []
    for ri in range(TM):
      for rj in range(TN):
        qrow = wave_m * WMMA_M + ri * LANES_PER_WAVE_M + lane_m
        kcol = rj * LANES_PER_WAVE_N + lane_n
        p_lds_stores.append(dp[block_bh, block_m, n_tile, wave_n, qrow, kcol].store(P_lds[wave_n, qrow, kcol].float()))
    P_lds = P_lds.after(UOp.group(*p_lds_stores))
  if dumps and "v_lds" in dumps:
    dv = dumps["v_lds"]  # (BH, Nm, Ntiles, D, BLOCK_N) — unique (drow, ncol) per thread
    v_lds_stores = []
    for ei in range(KV_ELEMS_PER_THREAD):
      # V_store writes V_lds.permute(1,0)==(BLOCK_N,D) as (THREADS, KV_ELEMS) row-major.
      flat = tid * KV_ELEMS_PER_THREAD + ei
      ncol, drow = flat // D, flat % D
      v_lds_stores.append(dv[block_bh, block_m, n_tile, drow, ncol].store(V_lds[drow, ncol].float()))
    V_lds = V_lds.after(UOp.group(*v_lds_stores))
  pv_acc = _reg((TM, TD), 10, 0, n_tile).after(pv_barrier)
  pv_view = pv_acc.reshape(TM // WMMA_ACC, WMMA_ACC, TD).permute(0, 2, 1)
  P_view = P_lds[wave_n].reshape(WAVES_M, TM // WMMA_ACC, WMMA_M, BLOCK_N // WMMA_K, WMMA_K)
  V_view = V_lds.reshape(WAVES_N, TD, WMMA_N, BLOCK_N // WMMA_K, WMMA_K)
  if _acc_small and _k_hip_pv:
    pv_stores = []
    for td_i in range(TD):
      for tm_i in range(TM // WMMA_ACC):
        pv_frag = pv_view[tm_i, td_i]
        wmma_c = pv_frag
        for k_i in range(BLOCK_N // WMMA_K):
          p_frag = P_view[wave_m, tm_i, lane_n, k_i]
          v_frag = V_view[wave_n, td_i, lane_n, k_i]
          wmma_c = UOp.wmma(p_frag, v_frag, wmma_c, *WMMA_ARG)
          if _mid_every and (k_i + 1) % _mid_every == 0 and k_i + 1 < BLOCK_N // WMMA_K:
            wmma_c = pv_frag.after(pv_frag.store(wmma_c))
        pv_stores.append(pv_frag.store(wmma_c))
    pv_done = UOp.group(*pv_stores)
  elif _acc_small and _k_unroll_full:
    pv_stores = []
    for td_i in range(TD):
      for tm_i in range(TM // WMMA_ACC):
        pv_frag = pv_view[tm_i, td_i]
        wmma_c = pv_frag
        for k_i in range(BLOCK_N // WMMA_K):
          p_frag = P_view[wave_m, tm_i, lane_n, k_i]
          v_frag = V_view[wave_n, td_i, lane_n, k_i]
          wmma_c = UOp.wmma(p_frag, v_frag, wmma_c, *WMMA_ARG)
          if _mid_every and (k_i + 1) % _mid_every == 0 and k_i + 1 < BLOCK_N // WMMA_K:
            wmma_c = pv_frag.after(pv_frag.store(wmma_c))
        pv_stores.append(pv_frag.store(wmma_c))
    pv_done = UOp.group(*pv_stores)
  elif _acc_small and _factor_pv and (BLOCK_N // WMMA_K) % _k_unroll_factor == 0:
    k_outer = UOp.range((BLOCK_N // WMMA_K) // _k_unroll_factor, 400, AxisType.LOOP)
    pv_stores = []
    for td_i in range(TD):
      for tm_i in range(TM // WMMA_ACC):
        pv_frag = pv_view[tm_i, td_i]
        wmma_c = pv_frag.after(k_outer)
        for k_off in range(_k_unroll_factor):
          k_i = k_outer * _k_unroll_factor + k_off
          p_frag = P_view[wave_m, tm_i, lane_n, k_i]
          v_frag = V_view[wave_n, td_i, lane_n, k_i]
          wmma_c = UOp.wmma(p_frag, v_frag, wmma_c, *WMMA_ARG)
        pv_stores.append(pv_frag.store(wmma_c))
    pv_done = UOp.group(*pv_stores).end(k_outer)
  elif _acc_small:
    k_pv = UOp.range(BLOCK_N//WMMA_K, 400, AxisType.LOOP)
    pv_stores = []
    for td_i in range(TD):
      for tm_i in range(TM // WMMA_ACC):
        pv_frag = pv_view[tm_i, td_i]
        p_frag = P_view[wave_m, tm_i, lane_n, k_pv]
        v_frag = V_view[wave_n, td_i, lane_n, k_pv]
        pv_stores.append(pv_frag.store(UOp.wmma(p_frag, v_frag, pv_frag.after(k_pv), *WMMA_ARG)))
    pv_done = UOp.group(*pv_stores).end(k_pv)
  else:
    k_pv, tm2, tn2 = UOp.range(BLOCK_N//WMMA_K, 400, AxisType.LOOP), UOp.range(TM//WMMA_ACC, 401), UOp.range(TD, 402)
    pv_frag = pv_view[tm2, tn2]
    p_frag = P_view[wave_m, tm2, lane_n, k_pv]
    v_frag = V_view[wave_n, tn2, lane_n, k_pv]
    pv_done = pv_frag.store(UOp.wmma(p_frag, v_frag, pv_frag.after(k_pv), *WMMA_ARG)).end(tm2, tn2).end(k_pv)
  pv_acc = pv_acc.after(pv_done)
  def _pv_logical_stores(dp: UOp, src: UOp) -> list[UOp]:
    # Each thread owns unique (qrow, dcol); same layout for pre- and post-copy dumps.
    stores = []
    for ri in range(TM):
      for rj in range(TD):
        qrow = wave_m * WMMA_M + ri * LANES_PER_WAVE_M + lane_m
        dcol = wave_n * (TD * LANES_PER_WAVE_N) + rj * LANES_PER_WAVE_N + lane_n
        stores.append(dp[block_bh, block_m, n_tile, qrow, dcol].store(src[ri, rj]))
    return stores
  # Pre-copy dump: slot-10 WMMA C / parked ACC result (before ACC→pv_soft).
  if dumps and "pv_wmma" in dumps:
    pv_acc = pv_acc.after(UOp.group(*_pv_logical_stores(dumps["pv_wmma"], pv_acc)))
  # Fully overwritten before read — skip REG zero-fill (same as S_soft).
  # Keep soft copy even under ACC_SMALL: PV ACC→soft is faster than EXTRACT-at-use
  # (~670 vs ~810µs). AMD_FLASH_PV_ACC_DIRECT=1 skips copy (slower on HW).
  # PV ACC→soft (PV_ACC_DIRECT removed — was slower).
  pv_soft = UOp.placeholder((TM, TD), dtypes.float, slot=17, addrspace=AddrSpace.REG)
  if _acc_small and (TM * TD) % 4 == 0:  # VEC_COPY hardcoded on with ACC_SMALL
    src, dst = pv_acc.reshape(TM * TD), pv_soft.reshape(TM * TD)
    pv_acc = pv_soft.after(UOp.group(*[dst[i:i+4].store(src[i:i+4]) for i in range(0, TM * TD, 4)]))
  else:
    pv_acc = pv_soft.after(UOp.group(*[pv_soft[ri, rj].store(pv_acc[ri, rj]) for ri in range(TM) for rj in range(TD)]))
  # Post-copy dump: slot-17 soft buffer (or still slot-10 if PV_ACC_DIRECT).
  if dumps and "pv" in dumps:
    pv_acc = pv_acc.after(UOp.group(*_pv_logical_stores(dumps["pv"], pv_acc)))
  ri5, rj5 = UOp.range(TM, 410), UOp.range(TD, 411)
  # When ACC_WORK is on, normalize from the last tile's VGPR work copy so the epilogue
  # does not reload slot-2 scratch (writeback still feeds the next n_tile iter).
  acc_work_final: UOp|None = None
  if _fu & 4:  # ACC_UNROLL hardcoded on with soft/corr
    # Const-index acc update. Slot 2 stays unpromoted (REDUCE-carried); soft/stats promote.
    acc_c = acc.after(n_tile)
    if _acc_work:
      assert acc_work is not None
      work = acc_work
      acc_up = UOp.group(*[work[ri, rj].store(work[ri, rj] + beta_i[ri] * pv_acc[ri, rj])
                           for ri in range(TM) for rj in range(TD)])
      work = work.after(acc_up)
      if (TM * TD) % 4 == 0:
        src, dst = work.reshape(TM * TD), acc_c.reshape(TM * TD)
        writeback = UOp.group(*[dst[i:i+4].store(src[i:i+4]) for i in range(0, TM * TD, 4)])
      else:
        writeback = UOp.group(*[acc_c[ri, rj].store(work[ri, rj]) for ri in range(TM) for rj in range(TD)])
      n_tile_end = writeback.barrier().end(n_tile)
      acc_work_final = work.after(n_tile_end)
    else:
      acc_up = UOp.group(*[acc_c[ri, rj].store(acc_c[ri, rj] + beta_i[ri] * pv_acc[ri, rj])
                           for ri in range(TM) for rj in range(TD)])
      n_tile_end = acc_up.barrier().end(n_tile)
  else:
    n_tile_end = acc[ri5, rj5].store(acc[ri5, rj5] + beta_i[ri5] * pv_acc[ri5, rj5]).end(ri5, rj5).barrier().end(n_tile)
  acc, l_i, m_i = acc.after(n_tile_end), l_i.after(n_tile_end), m_i.after(n_tile_end)
  # Diagnostic: pre-normalize accumulator → logical (BLOCK_M, D).
  if dumps and "acc" in dumps:
    da = dumps["acc"]  # (BH, Nm, BLOCK_M, D)
    acc_stores = []
    for ri in range(TM):
      for rj in range(TD):
        qrow = wave_m * WMMA_M + ri * LANES_PER_WAVE_M + lane_m
        dcol = wave_n * (TD * LANES_PER_WAVE_N) + rj * LANES_PER_WAVE_N + lane_n
        acc_stores.append(da[block_bh, block_m, qrow, dcol].store(acc[ri, rj]))
    acc = acc.after(UOp.group(*acc_stores))
  # Fuse normalize into the global store — avoid writing scaled acc back through slot-2 scratch.
  # Prefer ACC_WORK VGPRs when present so we also skip the epilogue scratch reload.
  acc_out = acc_work_final if acc_work_final is not None else acc
  inv_l = (1 / l_i).reshape(TM, 1).expand(TM, TD)
  o = o.reshape(WAVES_M, TM, LANES_PER_WAVE_M, 1, WAVES_N, TD, LANES_PER_WAVE_N, 1) \
    .permute((0, 4, 2, 6, 1, 3, 5, 7)).reshape(THREADS_PER_BLOCK, TM, TD)
  return o[tid].store(acc_out * inv_l).end(wave_m, wave_n, lane).end(block_m, block_bh).sink(arg=KernelInfo(opts_to_apply=()))

def flash_attention(q:Tensor, assigned_kv:Tensor, valid_end:int|UOp) -> Tensor:
  # cached flash attention on the half KV cache (already written through assigned_kv); valid_end stays bound at the graph level
  T_real, q_start = q.shape[2], None
  D, N, group = q.shape[3], assigned_kv.shape[3], q.shape[1] // assigned_kv.shape[2]
  decode = resolve(T_real == 1, False)
  direct_isa = amd_direct_isa(q.device)
  # Shape/LDS gates from upstream master; fall back to SDPA when hand kernels cannot run.
  # The C/LLVM prefill mapping needs one additional float32 BLOCK_M×BLOCK_N LDS tile;
  # without accounting for it D=320 requests 68 KiB and fails HIP codegen on gfx1100.
  lds_extra = 0 if direct_isa else BLOCK_M * BLOCK_N * 4
  supported = D % 32 == 0 and (D & (D-1) == 0 and N % 64 == 0 and group*((D+LDS_PAD)*2+8) <= 65536 if decode else
    D >= 64 and 2*(2*BLOCK_M*(D+LDS_PAD) + D*(BLOCK_N+LDS_PAD)) + lds_extra <= 65536 and N % BLOCK_N == 0 and
    q.max_shape[2] % BLOCK_M == 0)
  # The direct ISA prefill WMMA mapping is numerically unsafe for D=64 on Navi31; use the
  # generic SDPA path for that case rather than exposing uninitialized fragment lanes.
  if not supported or (not decode and D == 64 and amd_direct_isa(q.device)):
    _flash_clear_decode_policy()
    k, v = (assigned_kv[i, :, :, :valid_end].float() for i in range(2))
    mask = None if decode else Tensor.full((T_real, valid_end), -math.inf, dtype=dtypes.float32, device=q.device).triu(valid_end-T_real+1)
    return q.float().scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)
  if decode:
    if not (amd_direct_isa(q.device) or getenv("AMD_FLASH_DIRECT", 0)):
      return amd_flash_attention_decode(q.half(), assigned_kv, valid_end, cast(int, N))
    # ALLOW_UPCAST16=0 for whole decode build+realize (see _flash_direct_compile_env).
    with _flash_direct_compile_env(flash_acc_small=False):
      return amd_flash_attention_decode(q.half(), assigned_kv, valid_end, cast(int, N)).realize()
  # Keep hand WMMA prefill opt-in and restricted to aligned static starts; other cases use SDPA.
  # ACC_SMALL stays scoped to flash because global WMMA accumulator parking changes quantized GEMV behavior.
  q_tokens = _bound_int(T_real)
  aligned_causal_start = isinstance(valid_end, int) and q_tokens is not None and (valid_end-q_tokens) % BLOCK_M == 0
  # The hand PV path is wrong for partial query/KV-tile overlaps. Keep it to aligned,
  # statically known lengths; a bound UOp is still a runtime scalar in the compiled kernel.
  if direct_isa and (not getenv("AMD_FLASH_DIRECT", 0) or not aligned_causal_start):
    _flash_clear_decode_policy()
    k, v = assigned_kv[0, :, :, 0:valid_end, :], assigned_kv[1, :, :, 0:valid_end, :]
    mask = Tensor.full((1, 1, T_real, valid_end), float("-inf"), dtype=dtypes.float32, device=q.device, buffer=False).triu(valid_end-T_real+1)
    return q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)
  # Flash-only ACC residency via python-unrolled WMMA. Renderer parks ≤64 only while
  # AMD_FLASH_ACC_SMALL is set for this realize (not via FLASH_DIRECT process-wide —
  # that parked every matmul and broke eye/GEMM). ACC_SMALL defaults on for DIRECT;
  # AMD_FLASH_ACC_SMALL=0 disables; AMD_WMMA_ACC_SMALL=1 forces (unsafe for quant).
  # Keep the partial K unroll on direct ISA; HIP retains its ranged QK mapping.
  use_acc_small = bool(getenv("AMD_WMMA_ACC_SMALL", 0) or getenv("AMD_FLASH_ACC_SMALL", 1))
  use_k_unroll = 2 if direct_isa else 0
  use_k_hip_scope = "all"
  # Mid-store belongs in the cache key (not a bare getenv inside the cached builder).
  _mid_default = 4 if (use_k_unroll == 1 or use_k_unroll >= 8) else 0
  use_k_midstore = _mid_default if use_acc_small else 0  # K_MIDSTORE follows factor default
  # Build+realize under ACC_SMALL remat + ALLOW_UPCAST16=0 (must wrap custom_kernel).
  # Read SKIP_SLOTS only after compile_env re-applies prefill policy (decode leaves sticky "").
  with _flash_direct_compile_env(flash_acc_small=use_acc_small):
    # Ship prefill: ACC_VGPR + HOIST_Q; SKIP clears slot 2 under ACC_VGPR.
    use_acc_vgpr = bool(use_acc_small)  # ACC_VGPR hardcoded on with ACC_SMALL
    # REG_PROMOTE_SKIP folded into ACC_SMALL/VOPD_FMAC in renderer (promote slot 2 under flash)
    _skip_reg: set[int] = set() if use_acc_vgpr else {2}
    use_acc_work = bool(use_acc_small and not use_acc_vgpr and 2 in _skip_reg)
    use_hoist_q = bool(use_acc_small)  # HOIST_Q hardcoded on with ACC_SMALL
    if isinstance(T_real, UOp):
      # symbolic chunk: pad the queries to the static tile; garbage rows are sliced off
      T_pad = q.max_shape[2]
      assert T_pad % BLOCK_M == 0, "chunk_size must be a multiple of 32"
      q, q_start = q.pad_to((*q.shape[:2], T_pad, q.shape[3])), valid_end - T_real
    B, H, T, D = q.shape
    out = Tensor.empty(B*H, T, D, dtype="float32", device=q.device)
    fxn = functools.partial(_amd_flash_attention, valid_kv_len=valid_end, q_start=q_start,
                            acc_small=use_acc_small, k_unroll=use_k_unroll, use_acc_work=use_acc_work,
                            k_hip_scope=use_k_hip_scope, k_midstore=use_k_midstore,
                            acc_vgpr=use_acc_vgpr, hoist_q=use_hoist_q)
    q_input = q.half().reshape(B*H, T, D)
    if isinstance(valid_end, UOp) or isinstance(q_start, UOp):
      # Keep bound dimensions as free variables in the kernel body. The outer LINEAR carries their values,
      # while JIT replay supplies updated bindings through var_vals.
      scalars = tuple(sorted({v for root in (valid_end, q_start) if isinstance(root, UOp)
                              for v in root.backward_slice_with_self if v.is_bound_var}, key=lambda v: v.expr))
      args = (out.uop, q_input.uop, assigned_kv.uop)
      scalar_params = {v: v.unbound() for v in scalars}
      body = fxn(*(UOp.placeholder_like(x, slot=i) for i,x in enumerate(args)),
                 valid_kv_len=valid_end.substitute(scalar_params, walk=True) if isinstance(valid_end, UOp) else valid_end,
                 q_start=q_start.substitute(scalar_params, walk=True) if isinstance(q_start, UOp) else q_start)
      call = body.call(*args)
      out = Tensor(out.uop.after(call).after(*scalars))
    else: out = Tensor.custom_kernel(out, q_input, assigned_kv, fxn=fxn)[0]
    out = out.reshape(B, H, T, D)
    if q_start is not None: out = out[:, :, :T_real]
    if not use_acc_small or not ALLOW_DEVICE_USAGE: return out
    return out.realize()

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
  token = UOp.range(tokens, 2, AxisType.LOOP)
  keys = tuple(valid.where(k[bh, token, col].load(), UOp.const(0, dtypes.float32)) for col,valid in zip(cols, col_valids))
  queries = tuple(valid.where(q[bh, token, col].load(), UOp.const(0, dtypes.float32)) for col,valid in zip(cols, col_valids))
  updates, stores = [], []
  for row_idx,row in enumerate(rows):
    previous = tuple(current.after(token)[row_idx*key_chunks+i].load() for i in range(key_chunks))
    bv = beta[bh, token].load()
    if alpha_dim == key_dim:
      av = tuple(valid.where(alpha[bh, token, col].load(), UOp.const(0, dtypes.float32)) for col,valid in zip(cols, col_valids))
      state_k = warp_reduce(sum((x*a*y for x,a,y in zip(previous, av, keys)), UOp.const(0, dtypes.float32)), full_wave=True)
      state_q = warp_reduce(sum((x*a*y for x,a,y in zip(previous, av, queries)), UOp.const(0, dtypes.float32)), full_wave=True)
      delta = (v[bh, token, row].load() - state_k) * bv
      updates += [x*a + delta*y for x,a,y in zip(previous, av, keys)]
      stores.append(core[bh, token, row.valid(lane.eq(0))].store(state_q + delta*kq[bh, token]))
    else:
      av = alpha[bh, token, row if alpha_dim > 1 else 0].load()
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
  assert alpha.shape[:3] == (batch, heads, tokens) and (len(alpha.shape) == 3 or alpha.shape[-1] in (1, value_dim, key_dim))
  assert (key_dim <= 32 or key_dim % 32 == 0) and value_dim % 4 == 0
  assert q.dtype == k.dtype == dtypes.float32, "recurrent Q/K must be float32"
  assert state.uop.contiguous_view_offset() is not None, "recurrent state must be contiguous"
  core, kq = Tensor.empty_like(v), (q*k).sum(-1).contiguous()
  srcs = (core, q.contiguous(), k.contiguous(), v.contiguous(), beta.contiguous(), alpha.contiguous(), state, kq)
  if start_pos is None: return Tensor.custom_kernel(*srcs, fxn=_gated_delta_prefill_kernel)[0]
  contig = tuple(x.uop if x.uop.op is Ops.AFTER else x.uop.contiguous() for x in srcs)
  params = tuple(UOp.placeholder_like(x, slot=i) for i,x in enumerate(contig))
  assert start_pos.uop.is_bound_var
  # Bound variables carry their value on ParamArg in current UOps, rather than as a source node.
  _, start_pos_val = start_pos.uop.unbind()
  call = _gated_delta_prefill_kernel(*params, UOp.const(start_pos_val, start_pos.uop.dtype)).call(*contig)
  return Tensor(contig[0].after(call))

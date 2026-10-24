import importlib.util, os, unittest
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, UOp, dtypes, nn, function, Device, TinyJit
from tinygrad.llm.kernels.amd import (Linear, _amd_flash_attention, amd_custom_kernels_supported, amd_direct_isa,
                                      q8_quantize, flash_attention, gated_delta_prefill, QUANT_SIZES, HALFWORD_QUANTS,
                                      SMALL_BLOCK_QUANTS, _wmma_rdna4)
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.helpers import DEV, OSX
from test.runtime.test_llm_quantized import QuantLinearMixin

class TestQ8Quantize(QuantLinearMixin, unittest.TestCase):
  def test_wmma_architecture_selection(self):
    if not amd_custom_kernels_supported(device:=Tensor.empty(1).device): self.skipTest("RDNA3/4 required")
    self.assertEqual(_wmma_rdna4(device), Device[device].target[0] == 12)

  def test_values_and_scales(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    x = np.linspace(-3.1, 2.7, 64, dtype=np.float32).reshape(2, 32)
    quant, scale, gsum = q8_quantize(Tensor(x), 2, 32)
    scale_np = np.maximum(np.max(np.abs(x), axis=-1, keepdims=True) / 127, 1e-8)
    expected = np.clip(np.rint(x / scale_np), -127, 127).astype(np.int8)
    np.testing.assert_array_equal(quant.bitcast(dtypes.int8).reshape(2, 32).numpy(), expected)
    np.testing.assert_allclose(scale.numpy(), scale_np, rtol=1e-6)
    # xsum holds the two per-16 sums per 32-wide group
    np.testing.assert_array_equal(gsum.numpy().reshape(2, 2), expected.reshape(2, 2, 16).sum(-1).astype(np.float32))

  def test_quantize_rounding_ties(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    values = np.array([-127,127]+[i+0.5 for i in range(-15,15)],dtype=np.float32)
    quant,_,_ = q8_quantize(Tensor(values),1,32)
    np.testing.assert_array_equal(quant.bitcast(dtypes.int8).reshape(32).numpy(),np.rint(values).astype(np.int8))

  def test_quantize_nonunit_rounding_ties(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3/4 required")
    values = np.zeros((4, 32), dtype=np.float16)
    values[:, 0] = [1.8837890625, 2.525390625, 2.23828125, 1.736328125]
    values[:, 1], values[:, 2] = values[:, 0]/2, -values[:, 0]/2
    values[3, 3:5] = [-0.3076171875, -0.6904296875]
    expected_scale = np.abs(values.astype(np.float32)).max(-1, keepdims=True)/127
    quant, scale, _ = q8_quantize(Tensor(values), 4, 32)
    actual_scale = scale.numpy()
    np.testing.assert_allclose(actual_scale, expected_scale, rtol=1e-6)
    fp64_values = values.astype(np.float64)
    ratio = fp64_values*127/np.abs(fp64_values).max(-1, keepdims=True)
    expected_quant = np.clip(np.rint(ratio), -127, 127).astype(np.int8)
    actual_quant = quant.bitcast(dtypes.int8).reshape(4, 32).numpy()
    ties = np.isclose(np.abs(ratio-np.floor(ratio)), 0.5, atol=1e-12)
    np.testing.assert_array_equal(actual_quant[~ties], expected_quant[~ties])
    self.assertTrue(ties.any())
    np.testing.assert_array_less(np.abs(actual_quant[ties].astype(np.int16)-expected_quant[ties]), 2)

  def test_q6_linear_compiles_in_function(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    packed = rng.integers(0, 256, 210, dtype=np.uint8)
    packed[-2:] = np.array([0.01], dtype=np.float16).view(np.uint8)
    raw = Tensor(np.pad(packed, (2, 0))).contiguous().realize()[2:]
    decoded = ggml_data_to_tensor(raw, 256, 14).reshape(1, 256)
    linear = Linear(256, 1, bias=False)
    nn.state.load_state_dict(linear, {"weight":decoded}, verbose=False, realize=False)
    @function(allow_implicit=True)
    def run(x:Tensor): return linear(x)
    self.assertTrue(np.isfinite(run(Tensor.randn(1, 256)).realize().item()))
    self.assertEqual(linear.weight.nbytes(), 210)
    self.assertEqual(linear.weight.dtype, dtypes.uint16)

  def test_q4_k_linear(self): self._test_quant_linear(12, 144)
  def test_iq4_linear(self): self._test_quant_linear(23, 136)
  def test_q5_linear(self): self._test_quant_linear(13, 176)
  def test_q6_linear_wmma(self): self._test_quant_linear(14, 210, token_counts=(16, 32, 64, 128, 256))
  def test_q8_0_linear_wmma(self): self._test_quant_linear(8, 272, token_counts=(32,))
  def test_q8_0_linear_small_blocks(self): self._test_quant_linear(8, 272, in_features=288, out_features=16, token_counts=(1, 16))
  def test_q6_linear_odd_blocks(self): self._test_quant_linear(14, 210, in_features=768, out_features=3, token_counts=(1, 3, 16))
  def test_q2_k_linear(self): self._test_quant_linear(10, 84, in_features=768, token_counts=(1, 3, 16))
  def test_q3_k_linear(self): self._test_quant_linear(11, 110, in_features=768, token_counts=(1, 3, 16))
  def test_iq2_xs_linear(self): self._test_quant_linear(17, 74, in_features=768, token_counts=(1, 3, 16))
  def test_iq3_xxs_linear(self): self._test_quant_linear(18, 98, in_features=768, token_counts=(1, 3, 16))
  def test_iq4_nl_linear(self): self._test_quant_linear(20, 144, in_features=768, token_counts=(1, 3, 16))
  def test_iq3_s_linear(self): self._test_quant_linear(21, 110, in_features=768, token_counts=(1, 3, 16))
  def test_iq2_s_linear(self): self._test_quant_linear(22, 82, in_features=768, token_counts=(1, 3, 16))

  def test_wmma_linear_symbolic(self):
    with patch("tinygrad.llm.kernels.amd.q8_quantize", side_effect=AssertionError("expected WMMA")):
      for typ in QUANT_SIZES:
        self._test_quant_linear(typ, QUANT_SIZES[typ], in_features=256, out_features=48, token_counts=(3, 17), bias=True, symbolic=True)

  def test_mixed_quant_linear_wmma_tiles(self):
    with patch("tinygrad.llm.kernels.amd.q8_quantize", side_effect=AssertionError("expected WMMA")):
      for typ in HALFWORD_QUANTS:
        self._test_quant_linear(typ, QUANT_SIZES[typ], in_features=768, out_features=64, token_counts=(16, 32, 64), bias=True)

  def test_wmma_dequantization_basis(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3/4 required")
    if importlib.util.find_spec("gguf") is None: self.skipTest("optional gguf package is not installed")
    from gguf import dequantize, GGMLQuantizationType
    rng = np.random.default_rng(1907)
    scales = np.array([0, 2**-24, -2**-24, 2**-14, -0.00035, .001, .0037, -.125, 1, -4, 8], np.float16)
    identity = Tensor(np.eye(768, dtype=np.float16)).realize()
    for typ, size in QUANT_SIZES.items():
      with self.subTest(ggml_type=typ):
        packed = rng.integers(0, 256, (48*3, size), dtype=np.uint8)
        blocks = packed.reshape(-1, SMALL_BLOCK_QUANTS[typ]*2) if typ in SMALL_BLOCK_QUANTS else packed
        offset = size-2 if typ in (11, 14) else 80 if typ == 10 else 0
        blocks[:, offset:offset+2] = np.resize(scales, len(blocks)).view(np.uint8).reshape(-1, 2)
        if typ in (10, 12, 13):
          offset = 82 if typ == 10 else 2
          blocks[:, offset:offset+2] = np.resize(scales[::-1], len(blocks)).view(np.uint8).reshape(-1, 2)
        expected = dequantize(packed.flatten(), GGMLQuantizationType(typ)).reshape(48, 768).astype(np.float16).astype(np.float32).T
        # Odd super-block counts and halfword-only alignment exercise both row and block offsets.
        pad = 2 if typ in HALFWORD_QUANTS else 4
        raw = Tensor(np.pad(packed.flatten(), (pad, 0))).realize()[pad:]
        linear = Linear(768, 48, bias=False)
        linear.weight = ggml_data_to_tensor(raw, 48*768, typ).reshape(48, 768)
        actual = linear(identity).numpy()
        self.assertEqual(linear.ggml_type, typ)
        # Bare RDNA3 WMMA has a small FP32 accumulation bias even for identity products.
        np.testing.assert_array_equal(actual.astype(np.float16), expected.astype(np.float16))
        np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2**-36)

  def test_quant_linear_unaligned_storage_fallback(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3/4 required")
    if importlib.util.find_spec("gguf") is None: self.skipTest("optional gguf package is not installed")
    from gguf import dequantize, GGMLQuantizationType
    rng = np.random.default_rng(73)
    x = rng.normal(size=(16, 256)).astype(np.float32)
    for typ, offset in ((12, 1), (12, 2), (12, 3), (21, 1), (21, 3)):
      with self.subTest(ggml_type=typ, offset=offset):
        packed = rng.integers(0, 256, (16, QUANT_SIZES[typ]), dtype=np.uint8)
        packed[:, :2] = np.array([.001], np.float16).view(np.uint8)
        if typ == 12: packed[:, 2:4] = np.array([.0002], np.float16).view(np.uint8)
        weight = dequantize(packed.flatten(), GGMLQuantizationType(typ)).reshape(16, 256)
        raw = Tensor(np.pad(packed.flatten(), (offset, 0))).realize()[offset:]
        linear = Linear(256, 16, bias=False)
        linear.weight = ggml_data_to_tensor(raw, 16*256, typ).reshape(16, 256)
        np.testing.assert_allclose(linear(Tensor(x)).numpy(), x @ weight.T, rtol=2e-5, atol=1e-5)
        self.assertIsNone(linear.ggml_type)

  def test_q6_short_wmma_dequantization_basis(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3/4 required")
    if importlib.util.find_spec("gguf") is None: self.skipTest("optional gguf package is not installed")
    from gguf import dequantize, GGMLQuantizationType
    rng = np.random.default_rng(1907)
    scales = np.array([0, 2**-24, -2**-24, 2**-14, -0.00035, .001, .0037, -.125, 1, -4, 8], np.float16)
    packed = rng.integers(0, 256, (48*3, 210), dtype=np.uint8)
    packed[:, 208:210] = np.resize(scales, len(packed)).view(np.uint8).reshape(-1, 2)
    expected = dequantize(packed.flatten(), GGMLQuantizationType(14)).reshape(48, 768).T.astype(np.float16).astype(np.float32)
    raw = Tensor(np.pad(packed.flatten(), (2, 0))).realize()[2:]
    linear = Linear(768, 48, bias=False)
    linear.weight = ggml_data_to_tensor(raw, 48*768, 14).reshape(48, 768)
    # Check every K position while retaining the small token counts of the specialized path.
    basis = np.eye(768, dtype=np.float16)
    for tokens in (16, 32, 64):
      for first in range(0, 768, tokens):
        with self.subTest(tokens=tokens, first=first):
          actual = linear(Tensor(basis[first:first+tokens])).numpy()
          ref = expected[first:first+tokens]
          np.testing.assert_array_equal(actual.astype(np.float16), ref.astype(np.float16))
          np.testing.assert_allclose(actual, ref, rtol=2e-7, atol=2**-36)

  def test_wmma_preserves_nonzero_initial_accumulator(self):
    from tinygrad.dtype import AddrSpace
    from tinygrad.llm.kernels.amd import _quant_linear_f16_wmma_kernel
    from tinygrad.uop import Ops
    if not amd_custom_kernels_supported(device:=Tensor.empty(1).device) or _wmma_rdna4(device): self.skipTest("RDNA3 required")
    # The resident-ACC optimization may replace only proven zero initializers.
    def kernel(o, w, x):
      graph = _quant_linear_f16_wmma_kernel(o, w, x, out_features=32, in_features=256, ggml_type=14, direct_isa=amd_direct_isa(device))
      initial = [u for u in graph.toposort() if u.op is Ops.STORE and u.src[0].op is Ops.ALLOC and
                 u.src[0].addrspace is AddrSpace.REG and u.src[1].vmin == u.src[1].vmax == 0]
      self.assertTrue(initial)
      return graph.substitute({u:u.replace(src=(u.src[0], u.src[1].const_like(1.0))) for u in initial})
    out = Tensor.empty(128, 32, dtype=dtypes.float32)
    raw, x = Tensor.zeros(32*105, dtype=dtypes.uint16).realize(), Tensor.zeros(128, 256, dtype=dtypes.half).realize()
    result = Tensor.custom_kernel(out, raw, x, fxn=kernel)[0].numpy()
    np.testing.assert_array_equal(result, np.ones((128, 32), dtype=np.float32))

  def test_mixed_quant_linear_split_k(self):
    for typ in (10, 11, 17, 18, 20, 21, 22):
      with self.subTest(ggml_type=typ):
        self._test_quant_linear(typ, QUANT_SIZES[typ], in_features=1280, out_features=3, token_counts=(1,))

  def test_quant_linear_decode_row_groups(self):
    # One, two, and four waves per workgroup; ragged split-K and multiple tokens must stay independent.
    for typ in (12, 13, 14, 23):
      for outputs in (3, 6, 12):
        with self.subTest(ggml_type=typ, out_features=outputs):
          self._test_quant_linear(typ, QUANT_SIZES[typ], in_features=1280, out_features=outputs, token_counts=(1, 3))

  def test_quant_linear_bias(self):
    for typ in (12, 21, 23):
      with self.subTest(ggml_type=typ):
        self._test_quant_linear(typ, QUANT_SIZES[typ], in_features=256, out_features=16, token_counts=(1, 3, 16), bias=True)

  def test_quant_linear_partial_output_tile(self):
    # Cover a sub-tile output, a trailing tile, and IQ4's larger-output tile selection.
    for typ, size, outputs, tokens in ((12, 144, 16, 16), (12, 144, 48, 32), (13, 176, 48, 16), (23, 136, 4112, 32)):
      with self.subTest(ggml_type=typ, out_features=outputs):
        self._test_quant_linear(typ, size, in_features=256, out_features=outputs, token_counts=(tokens,))

  def test_quant_linear_preserves_rope_permutation(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    for typ, size in ((12, 144), (13, 176), (14, 210), (23, 136)):
      with self.subTest(ggml_type=typ):
        packed = rng.integers(0, 256, (16, size), dtype=np.uint8)
        packed[:, -2:] = np.array([0.001], dtype=np.float16).view(np.uint8)
        if typ != 14: packed[:, :2] = np.array([0.001], dtype=np.float16).view(np.uint8)
        if typ in (12, 13): packed[:, 2:4] = np.array([0.0002], dtype=np.float16).view(np.uint8)
        raw = Tensor(np.pad(packed.flatten(), (4, 0))).contiguous().realize()[4:]
        decoded = ggml_data_to_tensor(raw, 16*256, typ).reshape(16, 256).half()
        original = decoded.numpy()
        x = rng.normal(size=(3, 256)).astype(np.float16)
        for prefix in (None, 0, 4):
          with self.subTest(prefix=prefix):
            w = decoded.reshape(2, 8, 256)
            if prefix is None:
              weight = w.rearrange("n (h two) d -> n (two h) d", two=2)
            else:
              weight = w[:, :prefix].cat(w[:, prefix:].rearrange("n (h two) d -> n (two h) d", two=2), dim=1)
            start = prefix or 0
            rows = np.arange(16).reshape(2, 8)
            order = np.concatenate((rows[:, :start], rows[:, start:].reshape(2, -1, 2).transpose(0, 2, 1).reshape(2, -1)), axis=1)
            linear = Linear(256, 16, bias=False)
            linear.weight = weight.reshape(16, 256)
            np.testing.assert_allclose(linear(Tensor(x)).numpy(), x.astype(np.float32) @ original[order.flatten()].astype(np.float32).T,
                                       rtol=3e-3, atol=2e-2)
            self.assertIsNone(linear.ggml_type)

  def test_quant_linear_rejects_unaligned_rows_and_integer_casts(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    for width in (128, 256):
      with self.subTest(width=width):
        packed = np.zeros((2*width//256, 136), dtype=np.uint8)
        packed[:, :2] = np.array([0.001], dtype=np.float16).view(np.uint8)
        packed[:, 8:] = np.arange(128, dtype=np.uint8)
        raw = Tensor(np.pad(packed.flatten(), (4, 0))).realize()[4:]
        weight = ggml_data_to_tensor(raw, 2*width, 23).reshape(2, width)
        if width == 256: weight = weight.int().float()
        expected = weight.numpy().sum(-1)[None]
        linear = Linear(width, 2, bias=False)
        linear.weight = weight
        np.testing.assert_allclose(linear(Tensor.ones(1, width)).numpy(), expected, rtol=1e-3, atol=1e-3)
        self.assertIsNone(linear.ggml_type)

  def test_dense_gemv_preserves_integer_casts(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    linear = Linear(128, 1)
    linear.weight = Tensor.full((1, 128), 0.75).contiguous().realize().int().float()
    linear.bias = Tensor.full((1,), 0.75).contiguous().realize().int().float()
    np.testing.assert_array_equal(linear(Tensor.ones(1, 128)).numpy(), 0)

  def test_dense_gemv_preserves_float_rounding(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    # 1+2^-12 is not representable in fp16: the declared cast must round it to 1.
    weight = Tensor.full((1, 128), 1+2**-12, dtype=dtypes.float32).contiguous().realize()
    bias = Tensor.full((1,), 1+2**-12, dtype=dtypes.float32).contiguous().realize()
    for widen in (False, True):
      with self.subTest(widen=widen):
        linear = Linear(128, 1)
        linear.weight, linear.bias = weight.half(), bias.half()
        if widen: linear.weight, linear.bias = linear.weight.float(), linear.bias.float()
        np.testing.assert_array_equal(linear(Tensor.ones(1, 128)).numpy(), 129)

  def test_dense_gemv_float32_range(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    linear = Linear(128, 1, bias=False)
    linear.weight = Tensor.full((1, 128), 1/128, dtype=dtypes.float32).realize()
    np.testing.assert_array_equal(linear(Tensor.full((1, 128), 65536, dtype=dtypes.float32)).numpy(), 65536)

  def test_dense_gemv_high_half_register_pressure(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3/4 required")
    rng = np.random.default_rng(72)
    x = rng.normal(0, 0.1, (4, 2048)).astype(np.float32)
    weight = rng.normal(0, 0.01, (64, 2048)).astype(np.float16)
    linear = Linear(2048, 64, bias=False)
    linear.weight = Tensor(weight).realize()
    actual = linear(Tensor(x)).numpy()
    self.assertTrue(np.isfinite(actual).all())
    np.testing.assert_allclose(actual, x @ weight.astype(np.float32).T, rtol=2e-5, atol=1e-6)

  def test_dense_gemv_long_k(self):
    device = Tensor.empty(1).device
    if not amd_custom_kernels_supported(device): self.skipTest("RDNA3/4 required")
    if not amd_direct_isa(device): self.skipTest("Direct ISA required")
    rng = np.random.default_rng(82)
    weight = rng.normal(0, 0.01, (16, 8192)).astype(np.float16)
    linear = Linear(8192, 16, bias=False)
    linear.weight = Tensor(weight).realize()
    for tokens in (1, 4):
      with self.subTest(tokens=tokens):
        x = rng.normal(0, 0.1, (tokens, 8192)).astype(np.float32)
        actual = linear(Tensor(x)).numpy()
        np.testing.assert_allclose(actual, x @ weight.astype(np.float32).T, rtol=2e-5, atol=1e-5)

  def test_dense_gemm_wmma(self):
    device = Tensor.empty(1).device
    if not amd_custom_kernels_supported(device): self.skipTest("RDNA3/4 required")
    if not amd_direct_isa(device): self.skipTest("Direct ISA required")
    if _wmma_rdna4(device): self.skipTest("RDNA3 only")
    rng = np.random.default_rng(83)
    weight = rng.normal(0, 0.01, (4096, 2048)).astype(np.float16)
    bias = rng.normal(0, 0.01, (4096,)).astype(np.float32)
    for tokens, use_bias in ((16, False), (32, True)):
      with self.subTest(tokens=tokens, use_bias=use_bias):
        x = rng.normal(0, 0.1, (tokens, 2048)).astype(np.float32)
        linear = Linear(2048, 4096, bias=use_bias)
        linear.weight = Tensor(weight).realize()
        if use_bias: linear.bias = Tensor(bias).realize()
        expected = x @ weight.astype(np.float32).T + (bias if use_bias else 0)
        np.testing.assert_allclose(linear(Tensor(x)).numpy(), expected, rtol=2e-3, atol=2e-4)

  def test_gated_delta_state_and_precision(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    for case in ("view", "reset", "half"):
      with self.subTest(case=case):
        q = Tensor.full((1, 1, 1, 32), 256 if case == "half" else 1, dtype=dtypes.half if case == "half" else dtypes.float32)
        state = Tensor.full((1, 1, 32, 4), int(case == "reset"), dtype=dtypes.float32).contiguous().realize().transpose(-1, -2)
        if case != "view": state = state.contiguous().realize()
        start = Tensor(UOp.variable("start_pos", 0, 10).bind(0)) if case == "reset" else None
        beta = Tensor.full((1, 1, 1), 1/2097152 if case == "half" else 1, dtype=dtypes.float32)
        if case != "reset":
          message = "recurrent state must be contiguous" if case == "view" else "recurrent Q/K must be float32"
          with self.assertRaisesRegex(AssertionError, message):
            gated_delta_prefill(q, q, Tensor.ones(1, 1, 1, 4), beta, Tensor.ones(1, 1, 1), state, start)
          continue
        out = gated_delta_prefill(q, q, Tensor.ones(1, 1, 1, 4), beta, Tensor.ones(1, 1, 1), state, start)
        np.testing.assert_array_equal(out.numpy(), 32)
        np.testing.assert_array_equal(state.numpy(), 1)

  def test_gated_delta_random_scan(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3/4 required")
    rng = np.random.default_rng(42)
    q, k = (rng.normal(0, 0.1, (1, 2, 4, 32)).astype(np.float32) for _ in range(2))
    v = rng.normal(size=(1, 2, 4, 8)).astype(np.float32)
    beta = rng.uniform(0, 1, (1, 2, 4)).astype(np.float32)
    beta[:, :, -1] = 0  # padded step must leave the recurrent state unchanged
    initial = rng.normal(0, 0.1, (1, 2, 8, 32)).astype(np.float32)
    for per_channel in (False, True):
      alpha = rng.uniform(0.8, 1, (1, 2, 4, v.shape[-1]) if per_channel else (1, 2, 4)).astype(np.float32)
      alpha[:, :, -1] = 1
      for start in (0, 3):
        with self.subTest(per_channel=per_channel, start=start):
          expected_state = np.zeros_like(initial) if start == 0 else initial.copy()
          outputs = []
          for t in range(4):
            decay = alpha[:, :, t, :, None] if per_channel else alpha[:, :, t, None, None]
            expected_state *= decay
            delta = (v[:, :, t] - (expected_state*k[:, :, t, None]).sum(-1)) * beta[:, :, t, None]
            expected_state += delta[..., None]*k[:, :, t, None]
            outputs.append((expected_state*q[:, :, t, None]).sum(-1))
          state = Tensor(initial).contiguous().realize()
          out = gated_delta_prefill(*map(Tensor, (q, k, v, beta, alpha)), state, Tensor(UOp.variable("start", 0, 3).bind(start)))
          np.testing.assert_allclose(out.numpy(), np.stack(outputs, axis=2), rtol=2e-4, atol=2e-6)
          np.testing.assert_allclose(state.numpy(), expected_state, rtol=2e-4, atol=2e-6)

  def test_dense_gemv_bias(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    w, bias = rng.normal(size=(32, 128)).astype(np.float16), rng.normal(size=32).astype(np.float16)
    linear = Linear(128, 32)
    linear.weight, linear.bias = Tensor(w), Tensor(bias)
    for tokens in (1, 3):
      with self.subTest(tokens=tokens):
        x = rng.normal(size=(tokens, 128)).astype(np.float16)
        np.testing.assert_allclose(linear(Tensor(x)).numpy(), x.astype(np.float32) @ w.astype(np.float32).T + bias, rtol=2e-3, atol=2e-3)

  def test_q6_linear_multiple_tokens(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    in_features, blocks = 2048, 16*2048//256
    packed = rng.integers(0, 256, blocks*210, dtype=np.uint8)
    for i in range(blocks): packed[i*210+208:i*210+210] = np.array([0.01], dtype=np.float16).view(np.uint8)
    raw = Tensor(np.pad(packed, (4, 0))).contiguous().realize()[4:]
    decoded = ggml_data_to_tensor(raw, 16*in_features, 14).reshape(16, in_features)
    raw_ref = Tensor(np.pad(packed, (4, 0)), device="CPU").contiguous().realize()[4:]
    weight = ggml_data_to_tensor(raw_ref, 16*in_features, 14).reshape(16, in_features).numpy()
    linear = Linear(in_features, 16, bias=False)
    nn.state.load_state_dict(linear, {"weight":decoded}, verbose=False, realize=False)
    x = rng.normal(size=(3, in_features)).astype(np.float32)
    scale = np.maximum(np.abs(x).reshape(3, in_features//32, 32).max(-1, keepdims=True) / 127, 1e-8)
    xq = np.clip(np.rint(x.reshape(3, in_features//32, 32) / scale), -127, 127) * scale
    np.testing.assert_allclose(linear(Tensor(x)).numpy(), xq.reshape(3, in_features) @ weight.T, rtol=2e-3, atol=2e-2)
    self.assertEqual(linear.ggml_type, 14)

    # symbolic token counts take the padded kernel path and give the same results
    generic = Linear(in_features, 16, bias=False)
    nn.state.load_state_dict(generic, {"weight":decoded}, verbose=False, realize=False)
    sym = Tensor(np.concatenate([x, np.zeros((1, in_features), np.float32)])).contiguous()[:UOp.variable("tokens", 1, 4).bind(3)]
    np.testing.assert_allclose(generic(sym)[:3].numpy(), xq.reshape(3, in_features) @ weight.T, rtol=2e-3, atol=2e-2)
    self.assertTrue(generic.use_custom_quant)
    self.assertEqual(generic.ggml_type, 14)

  def test_attention_fallback_shapes(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    for tokens, capacity, dim in ((1, 65, 64), (32, 64, 32), (32, 64, 384), (32, 64, 512)):
      with self.subTest(tokens=tokens, capacity=capacity, dim=dim):
        valid = 33
        cache = np.full((2, 1, 1, capacity, dim), np.nan, dtype=np.float16)
        cache[0, :, :, :valid] = 0
        cache[1, :, :, :valid] = np.arange(valid)[:, None]
        q = Tensor.zeros(1, 2, tokens, dim, dtype=dtypes.half)
        expected = np.broadcast_to(np.arange(valid-tokens, valid)[None, None, :, None]/2, q.shape)
        np.testing.assert_allclose(flash_attention(q, Tensor(cache), valid).numpy(), expected, rtol=1e-3, atol=1e-3)

  def test_prefill_attention_fallback_random_gqa(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    batch, heads, kv_heads, tokens, dim, valid, capacity = 2, 4, 2, 32, 64, 45, 64
    q = rng.normal(size=(batch, heads, tokens, dim)).astype(np.float16)
    cache = rng.normal(size=(2, batch, kv_heads, capacity, dim)).astype(np.float16)
    cache[:, :, :, valid:] = np.nan
    k, v = (np.repeat(c[:, :, :valid].astype(np.float32), heads//kv_heads, axis=1) for c in cache)
    scores = q.astype(np.float32) @ k.swapaxes(-1, -2) / np.sqrt(dim)
    scores = np.where(np.arange(valid)[None, :] <= np.arange(valid-tokens, valid)[:, None], scores, -np.inf)
    probs = np.exp(scores - scores.max(-1, keepdims=True))
    expected = (probs / probs.sum(-1, keepdims=True)) @ v
    from tinygrad.helpers import getenv
    with patch.dict(os.environ, {"AMD_FLASH_DIRECT": "1"}):
      getenv.cache_clear()
      try: actual = flash_attention(Tensor(q), Tensor(cache).realize(), valid).numpy()
      finally: getenv.cache_clear()
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)

  def test_attention_uses_physical_cache_length(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    q, k, v = Tensor.zeros(1, 2, 1, 32), Tensor.randn(1, 1, 1, 32), Tensor.randn(1, 1, 1, 32)
    cache = Tensor.empty(2, 1, 1, 256, 32, dtype=dtypes.half).contiguous()
    assigned = Tensor(cache.uop.after(cache[:, :, :, 0:1, :].uop.store(Tensor.stack(k, v).cast(dtypes.half).uop)))
    out = flash_attention(q, assigned, 1).realize()
    np.testing.assert_allclose(out.numpy(), v.expand(1, 2, 1, 32).numpy(), rtol=2e-2, atol=2e-2)

  def test_flash_decode_nonfinite_cache_tail(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("AMD hand attention required")
    rng = np.random.default_rng(10518)
    for dim in (32, 64, 128):
      for valid in (1, 5, 33):
        q = rng.normal(size=(1, 2, 1, dim)).astype(np.float16)
        live = rng.normal(size=(2, 1, 1, valid, dim)).astype(np.float16)
        k, v = (np.repeat(x.astype(np.float64), 2, axis=1) for x in live)
        scores = q.astype(np.float64) @ k.swapaxes(-1, -2) / np.sqrt(dim)
        probs = np.exp(scores - scores.max(-1, keepdims=True))
        expected = (probs / probs.sum(-1, keepdims=True)) @ v
        for tail in (np.nan, np.inf, -np.inf):
          with self.subTest(dim=dim, valid=valid, tail=tail):
            cache = np.full((2, 1, 1, 256, dim), tail, dtype=np.float16)
            cache[:, :, :, :valid] = live
            out = flash_attention(Tensor(q), Tensor(cache).realize(), valid).numpy()
            self.assertTrue(np.isfinite(out).all())
            np.testing.assert_allclose(out, expected, rtol=2e-3, atol=2e-3)

  def test_prefill_attention_multiple_tiles(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    q = Tensor.zeros(1, 4, 32, 128)
    k = Tensor.zeros(1, 1, 64, 128, dtype=dtypes.half)
    v = Tensor.cat(Tensor.ones(1, 1, 32, 128, dtype=dtypes.half), Tensor.ones(1, 1, 32, 128, dtype=dtypes.half) * 3, dim=2)
    old = os.environ.get("AMD_FLASH_DIRECT")
    from tinygrad.helpers import getenv
    os.environ["AMD_FLASH_DIRECT"] = "1"
    getenv.cache_clear()
    try: out = flash_attention(q.half(), Tensor.stack(k, v), 64).numpy()
    finally:
      if old is None: os.environ.pop("AMD_FLASH_DIRECT", None)
      else: os.environ["AMD_FLASH_DIRECT"] = old
      getenv.cache_clear()
    expected = np.array([(32 + (i+1)*3) / (33+i) for i in range(32)], dtype=np.float32)[None, None, :, None]
    np.testing.assert_allclose(out, np.broadcast_to(expected, out.shape), rtol=2e-3, atol=2e-3)

  def test_generic_sdpa_gqa_strided_values(self):
    rng = np.random.default_rng(10618)
    cases = ((2, 1, 64, 32, 64), (3, 1, 64, 32, 64), (4, 1, 128, 32, 64), (8, 2, 64, 32, 64),
             (8, 2, 32, 1, 37), (16, 4, 64, 16, 192), (32, 8, 128, 32, 256), (4, 2, 256, 64, 192), (6, 2, 96, 17, 65))
    for heads, kv_heads, dim, tokens, length in cases:
      with self.subTest(heads=heads, kv_heads=kv_heads, dim=dim, tokens=tokens, length=length):
        qn = rng.normal(0, 1, (1, heads, tokens, dim)).astype(np.float16)
        kn, vn = (rng.normal(0, 1, (1, kv_heads, length, dim)).astype(np.float16) for _ in range(2))
        q, k, v = (Tensor(x).realize() for x in (qn, kn, vn))
        keys, values = (np.repeat(x.astype(np.float64), heads // kv_heads, axis=1) for x in (kn, vn))
        score = qn.astype(np.float64) @ keys.swapaxes(-1, -2) / np.sqrt(dim)
        probability = np.exp(score - score.max(-1, keepdims=True))
        expected = probability / probability.sum(-1, keepdims=True) @ values
        runner = TinyJit(lambda: q.scaled_dot_product_attention(k, v, enable_gqa=True).realize())
        for _ in range(4):
          actual = runner().numpy()
          self.assertTrue(np.isfinite(actual).all())
          np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=2e-3)

  def test_prefill_attention_nonzero_asymmetric_qk(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    from tinygrad.helpers import getenv
    from tinygrad.codegen import to_program_cache
    Tensor.manual_seed(7)
    q = Tensor.randn(1, 2, 32, 128, dtype=dtypes.half).realize()
    k = Tensor.randn(1, 1, 64, 128, dtype=dtypes.half).realize()
    v = Tensor.randn(1, 1, 64, 128, dtype=dtypes.half).realize()
    old = {key: os.environ.get(key) for key in ("AMD_FLASH_DIRECT", "AMD_FLASH_ACC_SEP", "AMD_WMMA_ACC_SMALL")}
    os.environ.update(AMD_FLASH_DIRECT="1", AMD_FLASH_ACC_SEP="1", AMD_WMMA_ACC_SMALL="0")
    getenv.cache_clear()
    to_program_cache.clear()
    try:
      out = flash_attention(q, Tensor.stack(k, v), 64).realize()
    finally:
      for key, value in old.items():
        if value is None: os.environ.pop(key, None)
        else: os.environ[key] = value
      getenv.cache_clear()
      to_program_cache.clear()
    mask = Tensor.full((1, 1, 32, 64), float("-inf"), dtype=q.dtype, buffer=False).triu(33)
    expected = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)
    np.testing.assert_allclose(out.numpy(), expected.numpy(), rtol=5e-3, atol=5e-3)

  @unittest.skipIf(OSX and DEV.interface.startswith("MOCK"), "TODO: incorrect results in the macOS AMD emulator")
  def test_flash_attention_decode_symbolic_gqa(self):
    with patch.object(Tensor, "scaled_dot_product_attention", side_effect=AssertionError("expected custom decode")):
      self._test_flash_decode(8, 2, 256, 128, 37, symbolic=True)

  def test_flash_attention_decode_gqa_tail(self): self._test_flash_decode(3, 1, 192, 64, 37)

  def test_flash_attention_decode_gqa_output_layout(self): self._test_flash_decode(4, 1, 128, 256, 3)

  def test_flash_attention_decode_llama1b_shapes(self):
    for valid in (13, 32, 65, 1025):
      with self.subTest(valid=valid): self._test_flash_decode(32, 8, 64, 2048, valid, symbolic=True)

  def test_flash_attention_decode_gqa4_long_jit(self):
    if not amd_direct_isa(Tensor.empty(1).device): self.skipTest("RDNA3 DIRECT required")
    from tinygrad.helpers import getenv
    def patterned(shape, dtype=np.float32):
      size = int(np.prod(shape))
      return (((np.arange(size, dtype=np.int32) % 127) - 63) / 64.0).astype(dtype).reshape(shape)
    q_np, cache_np = patterned((1, 32, 1, 128), np.float16), patterned((2, 1, 8, 2048, 128), np.float16)
    k = np.repeat(cache_np[0, 0].astype(np.float32), 4, axis=0)
    v = np.repeat(cache_np[1, 0].astype(np.float32), 4, axis=0)
    scores = np.einsum("hd,hnd->hn", q_np[0, :, 0].astype(np.float32), k) / np.sqrt(128)
    probs = np.exp(scores - scores.max(-1, keepdims=True))
    expected = np.einsum("hn,hnd->hd", probs / probs.sum(-1, keepdims=True), v)[None, :, None]
    q, cache = Tensor(q_np).realize(), Tensor(cache_np).realize()
    with patch.dict(os.environ, {"AMD_FLASH_DIRECT": "1"}):
      getenv.cache_clear()
      try:
        run = TinyJit(lambda qq: flash_attention(qq, cache, 2048).realize())
        for _ in range(3): out = run(q)
        np.testing.assert_allclose(out.numpy(), expected, rtol=2e-4, atol=2e-4)
      finally: getenv.cache_clear()

  @unittest.skipIf(OSX and DEV.interface.startswith("MOCK"), "TODO: incorrect results in the macOS AMD emulator")
  def test_flash_attention_decode_large_gqa_group(self): self._test_flash_decode(8, 1, 256, 256, 73)

  def _test_flash_decode(self, heads, kv_heads, dim, n, valid, symbolic=False):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    q = rng.normal(size=(1, heads, 1, dim)).astype(np.float16)
    cache = rng.normal(size=(2, 1, kv_heads, n, dim)).astype(np.float16)
    k, v = (np.repeat(c[0, :, :valid].astype(np.float32), heads//kv_heads, axis=0) for c in cache)
    scores = q[0].astype(np.float32) @ k.transpose(0, 2, 1) / np.sqrt(dim)
    probs = np.exp(scores - scores.max(-1, keepdims=True))
    expected = (probs / probs.sum(-1, keepdims=True)) @ v
    cache_tensor = Tensor(cache)
    if symbolic:
      start_pos = UOp.variable("start_pos", 0, n-1).bind(valid-1)
      valid = start_pos + 1
      cache_tensor = cache_tensor.realize()
    np.testing.assert_allclose(flash_attention(Tensor(q), cache_tensor, valid).numpy(), expected[None], rtol=2e-3, atol=2e-3)

  def test_attention_symbolic_length_jit(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3/4 required")
    cache = np.zeros((2, 1, 1, 128, 128), dtype=np.float16)
    cache[1] = np.arange(128, dtype=np.float16)[:, None]
    kv = Tensor(cache).realize()
    for tokens in (1, 32):
      q = Tensor(np.zeros((1, 2, tokens, 128), dtype=np.float16)).realize()
      run = TinyJit(lambda q, kv, end: flash_attention(q, kv, end).realize())
      for valid in (tokens, 33, 97, 128, 64, tokens):
        with self.subTest(tokens=tokens, valid=valid):
          end = UOp.variable("valid_length", tokens, 128).bind(valid)
          expected = np.broadcast_to(np.arange(valid-tokens, valid)[None, None, :, None]/2, q.shape)
          np.testing.assert_allclose(run(q, kv, end).numpy(), expected, rtol=1e-3, atol=1e-3)

  def test_prefill_attention_random(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3/4 required")
    rng = np.random.default_rng(42)
    cases = ((2, 4, 2, 32, 64, 45, 64, False), (1, 3, 1, 64, 128, 83, 128, False),
             (1, 2, 1, 96, 256, 149, 192, False), (1, 4, 2, 17, 128, 128, 128, True), (1, 2, 1, 3, 64, 33, 64, True),
             (2, 3, 1, 64, 96, 99, 128, False), (1, 2, 1, 32, 320, 33, 64, False))
    for batch, heads, kv_heads, tokens, dim, valid, capacity, symbolic in cases:
      with self.subTest(batch=batch, tokens=tokens, dim=dim, symbolic=symbolic):
        q = rng.normal(size=(batch, heads, tokens, dim)).astype(np.float16)
        cache = rng.normal(size=(2, batch, kv_heads, capacity, dim)).astype(np.float16)
        cache[:, :, :, valid:] = np.nan
        k, v = (np.repeat(c[:, :, :valid].astype(np.float32), heads//kv_heads, axis=1) for c in cache)
        scores = q.astype(np.float32) @ k.swapaxes(-1, -2) / np.sqrt(dim)
        scores = np.where(np.arange(valid)[None, :] <= np.arange(valid-tokens, valid)[:, None], scores, -np.inf)
        probs = np.exp(scores - scores.max(-1, keepdims=True))
        expected = (probs / probs.sum(-1, keepdims=True)) @ v
        query, kv = Tensor(q), Tensor(cache).realize()
        end = UOp.variable("valid", 1, capacity).bind(valid)
        if symbolic:
          count = UOp.variable("query_tokens", 1, 32).bind(tokens)
          query = Tensor(np.pad(q, ((0, 0), (0, 0), (0, 32-tokens), (0, 0)))).contiguous()[:, :, :count]
          @function(allow_implicit=True)
          def run(q:Tensor, kv:Tensor, end:UOp): return flash_attention(q, kv, end)
        else: run = flash_attention
        from tinygrad.helpers import getenv
        with patch.dict(os.environ, {"AMD_FLASH_DIRECT": "1"}):
          getenv.cache_clear()
          try:
            if (amd_direct_isa(query.device) and (dim == 64 or (valid-tokens) % 32 != 0)) or (dim == 320 and not amd_direct_isa(query.device)):
              actual = run(query, kv, end)[:, :, :tokens].numpy()
            else:
              with patch.object(Tensor, "scaled_dot_product_attention", side_effect=AssertionError("expected WMMA attention")):
                actual = run(query, kv, end)[:, :, :tokens].numpy()
          finally: getenv.cache_clear()
        np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=2e-3)

  def test_prefill_attention_direct_random_aligned(self):
    if not amd_direct_isa(Tensor.empty(1).device): self.skipTest("RDNA3 DIRECT required")
    rng = np.random.default_rng(43)
    batch, heads, kv_heads, tokens, dim, valid = 1, 4, 1, 32, 128, 2048
    q = rng.normal(size=(batch, heads, tokens, dim)).astype(np.float16)
    cache = rng.normal(size=(2, batch, kv_heads, valid, dim)).astype(np.float16)
    k, v = (np.repeat(c.astype(np.float32), heads//kv_heads, axis=1) for c in cache)
    scores = q.astype(np.float32) @ k.swapaxes(-1, -2) / np.sqrt(dim)
    scores = np.where(np.arange(valid)[None, :] <= np.arange(valid-tokens, valid)[:, None], scores, -np.inf)
    probs = np.exp(scores - scores.max(-1, keepdims=True))
    expected = (probs / probs.sum(-1, keepdims=True)) @ v
    from tinygrad.helpers import getenv
    with patch.dict(os.environ, {"AMD_FLASH_DIRECT": "1"}):
      getenv.cache_clear()
      try:
        with patch.object(Tensor, "scaled_dot_product_attention", side_effect=AssertionError("expected DIRECT flash")):
          actual = flash_attention(Tensor(q), Tensor(cache).realize(), valid).numpy()
      finally: getenv.cache_clear()
    np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=2e-3)

  def test_prefill_attention_direct_d64_uses_custom_kernel(self):
    if not amd_direct_isa(Tensor.empty(1).device): self.skipTest("RDNA3 DIRECT required")
    rng = np.random.default_rng(44)
    q = rng.normal(size=(1, 4, 32, 64)).astype(np.float16)
    cache = rng.normal(size=(2, 1, 1, 64, 64)).astype(np.float16)
    k, v = (np.repeat(cache[i].astype(np.float32), 4, axis=1) for i in range(2))
    scores = (q.astype(np.float32) @ k.swapaxes(-1, -2)) / np.sqrt(64)
    scores = np.where(np.arange(64)[None, :] <= np.arange(32, 64)[:, None], scores, -np.inf)
    probs = np.exp(scores - scores.max(-1, keepdims=True))
    expected = (probs / probs.sum(-1, keepdims=True)) @ v
    from tinygrad.helpers import getenv
    with patch.dict(os.environ, {"AMD_FLASH_ACC_SMALL": "1", "AMD_WMMA_ACC_SMALL": "0"}):
      getenv.cache_clear()
      try:
        with patch.object(Tensor, "scaled_dot_product_attention", side_effect=AssertionError("expected D64 WMMA attention")):
          actual = flash_attention(Tensor(q), Tensor(cache).realize(), 64).numpy()
      finally: getenv.cache_clear()
    np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=2e-3)

  def test_direct_flash_jit_repeated_submit(self):
    if not amd_direct_isa(Tensor.empty(1).device): self.skipTest("RDNA3 DIRECT required")
    from tinygrad.helpers import getenv
    rng = np.random.default_rng(71)
    q = Tensor(rng.normal(size=(1, 32, 32, 128)).astype(np.float16)).realize()
    cache = Tensor(rng.normal(size=(2, 1, 8, 2048, 128)).astype(np.float16)).realize()
    with patch.dict(os.environ, {"AMD_FLASH_DIRECT": "1"}):
      getenv.cache_clear()
      try:
        run = TinyJit(lambda qq: flash_attention(qq, cache, 2048).realize())
        for _ in range(3): reference = run(q).numpy()
        for _ in range(3): out = run(q)
        np.testing.assert_allclose(out.numpy(), reference, rtol=0, atol=0)
      finally: getenv.cache_clear()

  def test_prefill_attention_nonfinite_cache_tail(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    q = Tensor.zeros(1, 2, 32, 128, dtype=dtypes.half)
    values = rng.normal(size=(33, 128)).astype(np.float16)
    expected = np.stack([values[:i+2].astype(np.float32).mean(0) for i in range(32)])[None, None].repeat(2, axis=1)
    for tail in (np.nan, np.inf, -np.inf):
      with self.subTest(tail=tail):
        cache = np.full((2, 1, 1, 64, 128), tail, dtype=np.float16)
        cache[0, :, :, :33] = 0
        cache[1, :, :, :33] = values
        valid = UOp.variable("valid_end", 32, 64).bind(33)
        cache_tensor = Tensor(cache).realize()
        assigned = Tensor(cache_tensor.uop.after(Tensor(valid).uop))
        out = flash_attention(q, assigned, valid)
        np.testing.assert_allclose(out.numpy(), expected, rtol=2e-3, atol=2e-3)

  def test_flash_attention_decode_beyond_256_chunks(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    n = 257 * 64
    q = Tensor.zeros(1, 1, 1, 32, dtype=dtypes.half).realize()
    k = Tensor.zeros(1, 1, n, 32, dtype=dtypes.half)
    v = Tensor.zeros(1, 1, n-64, 32, dtype=dtypes.half).cat(Tensor.ones(1, 1, 64, 32, dtype=dtypes.half), dim=2)
    cache = Tensor.stack(k, v).contiguous().realize()
    for valid, expected in ((1, 0), (n, 1/257)):
      with self.subTest(valid=valid):
        valid_kv_len = UOp.variable("valid_kv_len", 1, n).bind(valid)
        assigned = Tensor(cache.uop.after(Tensor(valid_kv_len).uop))
        np.testing.assert_allclose(flash_attention(q, assigned, valid_kv_len).numpy(), expected, rtol=2e-3, atol=2e-4)

  def test_flash_attention_decode_long_context_random(self):
    self._test_flash_decode(8, 2, 128, 257*64, 257*64-13)  # past 256 chunks, with a ragged tail

  def test_flash_attention_decode_chunk_round_accumulator_range(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    valid_kv_len, max_kv_len = 6749, 6784  # three chunk rounds, with a ragged tail
    q = Tensor.zeros(1, 8, 1, 32, dtype=dtypes.half).realize()
    cache = Tensor.stack(Tensor.zeros(1, 1, max_kv_len, 32, dtype=dtypes.half),
                         Tensor.full((1, 1, max_kv_len, 32), 5500, dtype=dtypes.half)).contiguous().realize()
    np.testing.assert_allclose(flash_attention(q, cache, valid_kv_len).numpy(), 5500, rtol=2e-3, atol=2e-3)

  def test_prefill_attention_unaligned_start(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    start_pos = 1718
    q = Tensor.zeros(1, 8, 32, 128)
    old_kv = rng.normal(size=(2, 1, 1, start_pos, 128)).astype(np.float32)
    new_kv = rng.normal(size=(2, 1, 1, 32, 128)).astype(np.float32)
    cache = Tensor.zeros(2, 1, 1, 2048, 128, dtype=dtypes.half).contiguous()
    Tensor.realize(cache[:, :, :, :start_pos].assign(Tensor(old_kv).cast(dtypes.half)))
    sp = UOp.variable("start_pos", 0, 2047).bind(start_pos)
    assigned = Tensor(cache.uop.after(cache[:, :, :, sp:sp+32, :].uop.store(Tensor(new_kv).cast(dtypes.half).uop)))
    out = flash_attention(q, assigned, sp+32).realize()
    values = np.concatenate([old_kv[1, 0, 0], new_kv[1, 0, 0]]).astype(np.float16).astype(np.float32)
    expected = np.stack([values[:start_pos+i+1].mean(0) for i in range(32)])[None, None].repeat(8, axis=1)
    np.testing.assert_allclose(out.numpy(), expected, rtol=2e-3, atol=2e-3)

  def test_decode_realizes_dependencies_before_flash_policy(self):
    from tinygrad.helpers import getenv
    from tinygrad.llm.kernels import amd as amd_mod
    q, kv, out = Tensor.empty(1, 2, 1, 64, device="CPU"), Tensor.empty(2, 1, 1, 64, 64, device="CPU"), Tensor.empty(1, device="CPU")
    calls = []
    def realize(tensor, *others):
      calls.append(((tensor, *others), getenv("AMD_VOPD_FMAC", 0)))
      return tensor
    with patch.dict(os.environ):
      for key in amd_mod._FLASH_DECODE_ONLY: os.environ.pop(key, None)
      getenv.cache_clear()
      try:
        with patch.object(amd_mod, "amd_direct_isa", return_value=True), \
             patch.object(amd_mod, "amd_flash_attention_decode", return_value=out), \
             patch.object(Tensor, "realize", autospec=True, side_effect=realize):
          self.assertIs(amd_mod.flash_attention(q, kv, 64), out)
        self.assertEqual(calls, [((q, kv), 0), ((out,), 0)])
      finally: getenv.cache_clear()

  def test_decode_builds_without_device_execution(self):
    from tinygrad.helpers import Context
    from tinygrad.llm.kernels import amd as amd_mod
    q, kv, out = Tensor.empty(1, 2, 1, 64, device="CPU"), Tensor.empty(2, 1, 1, 64, 64, device="CPU"), Tensor.empty(1, device="CPU")
    with patch.object(amd_mod, "amd_direct_isa", return_value=True), \
         patch.object(amd_mod, "amd_flash_attention_decode", return_value=out), \
         patch.object(Tensor, "realize", side_effect=AssertionError("execution during graph capture")), \
         Context(ALLOW_DEVICE_USAGE=0):
      self.assertIs(amd_mod.flash_attention(q, kv, 64), out)

  def test_flash_attention_function_capture(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3/4 required")
    from tinygrad.helpers import getenv
    rng = np.random.default_rng(73)
    cache = rng.normal(size=(2, 1, 1, 128, 128)).astype(np.float16)
    kv = Tensor(cache).realize()
    keys, values = (np.repeat(cache[i].astype(np.float32), 4, axis=1) for i in range(2))
    with patch.dict(os.environ, {"AMD_FLASH_DIRECT":"1"}):
      getenv.cache_clear()
      try:
        for tokens in (1, 32):
          q_np = rng.normal(size=(1, 4, tokens, 128)).astype(np.float16)
          q = Tensor(q_np).realize()
          @function(allow_implicit=True)
          def wrapped(query:Tensor, cache:Tensor, end:int|UOp): return flash_attention(query, cache, end)
          run = TinyJit(lambda query, cache, end: wrapped(query, cache, end).realize())
          for valid in ((37, 37, 73, 37, 127) if tokens == 1 else (128, 128, 128)):
            with self.subTest(tokens=tokens, valid=valid):
              end = UOp.variable("function_valid", 1, 128).bind(valid) if tokens == 1 else valid
              actual = run(q, kv, end).numpy()
              scores = (q_np.astype(np.float32) @ keys[:, :, :valid].swapaxes(-1, -2))/np.sqrt(128)
              scores = np.where(np.arange(valid)[None, :] <= np.arange(valid-tokens, valid)[:, None], scores, -np.inf)
              probs = np.exp(scores-scores.max(axis=-1, keepdims=True))
              expected = (probs/probs.sum(axis=-1, keepdims=True)) @ values[:, :, :valid]
              np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=2e-3)
      finally: getenv.cache_clear()

  def test_prefill_compile_policy_overrides_sticky_decode(self):
    """Flash scheduling is scoped to compilation and prefill overrides stale decode settings."""
    from tinygrad.helpers import getenv
    from tinygrad.llm.kernels import amd as amd_mod
    keys = ("AMD_REG_PROMOTE_SKIP_SLOTS", "AMD_FLASH_PREFILL_SKIP_SLOTS", "AMD_FLASH_ACC_SMALL", "AMD_WMMA_ACC_SMALL",
            "AMD_VOPD_FMAC", "AMD_VOPD_FMAC_ALU", "TC_LDS_AB", "AMD_WMMA_ACC_BASE", "AMD_WMMA_ACC_N",
            "ALLOW_UPCAST16", "AMD_REMAT_ADDR", "AMD_WMMA_INTERLEAVE")
    old = {k: os.environ.get(k) for k in keys}
    try:
      # Simulate stale skip/schedule settings; ACC_SMALL selects the prefill promotion policy.
      os.environ["AMD_REG_PROMOTE_SKIP_SLOTS"] = ""
      os.environ.pop("AMD_FLASH_PREFILL_SKIP_SLOTS", None)
      os.environ["AMD_FLASH_ACC_SMALL"] = "0"
      os.environ["AMD_WMMA_ACC_SMALL"] = "0"
      os.environ["AMD_VOPD_FMAC"] = "1"
      os.environ["AMD_VOPD_FMAC_ALU"] = "1"
      os.environ.pop("AMD_WMMA_INTERLEAVE", None)
      before = {k:os.environ.get(k) for k in keys}
      getenv.cache_clear()
      with amd_mod._flash_direct_compile_env():
        self.assertEqual(getenv("AMD_FLASH_ACC_SMALL", 0), 1)
        self.assertEqual(getenv("AMD_WMMA_INTERLEAVE", 0), 1)
        self.assertEqual(os.environ.get("AMD_REG_PROMOTE_SKIP_SLOTS"), "")
        self.assertNotIn("AMD_VOPD_FMAC", os.environ)
        self.assertNotIn("AMD_VOPD_FMAC_ALU", os.environ)
        self.assertEqual(getenv("AMD_WMMA_ACC_BASE"), 145)
        self.assertEqual(getenv("AMD_WMMA_ACC_N"), 48)
        self.assertEqual(getenv("ALLOW_UPCAST16"), 0)
      self.assertEqual({k:os.environ.get(k) for k in keys}, before)
      # Decode-only options are active during compilation but must not leak into later kernels.
      os.environ.pop("AMD_VOPD_FMAC", None)
      os.environ.pop("AMD_VOPD_FMAC_ALU", None)
      before = {k:os.environ.get(k) for k in keys}
      getenv.cache_clear()
      with amd_mod._flash_direct_compile_env(flash_acc_small=False):
        self.assertEqual(getenv("AMD_VOPD_FMAC"), 0)
        self.assertEqual(getenv("ALLOW_UPCAST16"), 1)
      self.assertEqual({k:os.environ.get(k) for k in keys}, before)
      with self.assertRaisesRegex(RuntimeError, "compile failed"):
        with amd_mod._flash_direct_compile_env(): raise RuntimeError("compile failed")
      self.assertEqual({k:os.environ.get(k) for k in keys}, before)
    finally:
      for k, v in old.items():
        if v is None: os.environ.pop(k, None)
        else: os.environ[k] = v
      getenv.cache_clear()

  def test_flash_k_unroll_is_cache_key(self):
    """Ranged and factor-two mappings must not share a cached graph."""
    _amd_flash_attention.cache_clear()
    o = UOp.placeholder((32, 32, 128), dtypes.float32, 0)
    q = UOp.placeholder((32, 32, 128), dtypes.float16, 1)
    kv = UOp.placeholder((2, 1, 8, 64, 128), dtypes.float16, 2)
    a = _amd_flash_attention(o, q, kv, valid_kv_len=64, acc_small=True, k_unroll=0)
    b = _amd_flash_attention(o, q, kv, valid_kv_len=64, acc_small=True, k_unroll=2)
    self.assertIsNot(a, b)
    # Same key hits cache.
    self.assertIs(a, _amd_flash_attention(o, q, kv, valid_kv_len=64, acc_small=True, k_unroll=0))
    for factor in (-1, 1, 4, 8):
      with self.subTest(factor=factor), self.assertRaisesRegex(AssertionError, "validated"):
        _amd_flash_attention(o, q, kv, valid_kv_len=64, acc_small=True, k_unroll=factor)

  def test_flash_graph_policy_is_explicit(self):
    # Compiler partition settings must not silently change an already-cached kernel graph.
    from unittest.mock import patch
    import os
    from tinygrad.helpers import getenv
    args = (UOp.placeholder((32, 32, 128), dtypes.float32, 0), UOp.placeholder((32, 32, 128), dtypes.float16, 1),
            UOp.placeholder((2, 1, 8, 64, 128), dtypes.float16, 2))
    graphs = []
    try:
      for setting in ("0", "1"):
        with patch.dict(os.environ, {"AMD_FLASH_ACC_SMALL": setting, "AMD_WMMA_ACC_SMALL": setting}), \
             patch("tinygrad.llm.kernels.amd._reg_swizzle_slot", 64):
          getenv.cache_clear()
          _amd_flash_attention.cache_clear()
          graphs.append(_amd_flash_attention(*args, valid_kv_len=64, acc_small=False))
      self.assertIs(graphs[0], graphs[1])
    finally:
      getenv.cache_clear()
      _amd_flash_attention.cache_clear()

if __name__ == "__main__": unittest.main()

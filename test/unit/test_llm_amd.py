import functools, unittest
import numpy as np
from tinygrad import Tensor, UOp, dtypes, nn, function
from tinygrad.llm.kernels.amd import Linear, _amd_flash_attention, amd_custom_kernels_supported, q8_quantize, flash_attention
from tinygrad.llm.gguf import ggml_data_to_tensor

class TestQ8Quantize(unittest.TestCase):
  def test_word_quant_weights_use_typed_buffer_view(self):
    for ggml_type, type_size in ((13, 176), (23, 136)):
      with self.subTest(ggml_type=ggml_type):
        raw = Tensor(np.zeros(type_size + 4, dtype=np.uint8), device="CPU").contiguous().realize()[4:]
        decoded = ggml_data_to_tensor(raw, 256, ggml_type).reshape(1, 256)
        linear = Linear(256, 1, bias=False)
        linear.set_quantized(decoded)
        self.assertEqual(linear.ggml_type, ggml_type)
        self.assertEqual(linear.weight.dtype, dtypes.uint32)
        self.assertEqual(linear.weight.nbytes(), type_size)
        self.assertEqual(linear.weight.uop.buf_uop.buffer.offset, 4)

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

  def test_q6_linear_compiles_in_function(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    packed = rng.integers(0, 256, 210, dtype=np.uint8)
    packed[-2:] = np.array([0.01], dtype=np.float16).view(np.uint8)
    raw = Tensor(np.pad(packed, (4, 0))).contiguous().realize()[4:]
    decoded = ggml_data_to_tensor(raw, 256, 14).reshape(1, 256)
    linear = Linear(256, 1, bias=False)
    nn.state.load_state_dict(linear, {"weight":decoded}, verbose=False, realize=False)
    @function(allow_implicit=True)
    def run(x:Tensor): return linear(x)
    self.assertTrue(np.isfinite(run(Tensor.randn(1, 256)).realize().item()))
    # the Q6 weight is repacked: 210-byte blocks padded to 212 (one block = 53 words)
    self.assertEqual(linear.weight.uop.buf_uop.buffer.nbytes, 53*4)
    self.assertEqual(linear.weight.dtype, dtypes.uint32)

  def test_q4_k_linear(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    in_features, blocks = 2048, 16*2048//256
    packed = rng.integers(0, 256, blocks*144, dtype=np.uint8)
    for i in range(blocks): packed[i*144:i*144+4] = np.array([0.01, 0.002], dtype=np.float16).view(np.uint8)
    raw = Tensor(np.pad(packed, (4, 0))).contiguous().realize()[4:]
    decoded = ggml_data_to_tensor(raw, 16*in_features, 12).reshape(16, in_features)
    weight = decoded.numpy()
    linear = Linear(in_features, 16, bias=False)
    nn.state.load_state_dict(linear, {"weight":decoded}, verbose=False, realize=False)
    x = rng.normal(size=(3, in_features)).astype(np.float32)
    scale = np.maximum(np.abs(x).reshape(3, in_features//32, 32).max(-1, keepdims=True) / 127, 1e-8)
    xq = np.clip(np.rint(x.reshape(3, in_features//32, 32) / scale), -127, 127) * scale
    np.testing.assert_allclose(linear(Tensor(x)).numpy(), xq.reshape(3, in_features) @ weight.T, rtol=2e-3, atol=2e-2)
    self.assertEqual(linear.ggml_type, 12)

  def _test_quant_linear(self, ggml_type:int, block_bytes:int, tokens:int=3):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng, in_features, out_features = np.random.default_rng(42), 2048, 16
    blocks = out_features * in_features // 256
    packed = rng.integers(0, 256, blocks*block_bytes, dtype=np.uint8).reshape(blocks, block_bytes)
    if ggml_type in (12, 13): packed[:, :4] = np.array([0.01, 0.002], dtype=np.float16).view(np.uint8)
    elif ggml_type == 14: packed[:, -2:] = np.array([0.01], dtype=np.float16).view(np.uint8)
    else: packed[:, :2] = np.array([0.01], dtype=np.float16).view(np.uint8)
    raw = Tensor(np.pad(packed.reshape(-1), (4, 0))).contiguous().realize()[4:]
    decoded = ggml_data_to_tensor(raw, out_features*in_features, ggml_type).reshape(out_features, in_features)
    weight = decoded.numpy()
    linear = Linear(in_features, out_features, bias=False)
    nn.state.load_state_dict(linear, {"weight":decoded}, verbose=False, realize=False)
    x = rng.normal(size=(tokens, in_features)).astype(np.float32)
    if tokens % 16 == 0:
      xq, weight = x.astype(np.float16).astype(np.float32), weight.astype(np.float16).astype(np.float32)
    else:
      scale = np.maximum(np.abs(x).reshape(tokens, in_features//32, 32).max(-1, keepdims=True) / 127, 1e-8)
      xq = (np.clip(np.rint(x.reshape(tokens, in_features//32, 32) / scale), -127, 127) * scale).reshape(x.shape)
    np.testing.assert_allclose(linear(Tensor(x)).numpy(), xq @ weight.T, rtol=2e-3, atol=2e-2)
    self.assertEqual(linear.ggml_type, ggml_type)

  def test_q5_k_linear(self): self._test_quant_linear(13, 176)

  def test_iq4_xs_linear(self): self._test_quant_linear(23, 136)

  def test_quant_linear_wmma(self):
    for ggml_type,block_bytes in ((12, 144), (13, 176), (14, 210), (23, 136)):
      with self.subTest(ggml_type=ggml_type): self._test_quant_linear(ggml_type, block_bytes, tokens=16)

  def test_q6_linear_multiple_tokens(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    in_features, blocks = 2048, 16*2048//256
    packed = rng.integers(0, 256, blocks*210, dtype=np.uint8)
    for i in range(blocks): packed[i*210+208:i*210+210] = np.array([0.01], dtype=np.float16).view(np.uint8)
    raw = Tensor(np.pad(packed, (4, 0))).contiguous().realize()[4:]
    decoded = ggml_data_to_tensor(raw, 16*in_features, 14).reshape(16, in_features)
    weight = decoded.numpy()
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

  def test_attention_uses_physical_cache_length(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    q, k, v = Tensor.zeros(1, 2, 1, 32), Tensor.randn(1, 1, 1, 32), Tensor.randn(1, 1, 1, 32)
    cache = Tensor.empty(2, 1, 1, 256, 32, dtype=dtypes.half).contiguous()
    assigned = Tensor(cache.uop.after(cache[:, :, :, 0:1, :].uop.store(Tensor.stack(k, v).cast(dtypes.half).uop)))
    out = flash_attention(q, assigned, 1).realize()
    np.testing.assert_allclose(out.numpy(), v.expand(1, 2, 1, 32).numpy(), rtol=2e-2, atol=2e-2)

  def test_attention_decode_multiple_chunks_and_gqa_heads(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng, valid = np.random.default_rng(42), 129
    q_np, kv = rng.normal(size=(1, 32, 1, 64)).astype(np.float32), rng.normal(size=(2, 1, 8, valid, 64)).astype(np.float32)
    q = Tensor(q_np)
    cache = Tensor.empty(2, 1, 8, 256, 64, dtype=dtypes.half).contiguous()
    assigned = Tensor(cache.uop.after(cache[:, :, :, :valid, :].uop.store(Tensor(kv).cast(dtypes.half).uop)))
    out = flash_attention(q, assigned, valid).realize().numpy()
    qh = q_np.astype(np.float16).astype(np.float32)
    kh, vh = (np.repeat(x.astype(np.float16).astype(np.float32), 4, axis=1) for x in kv)
    scores = np.einsum("bhqd,bhkd->bhqk", qh, kh) / np.sqrt(64)
    probs = np.exp(scores - scores.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    expected = np.einsum("bhqk,bhkd->bhqd", probs, vh)
    np.testing.assert_allclose(out, expected, rtol=5e-3, atol=5e-3)

  def test_prefill_attention_multiple_tiles(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    q = Tensor.zeros(1, 4, 32, 64)
    k = Tensor.zeros(1, 1, 64, 64, dtype=dtypes.half)
    v = Tensor.cat(Tensor.ones(1, 1, 32, 64, dtype=dtypes.half), Tensor.ones(1, 1, 32, 64, dtype=dtypes.half) * 3, dim=2)
    out_buf = Tensor.empty(4, 32, 64, dtype=dtypes.float32)
    out = Tensor.custom_kernel(out_buf, q.half().reshape(4, 32, 64), Tensor.stack(k, v),
                               fxn=functools.partial(_amd_flash_attention, valid_kv_len=64))[0].reshape(1, 4, 32, 64).numpy()
    expected = np.array([(32 + (i+1)*3) / (33+i) for i in range(32)], dtype=np.float32)[None, None, :, None]
    np.testing.assert_allclose(out, np.broadcast_to(expected, out.shape), rtol=2e-3, atol=2e-3)

  def test_flash_attention_decode_gqa_output_layout(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    Tensor.manual_seed(42)
    q = Tensor.randn(1, 4, 1, 128, dtype=dtypes.half).realize()
    cache = Tensor.randn(2, 1, 1, 256, 128, dtype=dtypes.half).realize()
    out = flash_attention(q, cache, 3).realize()
    expected = q.scaled_dot_product_attention(cache[0, :, :, :3], cache[1, :, :, :3], enable_gqa=True)
    np.testing.assert_allclose(out.numpy(), expected.numpy(), rtol=2e-3, atol=2e-3)

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

if __name__ == "__main__": unittest.main()

import unittest
from unittest.mock import patch
from tinygrad import Tensor, dtypes
from tinygrad.llm.kernels.amd import Linear, Q6_K, q8_linear

class TestQuantLinearSharding(unittest.TestCase):
  def test_sharded_quant_kernel_shapes(self):
    # No GPU execution: verify the per-device kernel contract and the reconstructed output.
    devices = ("CPU", "CPU:1")
    for axis in (0, 1):
      with self.subTest(weight_axis=axis):
        layer = Linear(512, 32, bias=False)
        layer.ggml_type, layer.shard_axis = Q6_K, axis
        layer.weight = Tensor.empty(32*512//256*105, dtype=dtypes.uint16, device=devices, axis=0)
        x = Tensor.empty(16, 512, device=devices, axis=1 if axis == 1 else None)
        calls = []
        def capture(*args, **kwargs):
          calls.append((args[0].shape, args[0].uop.axis, kwargs["fxn"].keywords))
          return args
        with patch("tinygrad.llm.kernels.amd._wmma_rdna4", return_value=False), patch.object(Tensor, "custom_kernel", side_effect=capture):
          result = q8_linear(layer, x)
        self.assertEqual(result.shape, (16, 32))
        self.assertEqual(result.uop.axis, None if axis == 1 else 1)
        self.assertEqual(calls[0][:2], ((32, 32), 0) if axis == 1 else ((16, 32), 1))
        self.assertEqual(calls[0][2]["in_features"], 256 if axis == 1 else 512)
        self.assertEqual(calls[0][2]["out_features"], 32 if axis == 1 else 16)

if __name__ == "__main__": unittest.main()

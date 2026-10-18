import unittest
from tinygrad import Tensor, dtypes
from tinygrad.llm.kernels.amd import _view_back

class TestLazyWeightViews(unittest.TestCase):
  def test_narrowing_float_cast_is_not_a_view(self):
    source = Tensor.empty(128, dtype=dtypes.float32, device="CPU")
    for dtype in (dtypes.float16, dtypes.bfloat16):
      with self.subTest(dtype=dtype):
        rounded = source.cast(dtype)
        self.assertIs(_view_back(rounded).uop, rounded.uop)
        self.assertIs(_view_back(rounded.float()).uop, rounded.uop)
    rounded = Tensor.empty(128, dtype=dtypes.float64, device="CPU").float()
    self.assertIs(_view_back(rounded).uop, rounded.uop)

  def test_exact_half_widening_can_be_a_view(self):
    for dtype in (dtypes.float16, dtypes.bfloat16):
      with self.subTest(dtype=dtype):
        source = Tensor.empty(128, dtype=dtype, device="CPU")
        self.assertIs(_view_back(source.float()).uop, source.uop)

if __name__ == "__main__": unittest.main()

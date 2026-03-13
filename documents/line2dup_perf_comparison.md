# line2Dup Performance Comparison

This note compares the original C++ implementation under `_third_party_shape_based_matching`
with the Python implementation in `line2dup_like_matcher.py`.

## Measurement scope

- C++: `shape_based_matching_test.exe`, timing from `Timer` around `detector.match(...)`
- Python: `Line2DupLikeDetector.match(...)`
- Excluded: template building, visualization drawing, image saving

## Match-time comparison

| Scene | C++ (s) | Python (s) | Python / C++ |
| --- | ---: | ---: | ---: |
| case0/1.jpg | 0.0388 | 4.6126 | 119.0x |
| case0/2.jpg | 0.0152 | 0.1246 | 8.2x |
| case0/3.png | 0.0069 | 0.0791 | 11.5x |
| case0/4.png | 0.0164 | 0.9046 | 55.0x |
| case0 total | 0.0772 | 5.7209 | 74.1x |
| case1/test.png | 0.0461 | 7.1231 | 154.7x |
| case2/test.png | 0.0397 | 4.7609 | 119.9x |
| all total | 0.1630 | 17.6049 | 108.0x |

## Notes

- Template counts are close enough that they do not explain the gap:
  - case0: C++ 89, Python 91
  - case1: C++ 361, Python 361
  - case2: C++ 361, Python 361
- The main Python hotspot is `similarity_local(...)`.
- The original C++ code uses:
  - linearized response-map memory layout
  - integer accumulation on contiguous buffers
  - MIPP SIMD kernels
- The Python code currently uses:
  - repeated NumPy slicing in the refinement loop
  - frequent `astype(np.float32)` conversions
  - a lower internal coarse threshold, which increases refinement candidates

## Main reasons Python is slower

1. Local refinement dominates runtime.
2. The Python implementation does not reproduce the C++ linearized memory path.
3. The C++ implementation uses SIMD for spread, response-map construction, and similarity accumulation.
4. Python keeps more candidates alive because the internal coarse threshold is lower.
5. Python does not reduce feature count on higher pyramid levels, unlike the C++ implementation.

## Optimization direction

1. Rebuild the Python matcher around linearized response memories.
2. Remove per-feature `astype(...)` in similarity accumulation.
3. Tighten coarse filtering to reduce `similarity_local(...)` calls.
4. Reduce feature count after `pyrDown()`.
5. Move the similarity kernels out of Python loops, ideally to C++/Cython/Numba.

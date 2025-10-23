This baseline corresponds to the Guided Non-Local Means algorithm for SAR image despeckling using MATLAB 2017a MEX functions with OpenCV.

Create a **math_compat.cpp** file with the following content to resolve missing math function references.
```cpp
#include <cmath>

extern "C" {
    double __log_finite(double x) { return log(x); }
    float __logf_finite(float x) { return logf(x); }
    double __exp_finite(double x) { return exp(x); }
    double __pow_finite(double x, double y) { return pow(x, y); }
    double __acos_finite(double x) { return acos(x); }
}
```

Navigate to the MATLAB directory and compile the required MEX functions.
The **removezeros.cpp** and the **math_compat.cpp** files uses OpenCV functions, so it needs to be linked with the OpenCV libraries.
```matlab
$ mex -glnxa64 -largeArrayDims -O -I../include removezeros.cpp math_compat.cpp ../lib_static_a64/libcv.a ../lib_static_a64/libcxcore.a -lm
```
It should create a **removezeros.mexa64** file. Then, compile the main GNLM function.
```matlab
$ mex -glnxa64 -largeArrayDims -O -v -D_GLIBCXX_USE_CXX11_ABI=0 -I../include -DGUIDA_NUM_BANDS=4 guidedNLMeans.cpp math_compat.cpp ../lib_static_a64/libcv.a ../lib_static_a64/libcxcore.a ../lib_static_a64/libopencv_lapack.a -lm -output guidedNLMeans_b04
```
It should create a **guidedNLMeans_b04.mexa64** file.

Finally, run the algorithm.
```bash
$ conda activate despeckling_gnlm

$ ~/miniforge3/envs/despeckling_gnlm/bin/python run.py \
    --input_path_sar ../../../dataset/fusion/sar.mat \
    --input_path_opt ../../../dataset/fusion/opt.mat \
    --matlab_script_path ../../../GNLM/matlab/guidedNLMeans.m \
    --L 1 \
    --stack_size 256 \
    --sharpness 0.002 \
    --balance 0.15 \
    --th_sar 2.0 \
    --block_size 8 \
    --win_size 39 \
    --stride 3
```
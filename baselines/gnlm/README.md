```bash
$ conda activate despeckling_glnm

$ c++ -O3 -shared -std=c++11 -fPIC \
    -D_GLIBCXX_USE_CXX11_ABI=0 \
    $(python3 -m pybind11 --includes) \
    -I../../../GNLM/include \
    denoise.cpp \
    -L../../../GNLM/lib_opencv210/glnxa64 \
    -lcxcore -lcv \
    -o guidedNLMeans$(python3-config --extension-suffix) \
    -lm

$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../../../GNLM/lib_opencv210/glnxa64

$ ~/miniforge3/envs/despeckling_gnlm/bin/python run.py \
    --input_path_sar ../../../dataset/fusion/sar.mat \
    --input_path_opt ../../../dataset/fusion/opt.mat \
    --L 1
```
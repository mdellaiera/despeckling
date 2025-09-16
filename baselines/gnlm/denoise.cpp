#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "opencv/cv.h"
#include "opencv/cxcore.h"
#include "opencv/highgui.h"
#include "../../../GNLM/src/GNLM.hpp"
namespace py = pybind11;

#ifndef GUIDA_NUM_BANDS
#define GUIDA_NUM_BANDS 4
#endif

typedef float PixelType;
typedef cv::Vec<PixelType, GUIDA_NUM_BANDS> PixelGuidaType;

// Helper: Convert numpy to OpenCV single-band
cv::Mat_<PixelType> numpy_to_cvmat(py::array_t<float> array) {
    auto buf = array.request();
    if (buf.ndim != 2)
        throw std::runtime_error("Input image must be 2D");
    return cv::Mat_<PixelType>(buf.shape[0], buf.shape[1], (PixelType*)buf.ptr);
}

// Helper: Convert numpy to OpenCV multi-band
cv::Mat_<PixelGuidaType> numpy_to_cvmat_guided(py::array_t<float> array) {
    auto buf = array.request();
    if (buf.ndim == 3 && buf.shape[2] == GUIDA_NUM_BANDS) {
        cv::Mat_<PixelGuidaType> out(buf.shape[0], buf.shape[1]);
        auto data_ptr = (PixelType*)buf.ptr;
        // Copy data: numpy is (H,W,B), OpenCV wants cv::Vec<float, B>
        for (ssize_t i = 0; i < buf.shape[0]; ++i)
            for (ssize_t j = 0; j < buf.shape[1]; ++j)
                for (ssize_t b = 0; b < GUIDA_NUM_BANDS; ++b)
                    out(i,j)[b] = *(data_ptr++); // assumes contiguous
        return out;
    } else if (buf.ndim == 2 && GUIDA_NUM_BANDS == 1) {
        // fallback for single-band guide
        return cv::Mat_<PixelGuidaType>(buf.shape[0], buf.shape[1], (PixelGuidaType*)buf.ptr);
    } else {
        throw std::runtime_error("Guide image must be (H,W,B) with correct band count");
    }
}

py::array_t<float> denoise(
    py::array_t<float> amplitude,
    py::array_t<float> data_opt,
    float L,
    float sharpness,
    float balance,
    float th_sar,
    int block_size,
    int win_size,
    int stride
) {
    int H = amplitude.shape(0);
    int W = amplitude.shape(1);

    // Convert numpy arrays to OpenCV
    cv::Mat_<PixelType> noisy = numpy_to_cvmat(amplitude);
    cv::Mat_<PixelGuidaType> guida = numpy_to_cvmat_guided(data_opt);

    // Dummy valClass (all ones, or use an argument if needed)
    cv::Mat_<bool> valClass(noisy.size(), true);

    // Configure GNLM parameters
    GuidedNLMeansProfile<PixelType> opt;
    // Map Python parameters to GNLM config
    opt.config(
        block_size,         // N1: block size
        win_size,           // N2: max number of blocks to consider
        win_size,           // Ns: search diameter
        stride,             // Nstep: step
        th_sar,             // tau_match
        sharpness,          // aggregationBeta
        balance,            // alpha
        0.0f,               // thDist (set to 0, or add argument)
        L,                  // lambda1
        0.0f                // lambda2 (set to 0, or add argument)
    );

    cv::Mat_<PixelType> denoised(noisy.size());
    cv::Mat_<PixelType> weights(noisy.size());

    // Call GNLM algorithm
    guided_nlmeans<PixelType, PixelGuidaType, DistanceSar_int_sum<PixelType>, DistanceAwgnVec<PixelType, GUIDA_NUM_BANDS>>(
        noisy, guida, valClass, denoised, weights, opt);

    // Prepare output numpy
    py::array_t<float> output({H, W});
    auto out_buf = output.mutable_unchecked<2>();
    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j)
            out_buf(i, j) = denoised(i, j);

    return output;
}

PYBIND11_MODULE(guidedNLMeans, m) {
    m.def("denoise", &denoise,
        py::arg("amplitude"),
        py::arg("data_opt"),
        py::arg("L"),
        py::arg("sharpness"),
        py::arg("balance"),
        py::arg("th_sar"),
        py::arg("block_size"),
        py::arg("win_size"),
        py::arg("stride"),
        "GNLM");
}

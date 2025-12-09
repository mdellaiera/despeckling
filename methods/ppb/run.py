import os
import numpy as np
import argparse
import logging
import matlab.engine
import sys
sys.path.insert(0, '../scripts')
from baselines_utils import read_image, save_image, prepare_output_directory, c2ap, ap2c

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

METHOD_NAME = "PPB"


def build_argparser():
    """
    Build the argument parser.
    """
    parser = argparse.ArgumentParser(
        description=f"Test {METHOD_NAME}."
    )
    parser.add_argument("--input_path", required=True, help="(Mandatory) Path to the .mat file containing the data.")
    parser.add_argument("--matlab_script_path", required=True, help="(Mandatory) Path to the .m script that implements the {METHOD_NAME} algorithm.")
    parser.add_argument("--output_path", required=False, default="./results/output.mat", help="(Optional) Path to save the denoised output. Default is './results/output.mat'.")
    parser.add_argument("--L", type=int, required=False, default=1, help="(Optional) ENL of the Nakagami-Rayleigh noise. Default is 1.")
    parser.add_argument("--hw", type=int, required=False, default=10, help="(Optional) Half sizes of the search window width. Default is 10.")
    parser.add_argument("--hd", type=int, required=False, default=3, help="(Optional) Half sizes of the  window width. Default is 3.")
    parser.add_argument("--alpha", type=float, required=False, default=0.92, help="(Optional) Alpha-quantile parameters on the noisy image. Default is 0.92.")
    parser.add_argument("--T", type=float, required=False, default=0.2, help="(Optional) Filtering parameters on the estimated image. Default is 0.2.")
    parser.add_argument("--nbits", type=int, required=False, default=4, help="(Optional) Numbers of iteration. Default is 4.")
    parser.add_argument("--estimate_path", required=False, default=None, help="(Optional) First noise-free image estimate path.")

    return parser


def process_image(data: np.ndarray, 
                  matlab_script_path: str, 
                  L: int, 
                  hw: int, 
                  hd: int, 
                  alpha: float, 
                  T: float, 
                  nbits: int, 
                  estimate_path: str = None) -> np.ndarray:
    assert estimate_path is None, "Not implemented error: estimate_path is not supported in this implementation."
    amplitude, phase = c2ap(data)
    eng = matlab.engine.start_matlab()
    eng.addpath(os.path.dirname(matlab_script_path), nargout=0)
    eng.eval("cd('{}')".format(os.path.dirname(matlab_script_path)), nargout=0)
    try:
        y = eng.ppb_nakagami(matlab.double(amplitude.tolist()), matlab.double(L), matlab.double(hw), matlab.double(hd), matlab.double(alpha), matlab.double(T), matlab.double(nbits), nargout=1)
        filtered = np.array(y).reshape(amplitude.shape)
        eng.quit()
    except Exception as e:
        eng.quit()
        raise RuntimeError(f"Error during execution: {e}")
    output = ap2c(filtered, phase)
    return output


def main():
    logger.info(f"Starting {METHOD_NAME} denoising process.")
    parser = build_argparser()
    args = parser.parse_args()

    prepare_output_directory(args.output_path)
    data = read_image(args.input_path, 'sar')
    output = process_image(data, args.matlab_script_path, args.L, args.hw, args.hd, args.alpha, args.T, args.nbits, args.estimate_path)
    save_image(args.output_path, output, 'sar_despeckled')
    logger.info(f"Completed. Output saved to {args.output_path}")


if __name__ == "__main__":
    main()

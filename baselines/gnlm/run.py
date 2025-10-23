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

METHOD_NAME = "GNLM"


def build_argparser():
    """
    Build the argument parser.
    """
    parser = argparse.ArgumentParser(
        description=f"Test {METHOD_NAME}."
    )
    parser.add_argument("--input_path_sar", required=True, help="(Mandatory) Path to the .mat file containing the SAR data.")
    parser.add_argument("--input_path_opt", required=True, help="(Mandatory) Path to the .mat file containing the optical data.")
    parser.add_argument("--matlab_script_path", required=True, help="(Mandatory) Path to the .m script that implements the {METHOD_NAME} algorithm.")
    parser.add_argument("--output_path", required=False, default="./results/output.mat", help="(Optional) Path to save the denoised output. Default is './results/output.mat'.")
    parser.add_argument("--L", type=float, required=True, help="(Mandatory) Number of looks of the noisy image.")
    parser.add_argument("--stack_size", type=int, default=256, help="(Optional) Maximum size of the 3rd dimension of the stack. Default is 256.")
    parser.add_argument("--sharpness", type=float, default=0.002, help="(Optional) Decay parameter. Default is 0.002.")
    parser.add_argument("--balance", type=float, default=0.15, help="(Optional) Balance parameter. Default is 0.15.")
    parser.add_argument("--th_sar", type=float, default=2.0, help="(Optional) Test threshold. Default is 2.0.")
    parser.add_argument("--block_size", type=int, default=8, help="(Optional) Number of rows/cols of the block. Default is 8.")
    parser.add_argument("--win_size", type=int, default=39, help="(Optional) Diameter of the search area. Default is 39.")
    parser.add_argument("--stride", type=int, default=3, help="(Optional) Dimension of step in sliding window processing. Default is 3.")

    return parser


def process_image(data_sar: np.ndarray, 
                  data_opt: np.ndarray, 
                  matlab_script_path: str, 
                  L: float, 
                  stack_size: int,
                  sharpness: float, 
                  balance: float, 
                  th_sar: float, 
                  block_size: int, 
                  win_size: int, 
                  stride: int) -> np.ndarray:
    os.environ['LD_LIBRARY_PATH'] = os.path.join(
        os.path.dirname(os.path.abspath(matlab_script_path)), "lib_opencv210/glnxa64"
    ) + ":" + os.environ.get('LD_LIBRARY_PATH', '')

    amplitude, phase = c2ap(data_sar)  
    eng = matlab.engine.start_matlab()
    eng.addpath(os.path.dirname(matlab_script_path), nargout=0)
    eng.eval("cd('{}')".format(os.path.dirname(matlab_script_path)), nargout=0)
    try:
        y = eng.guidedNLMeans(
            matlab.double(amplitude.tolist()), 
            matlab.double(L), 
            matlab.double(data_opt.tolist()),
            matlab.double(stack_size),
            matlab.double(sharpness), 
            matlab.double(balance), 
            matlab.double(th_sar), 
            matlab.double(block_size), 
            matlab.double(win_size), 
            matlab.double(stride), 
            nargout=2)
        filtered = np.array(y[0]).reshape(amplitude.shape)
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
    data_sar = read_image(args.input_path_sar)
    data_opt = read_image(args.input_path_opt)
    output = process_image(data_sar, 
                           data_opt, 
                           args.matlab_script_path,
                           args.L, 
                           args.stack_size,
                           args.sharpness, 
                           args.balance, 
                           args.th_sar, 
                           args.block_size, 
                           args.win_size, 
                           args.stride)
    save_image(args.output_path, output)
    logger.info(f"Completed. Output saved to {args.output_path}")


if __name__ == "__main__":
    main()

import os
import importlib
import numpy as np
import argparse
import logging
import matlab.engine
import sys
sys.path.insert(0, '../scripts')
from baselines_utils import read_image, save_image, prepare_output_directory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

METHOD_NAME = "FANS"


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

    return parser


def process_image(data: np.ndarray, 
                  matlab_script_path: str, 
                  L: int) -> np.ndarray:
    os.environ['LD_LIBRARY_PATH'] = os.path.join(
        os.path.dirname(os.path.abspath(matlab_script_path)), "lib_opencv210/glnxa64"
    ) + ":" + os.environ.get('LD_LIBRARY_PATH', '')

    input = np.log1p(np.abs(data))
    eng = matlab.engine.start_matlab()
    eng.addpath(os.path.dirname(matlab_script_path), nargout=0)
    eng.eval("cd('{}')".format(os.path.dirname(matlab_script_path)), nargout=0)
    try:
        y = eng.FANS(matlab.double(input.tolist()), matlab.double(L), nargout=2)
        output = np.array(y[0]).reshape(input.shape)
        eng.quit()
    except Exception as e:
        eng.quit()
        raise RuntimeError(f"Error during execution: {e}")
    return output


def main():
    logger.info(f"Starting {METHOD_NAME} denoising process.")
    parser = build_argparser()
    args = parser.parse_args()

    prepare_output_directory(args.output_path)
    data = read_image(args.input_path)
    output = process_image(data, args.matlab_script_path, args.L)
    save_image(args.output_path, output)
    logger.info(f"Completed. Output saved to {args.output_path}")


if __name__ == "__main__":
    main()

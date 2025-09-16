import numpy as np
import argparse
import logging
import bm3d
import sys
sys.path.insert(0, '../scripts')
from baselines_utils import read_image, save_image, prepare_output_directory, c2ap, ap2c

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

METHOD_NAME = "BM3D"


def build_argparser():
    """
    Build the argument parser.
    """
    parser = argparse.ArgumentParser(
        description=f"Test {METHOD_NAME}."
    )
    parser.add_argument("--input_path", required=True, help="(Mandatory) Path to the .mat file containing the data.")
    parser.add_argument("--output_path", required=False, default="./results/output.mat", help="(Optional) Path to save the denoised output. Default is './results/output.mat'.")
    parser.add_argument("--sigma_psd", type=float, required=True, help="(Mandatory) PSD noise level.")

    return parser


def process_image(data: np.ndarray, sigma_psd: float) -> np.ndarray:
    amplitude, phase = c2ap(data)
    filtered = bm3d.bm3d(amplitude, sigma_psd=sigma_psd, stage_arg=bm3d.BM3DStages.ALL_STAGES)
    output = ap2c(filtered, phase)
    return output


def main():
    logger.info(f"Starting {METHOD_NAME} denoising process.")
    parser = build_argparser()
    args = parser.parse_args()

    prepare_output_directory(args.output_path)
    data = read_image(args.input_path)
    output = process_image(data, args.sigma_psd)
    save_image(args.output_path, output)
    logger.info(f"Completed. Output saved to {args.output_path}")


if __name__ == "__main__":
    main()

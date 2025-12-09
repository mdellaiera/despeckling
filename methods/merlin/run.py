import os
import numpy as np
import argparse
import logging
import sys
sys.path.insert(0, '../scripts')
from baselines_utils import read_image, save_image, prepare_output_directory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

METHOD_NAME = "MERLIN"


def build_argparser():
    """
    Build the argument parser.
    """
    parser = argparse.ArgumentParser(
        description=f"Test {METHOD_NAME}."
    )
    parser.add_argument("--input_path", required=True, help="(Mandatory) Path to the .mat file containing the data.")
    parser.add_argument("--project_path", required=True, help="(Mandatory) Path to the project.")
    parser.add_argument("--output_path", required=False, default="./results/output.mat", help="(Optional) Path to save the denoised output. Default is './results/output.mat'.")

    return parser


def process_image(data: np.ndarray, project_path: str) -> np.ndarray:
    sys.path.insert(0, project_path)
    from deepdespeckling.utils.constants import PATCH_SIZE, STRIDE_SIZE
    from deepdespeckling.merlin.merlin_denoiser import MerlinDenoiser

    input = np.stack((np.real(data), np.imag(data)), axis=-1)
    denoiser = MerlinDenoiser(model_name='spotlight', symetrise=False)
    output = denoiser.denoise_image(input, patch_size=PATCH_SIZE, stride_size=STRIDE_SIZE)['denoised']['full']
    return output


def main():
    logger.info(f"Starting {METHOD_NAME} denoising process.")
    parser = build_argparser()
    args = parser.parse_args()

    prepare_output_directory(args.output_path)
    data = read_image(args.input_path, 'sar')
    output = process_image(data, args.project_path)
    save_image(args.output_path, output, 'sar_despeckled')
    logger.info(f"Completed. Output saved to {args.output_path}")


if __name__ == "__main__":
    main()

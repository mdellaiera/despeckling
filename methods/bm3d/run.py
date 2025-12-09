import os
import argparse
import logging
from scripts.io import read_image, save_image, prepare_output_directory, KEY_INPUT_SAR, KEY_OUTPUT_SAR
from methods.bm3d.bm3d import BM3D

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
    parser.add_argument("--input_path", required=True, help="(Mandatory) Path to the file containing the data.")
    parser.add_argument("--output_path", required=False, default=os.path.join(os.path.dirname(__file__), "results/output.mat"), help="(Optional) Path to save the denoised output. Default is './results/output.mat'.")
    parser.add_argument("--sigma_psd", type=float, required=True, help="(Mandatory) PSD noise level.")

    return parser


def main():
    logger.info(f"Starting {METHOD_NAME} denoising process.")
    parser = build_argparser()
    args = parser.parse_args()
    
    Filter = BM3D()
    input_sar = read_image(args.input_path, key=KEY_INPUT_SAR)
    output_sar = Filter.filter(input_sar, args.sigma_psd)

    prepare_output_directory(args.output_path)
    save_image(args.output_path, output_sar, key=KEY_OUTPUT_SAR)
    logger.info(f"Completed. Output saved to {args.output_path}")


if __name__ == "__main__":
    main()

import os
import argparse
import logging
from scripts.io import read_image, save_image, prepare_output_directory, KEY_INPUT_SAR, KEY_OUTPUT_SAR
from methods.fans.fans import FANS

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
    parser.add_argument("--input_path", required=True, help="(Mandatory) Path to the file containing the data.")
    parser.add_argument("--matlab_script_path", required=True, help="(Mandatory) Path to the .m script that implements the {METHOD_NAME} algorithm.")
    parser.add_argument("--output_path", required=False, default=os.path.join(os.path.dirname(__file__), "results/output.mat"), help="(Optional) Path to save the denoised output. Default is './results/output.mat'.")
    parser.add_argument("--L", type=int, required=False, default=1, help="(Optional) ENL of the Nakagami-Rayleigh noise. Default is 1.")

    return parser


def main():
    logger.info(f"Starting {METHOD_NAME} denoising process.")
    parser = build_argparser()
    args = parser.parse_args()

    Filter = FANS(args.matlab_script_path)
    input_sar = read_image(args.input_path, key=KEY_INPUT_SAR)
    output_sar = Filter.filter(input_sar, args.L)

    prepare_output_directory(args.output_path)
    save_image(args.output_path, output_sar, KEY_OUTPUT_SAR)
    logger.info(f"Completed. Output saved to {args.output_path}")


if __name__ == "__main__":
    main()

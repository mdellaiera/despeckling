import os
import argparse
import logging
from scripts.io import read_image, save_image, prepare_output_directory, KEY_INPUT_SAR, KEY_INPUT_EO, KEY_OUTPUT_SAR
from methods.gnlm.gnlm import GNLM

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
    parser.add_argument("--input_path", required=True, help="(Mandatory) Path to the file containing the data.")
    parser.add_argument("--matlab_script_path", required=True, help="(Mandatory) Path to the .m script that implements the {METHOD_NAME} algorithm.")
    parser.add_argument("--output_path", required=False, default=os.path.join(os.path.dirname(__file__), "results/output.mat"), help="(Optional) Path to save the denoised output. Default is './results/output.mat'.")
    parser.add_argument("--L", type=int, required=True, help="(Mandatory) Number of looks of the noisy image.")
    parser.add_argument("--stack_size", type=int, default=256, help="(Optional) Maximum size of the 3rd dimension of the stack. Default is 256.")
    parser.add_argument("--sharpness", type=float, default=0.002, help="(Optional) Decay parameter. Default is 0.002.")
    parser.add_argument("--balance", type=float, default=0.15, help="(Optional) Balance parameter. Default is 0.15.")
    parser.add_argument("--th_sar", type=float, default=2.0, help="(Optional) Test threshold. Default is 2.0.")
    parser.add_argument("--block_size", type=int, default=8, help="(Optional) Number of rows/cols of the block. Default is 8.")
    parser.add_argument("--win_size", type=int, default=39, help="(Optional) Diameter of the search area. Default is 39.")
    parser.add_argument("--stride", type=int, default=3, help="(Optional) Dimension of step in sliding window processing. Default is 3.")

    return parser


def main():
    logger.info(f"Starting {METHOD_NAME} denoising process.")
    parser = build_argparser()
    args = parser.parse_args()

    Filter = GNLM(args.matlab_script_path)
    input_sar = read_image(args.input_path, key=KEY_INPUT_SAR)
    input_eo = read_image(args.input_path, key=KEY_INPUT_EO)
    output_sar = Filter.filter(
        input_sar, 
        input_eo, 
        args.L, 
        args.stack_size, 
        args.sharpness, 
        args.balance, 
        args.th_sar, 
        args.block_size, 
        args.win_size, 
        args.stride)

    prepare_output_directory(args.output_path)
    save_image(args.output_path, output_sar, KEY_OUTPUT_SAR)
    logger.info(f"Completed. Output saved to {args.output_path}")


if __name__ == "__main__":
    main()

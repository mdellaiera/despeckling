import os
import argparse
import logging
from scripts.io import read_image, save_image, prepare_output_directory, KEY_INPUT_SAR, KEY_OUTPUT_SAR
from methods.ppb.ppb import PPB

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
    parser.add_argument("--output_path", required=False, default=os.path.join(os.path.dirname(__file__), "results/output.mat"), help="(Optional) Path to save the denoised output. Default is './results/output.mat'.")
    parser.add_argument("--L", type=int, required=False, default=1, help="(Optional) ENL of the Nakagami-Rayleigh noise. Default is 1.")
    parser.add_argument("--hw", type=int, required=False, default=10, help="(Optional) Half sizes of the search window width. Default is 10.")
    parser.add_argument("--hd", type=int, required=False, default=3, help="(Optional) Half sizes of the  window width. Default is 3.")
    parser.add_argument("--alpha", type=float, required=False, default=0.92, help="(Optional) Alpha-quantile parameters on the noisy image. Default is 0.92.")
    parser.add_argument("--T", type=float, required=False, default=0.2, help="(Optional) Filtering parameters on the estimated image. Default is 0.2.")
    parser.add_argument("--nbits", type=int, required=False, default=4, help="(Optional) Numbers of iteration. Default is 4.")
    parser.add_argument("--estimate_path", required=False, default=None, help="(Optional) First noise-free image estimate path.")

    return parser


def main():
    logger.info(f"Starting {METHOD_NAME} denoising process.")
    parser = build_argparser()
    args = parser.parse_args()

    Filter = PPB(args.matlab_script_path)
    input_sar = read_image(args.input_path, key=KEY_INPUT_SAR)
    output_sar = Filter.filter(input_sar, args.L, args.hw, args.hd, args.alpha, args.T, args.nbits)

    prepare_output_directory(args.output_path)
    save_image(args.output_path, output_sar, KEY_OUTPUT_SAR)
    logger.info(f"Completed. Output saved to {args.output_path}")


if __name__ == "__main__":
    main()

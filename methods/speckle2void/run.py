import os
import argparse
import logging
from scripts.io import read_image, save_image, prepare_output_directory, KEY_INPUT_SAR, KEY_OUTPUT_SAR
from methods.speckle2void.speckle2void import Speckle2Void

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

METHOD_NAME = "Speckle2Void"


def build_argparser():
    """
    Build the argument parser.
    """
    parser = argparse.ArgumentParser(
        description=f"Test {METHOD_NAME}."
    )
    parser.add_argument("--input_path", required=True, help="(Mandatory) Path to the .mat file containing the data.")
    parser.add_argument("--checkpoint_path", required=True, help="(Mandatory) Path to the weights of the model.")
    parser.add_argument("--libraries_path", required=True, help="(Mandatory) Path to the libraries of the model.")
    parser.add_argument("--norm", required=True, type=float, help="(Mandatory) Normalization factor for the model input.")
    parser.add_argument("--output_path", required=False, default=os.path.join(os.path.dirname(__file__), "results/output.mat"), help="(Optional) Path to save the denoised output. Default is './results/output.mat'.")

    return parser


def main():
    logger.info(f"Starting {METHOD_NAME} denoising process.")
    parser = build_argparser()
    args = parser.parse_args()

    Filter = Speckle2Void(args.checkpoint_path, args.libraries_path)
    input_sar = read_image(args.input_path, key=KEY_INPUT_SAR)
    output_sar = Filter.filter(input_sar, args.norm)

    prepare_output_directory(args.output_path)
    save_image(args.output_path, output_sar, KEY_OUTPUT_SAR)
    logger.info(f"Completed. Output saved to {args.output_path}")


if __name__ == "__main__":
    main()
    
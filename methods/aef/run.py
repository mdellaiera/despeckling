import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
import argparse
import logging
from scripts.io import read_image, save_image, prepare_output_directory, KEY_INPUT_SAR, KEY_INPUT_EMBEDDINGS, KEY_OUTPUT_SAR
from methods.aef.aef import AEF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

METHOD_NAME = "AEF"


def build_argparser():
    """
    Build the argument parser.
    """
    parser = argparse.ArgumentParser(
        description=f"Test {METHOD_NAME}."
    )
    parser.add_argument("--input_path", required=True, help="(Mandatory) Path to the file containing the data.")
    parser.add_argument("--output_path", required=False, default=os.path.join(os.path.dirname(__file__), "results/output.mat"), help="(Optional) Path to save the denoised output. Default is './results/output.mat'.")
    parser.add_argument("--sigma_distance", type=float, required=False, default=0.1, help="(Optional) Sigma distance for the Gaussian kernel. Default is 0.1.")
    parser.add_argument("--radius_despeckling", type=int, required=False, default=30, help="(Optional) Radius for despeckling. Default is 30.")
    parser.add_argument("--n_blocks", type=int, required=False, default=10, help="(Optional) Number of blocks for processing. Default is 10.")

    return parser


def main():
    logger.info(f"Starting {METHOD_NAME} denoising process.")
    parser = build_argparser()
    args = parser.parse_args()

    Filter = AEF()
    input_sar = read_image(args.input_path, key=KEY_INPUT_SAR)
    input_embeddings = read_image(args.input_path, key=KEY_INPUT_EMBEDDINGS)
    output_sar = Filter.filter(
        input_sar, 
        input_embeddings,
        sigma_distance=args.sigma_distance,
        radius_despeckling=args.radius_despeckling,
        n_blocks=args.n_blocks
    )

    prepare_output_directory(args.output_path)
    save_image(args.output_path, output_sar, KEY_OUTPUT_SAR)
    logger.info(f"Completed. Output saved to {args.output_path}")


if __name__ == "__main__":
    main()

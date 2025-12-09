import os
# os.environ['JAX_PLATFORM_NAME'] = 'cpu'
import argparse
import logging
from scripts.io import read_image, save_image, prepare_output_directory, KEY_INPUT_SAR, KEY_INPUT_EO, KEY_OUTPUT_SAR
from methods.tbog.tbog import TBOG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

METHOD_NAME = "TBOG"


def build_argparser():
    """
    Build the argument parser.
    """
    parser = argparse.ArgumentParser(
        description=f"Test {METHOD_NAME}."
    )
    parser.add_argument("--input_path", required=True, help="(Mandatory) Path to the file containing the data.")
    parser.add_argument("--output_path", required=False, default="./results/output.mat", help="(Optional) Path to save the denoised output. Default is './results/output.mat'.")
    parser.add_argument("--radius_descriptor", type=int, required=False, default=7, help="(Optional) Radius for the texture descriptor. Default is 7.")
    parser.add_argument("--sigma_spatial", type=float, required=False, default=5, help="(Optional) Spatial standard deviation for Gaussian kernel. Default is 5.")
    parser.add_argument("--sigma_luminance_eo", type=float, required=False, default=0.1, help="(Optional) Standard deviation for the optical data. Default is 0.1.")
    parser.add_argument("--sigma_luminance_sar", type=float, required=False, default=0.1, help="(Optional) Standard deviation for the SAR data. Default is 0.1.")
    parser.add_argument("--gamma_luminance_eo", type=float, required=False, default=1., help="(Optional) Gamma value for the optical data. Default is 1.")
    parser.add_argument("--gamma_luminance_sar", type=float, required=False, default=1., help="(Optional) Gamma value for the SAR data. Default is 1.")
    parser.add_argument("--alpha", type=float, required=False, default=1, help="(Optional) Scaling factor for the update step. Default is 1.")
    parser.add_argument("--n_iterations", type=int, required=False, default=100, help="(Optional) Number of iterations for the filtering process. Default is 100.")
    parser.add_argument("--sigma_distance", type=float, required=False, default=1.5, help="(Optional) Standard deviation for the Gaussian kernel used in similarity computation. Default is 1.5.")
    parser.add_argument("--radius_despeckling", type=int, required=False, default=30, help="(Optional) Radius to consider neighboring pixels. Default is 30.")
    parser.add_argument("--n_blocks", type=int, required=False, default=10, help="(Optional) Number of blocks for processing the image in parallel. Default is 10.")

    return parser


def main():
    logger.info(f"Starting {METHOD_NAME} denoising process.")
    parser = build_argparser()
    args = parser.parse_args()

    Filter = TBOG()
    input_sar = read_image(args.input_path, key=KEY_INPUT_SAR)
    input_eo = read_image(args.input_path, key=KEY_INPUT_EO)
    output_sar = Filter.filter(
        sar=input_sar, 
        eo=input_eo, 
        radius_descriptor=args.radius_descriptor, 
        sigma_spatial=args.sigma_spatial, 
        sigma_guides=[args.sigma_luminance_eo, args.sigma_luminance_sar], 
        gamma_guides=[args.gamma_luminance_eo, args.gamma_luminance_sar], 
        alpha=args.alpha, 
        n_iterations=args.n_iterations, 
        n_blocks_mubf=args.n_blocks,
        sigma_distance=args.sigma_distance,
        radius_despeckling=args.radius_despeckling, 
        n_blocks_despeckling=args.n_blocks
    )
    
    prepare_output_directory(args.output_path)
    save_image(args.output_path, output_sar, KEY_OUTPUT_SAR)
    logger.info(f"Completed. Output saved to {args.output_path}")


if __name__ == "__main__":
    main()

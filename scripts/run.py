import os
# os.environ['JAX_PLATFORM_NAME'] = 'cpu'
import numpy as np
import jax.numpy as jnp
import argparse
import logging
from typing import List, Tuple
from despeckle import SARDespeckling
from utils import read_image, save_image, prepare_output_directory

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
    parser.add_argument("--input_path_sar", required=True, help="(Mandatory) Path to the .mat file containing the SAR data.")
    parser.add_argument("--input_path_opt", required=True, help="(Mandatory) Path to the .mat file containing the optical data.")
    parser.add_argument("--output_path", required=False, default="./results/output.mat", help="(Optional) Path to save the denoised output. Default is './results/output.mat'.")
    parser.add_argument("--radius_descriptor", type=int, required=False, default=7, help="(Optional) Radius for the texture descriptor. Default is 7.")
    parser.add_argument("--sigma_spatial", type=float, required=False, default=5, help="(Optional) Spatial standard deviation for Gaussian kernel. Default is 5.")
    parser.add_argument("--sigma_luminance_opt", type=float, required=False, default=0.1, help="(Optional) Standard deviation for the optical data. Default is 0.1.")
    parser.add_argument("--sigma_luminance_sar", type=float, required=False, default=0.1, help="(Optional) Standard deviation for the SAR data. Default is 0.1.")
    parser.add_argument("--alpha", type=float, required=False, default=1, help="(Optional) Scaling factor for the update step. Default is 1.")
    parser.add_argument("--n_iterations", type=int, required=False, default=100, help="(Optional) Number of iterations for the filtering process. Default is 100.")
    parser.add_argument("--sigma_distance", type=float, required=False, default=1.5, help="(Optional) Standard deviation for the Gaussian kernel used in similarity computation. Default is 1.5.")
    parser.add_argument("--radius_despeckling", type=int, required=False, default=30, help="(Optional) Radius to consider neighboring pixels. Default is 30.")
    parser.add_argument("--n_blocks", type=int, required=False, default=10, help="(Optional) Number of blocks for processing the image in parallel. Default is 10.")

    return parser


def process_image(data_sar: np.ndarray, 
                  data_opt: np.ndarray, 
                  radius_descriptor: int,
                  sigma_spatial: float,
                  sigma_luminance_opt: float,
                  sigma_luminance_sar: float,
                  gamma_opt: float,
                  gamma_sar: float,
                  alpha: float, 
                  n_iterations: int, 
                  sigma_distance: float,
                  radius_despeckling: int,
                  n_blocks: int) -> Tuple[np.ndarray, List[float]]:
    input_opt = jnp.array(data_opt, dtype=jnp.float32)
    input_sar = jnp.array(data_sar, dtype=jnp.float32)

    despeckling = SARDespeckling()
    output = despeckling.run(
        sar=input_sar,
        opt=input_opt,
        radius_descriptor=radius_descriptor,
        sigma_spatial=sigma_spatial,
        sigma_guides=[sigma_luminance_opt, sigma_luminance_sar],
        gamma_guides=[gamma_opt, gamma_sar],
        alpha=alpha,
        n_iterations=n_iterations,
        n_blocks_mubf=n_blocks,
        sigma_distance=sigma_distance,
        radius_despeckling=radius_despeckling,
        n_blocks_despeckling=n_blocks
    )

    return output


def main():
    logger.info(f"Starting {METHOD_NAME} denoising process.")
    parser = build_argparser()
    args = parser.parse_args()

    prepare_output_directory(args.output_path)
    data_sar = read_image(args.input_path_sar, 'sar')
    data_opt = read_image(args.input_path_opt, 'eo')
    output = process_image(data_sar, 
                           data_opt, 
                           args.radius_descriptor, 
                           args.sigma_spatial, 
                           args.sigma_luminance_opt, 
                           args.sigma_luminance_sar, 
                           args.alpha, 
                           args.n_iterations, 
                           args.sigma_distance,
                           args.radius_despeckling, 
                           args.n_blocks)
    save_image(args.output_path, output, 'sar_despeckled')
    logger.info(f"Completed. Output saved to {args.output_path}")


if __name__ == "__main__":
    main()

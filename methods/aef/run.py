import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
import numpy as np
import jax.numpy as jnp
import argparse
import logging
from typing import List
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../scripts')
from baselines_utils import read_image, save_image, prepare_output_directory
from methods.aef.aef import SARDespeckling

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
    parser.add_argument("--input_path_sar", required=True, help="(Mandatory) Path to the .mat file containing the SAR data.")
    parser.add_argument("--input_path_embeddings", required=True, help="(Mandatory) Path to the .mat file containing the embeddings data.")
    parser.add_argument("--output_path", required=False, default="./results/output.mat", help="(Optional) Path to save the denoised output. Default is './results/output.mat'.")
    parser.add_argument("--sigma_distance", type=float, required=False, default=0.1, help="(Optional) Sigma distance for the Gaussian kernel. Default is 0.1.")
    parser.add_argument("--radius_despeckling", type=int, required=False, default=30, help="(Optional) Radius for despeckling. Default is 30.")
    parser.add_argument("--n_blocks", type=int, required=False, default=10, help="(Optional) Number of blocks for processing. Default is 10.")

    return parser


def process_image(data_sar: np.ndarray, 
                  data_embeddings: np.ndarray, 
                  sigma_distance: float,
                  radius_despeckling: int,
                  n_blocks: int) -> np.ndarray:
    input_embeddings = jnp.array(data_embeddings, dtype=jnp.float32)
    input_sar = jnp.array(data_sar, dtype=jnp.float32)

    despeckling = SARDespeckling()
    output = despeckling.filter(
        sar=input_sar,
        embeddings=input_embeddings,
        sigma_distance=sigma_distance,
        radius_despeckling=radius_despeckling,
        n_blocks=n_blocks
    )

    return output


def main():
    logger.info(f"Starting {METHOD_NAME} denoising process.")
    parser = build_argparser()
    args = parser.parse_args()

    prepare_output_directory(args.output_path)
    data_sar = read_image(args.input_path_sar, 'sar')
    data_embeddings = read_image(args.input_path_embeddings, 'embeddings')
    output = process_image(data_sar, 
                           data_embeddings=data_embeddings,
                           sigma_distance=args.sigma_distance,
                           radius_despeckling=args.radius_despeckling,
                           n_blocks=args.n_blocks)
    save_image(args.output_path, output, 'sar_despeckled')
    logger.info(f"Completed. Output saved to {args.output_path}")


if __name__ == "__main__":
    main()

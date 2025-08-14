import numpy as np
import jax.numpy as jnp
import argparse
import logging
from typing import List
import sys
sys.path.insert(0, '../scripts')
from baselines_utils import read_image, save_image, prepare_output_directory
from odnsd_utils import SARDespeckling

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

METHOD_NAME = "ODNSD"


def build_argparser():
    """
    Build the argument parser.
    """
    parser = argparse.ArgumentParser(
        description=f"Test {METHOD_NAME}."
    )
    parser.add_argument("--input_path_sar", required=True, help="(Mandatory) Path to the .mat file containing the SAR data.")
    parser.add_argument("--input_path_opt", required=True, help="(Mandatory) Path to the .mat file containing the optical data.")
    parser.add_argument("--matlab_script_path", required=True, help="(Mandatory) Path to the .m script that implements the {METHOD_NAME} algorithm.")
    parser.add_argument("--output_path", required=False, default="./results/output.mat", help="(Optional) Path to save the denoised output. Default is './results/output.mat'.")
    parser.add_argument("--L", type=float, required=False, default=1, help="(Optional) Number of looks. Default is 1.")
    parser.add_argument("--window_size", type=int, required=False, default=31, help="(Optional) Window size. Default is 31.")
    parser.add_argument("--lambda_S", type=float, required=False, default=0.005, help="(Optional) Weight for spatial Gaussian kernel. Default is 0.005.")
    parser.add_argument("--lambda_RO", type=float, required=False, default=0.02, help="(Optional) Weight for optical luminance kernel. Default is 0.02.")
    parser.add_argument("--lambda_RS", type=float, required=False, default=0.1, help="(Optional) Weight for SAR luminance kernel. Default is 0.1.")
    parser.add_argument("--N", type=list, required=False, default=np.arange(7, 63, 2).tolist(), help="(Optional) List of number of looks. Default is [7, 9, ..., 61].")
    parser.add_argument("--gamma", type=float, required=False, default=7, help="(Optional) Gamma parameter for the soft classifier. Default is 7.")
    parser.add_argument("--a0", type=float, required=False, default=0.64, help="(Optional) a0 parameter for the soft classifier. Default is 0.64.")

    return parser


def process_image(data_sar: np.ndarray, 
                  data_opt: np.ndarray, 
                  matlab_script_path: str,
                  L: int = 1,
                  window_size: int = 31, 
                  lambda_S: float = 0.005, 
                  lambda_RO: float = 0.02, 
                  lambda_RS: float = 0.1,
                  N: List[int] = np.arange(7, 63, 2).tolist(),
                  gamma: float = 7,
                  a0: float = 0.64) -> np.ndarray:
    input_opt = jnp.array(data_opt, dtype=jnp.float32)
    input_sar = jnp.array(data_sar, dtype=jnp.float32)

    despeckling = SARDespeckling()
    output = despeckling.filter(
        sar=input_sar,
        opt=input_opt,
        matlab_script_path=matlab_script_path,
        L=L,
        window_size=window_size,
        lambda_S=lambda_S,
        lambda_RO=lambda_RO,
        lambda_RS=lambda_RS,
        N=N,
        gamma=gamma,
        a0=a0
    )

    return output


def main():
    logger.info(f"Starting {METHOD_NAME} denoising process.")
    parser = build_argparser()
    args = parser.parse_args()

    prepare_output_directory(args.output_path)
    data_sar = read_image(args.input_path_sar)
    data_opt = read_image(args.input_path_opt)
    output = process_image(data_sar, 
                           data_opt, 
                           args.matlab_script_path, 
                           args.L, 
                           args.window_size, 
                           args.lambda_S, 
                           args.lambda_RO, 
                           args.lambda_RS, 
                           args.N, 
                           args.gamma, 
                           args.a0)
    save_image(args.output_path, output)
    logger.info(f"Completed. Output saved to {args.output_path}")


if __name__ == "__main__":
    main()

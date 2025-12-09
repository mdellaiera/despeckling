import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
import argparse
import logging
from scripts.io import read_image, save_image, prepare_output_directory, KEY_INPUT_SAR, KEY_INPUT_EO, KEY_OUTPUT_SAR
from methods.gbf.gbf import GBF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

METHOD_NAME = "GBF"


def build_argparser():
    """
    Build the argument parser.
    """
    parser = argparse.ArgumentParser(
        description=f"Test {METHOD_NAME}."
    )
    parser.add_argument("--input_path_sar", required=True, help="(Mandatory) Path to the file containing the data.")
    parser.add_argument("--matlab_script_path", required=True, help="(Mandatory) Path to the .m script that implements the {METHOD_NAME} algorithm.")
    parser.add_argument("--output_path", required=False, default=os.path.join(os.path.dirname(__file__), "results/output.mat"), help="(Optional) Path to save the denoised output. Default is './results/output.mat'.")
    parser.add_argument("--L", type=int, required=False, default=1, help="(Optional) Number of looks. Default is 1.")
    parser.add_argument("--window_size", type=int, required=False, default=31, help="(Optional) Window size. Default is 31.")
    parser.add_argument("--lambda_S", type=float, required=False, default=0.005, help="(Optional) Weight for spatial Gaussian kernel. Default is 0.005.")
    parser.add_argument("--lambda_RO", type=float, required=False, default=0.02, help="(Optional) Weight for optical luminance kernel. Default is 0.02.")
    parser.add_argument("--lambda_RS", type=float, required=False, default=0.1, help="(Optional) Weight for SAR luminance kernel. Default is 0.1.")
    parser.add_argument("--N", type=list, required=False, default=np.arange(7, 63, 2).tolist(), help="(Optional) List of number of looks. Default is [7, 9, ..., 61].")
    parser.add_argument("--gamma", type=float, required=False, default=7, help="(Optional) Gamma parameter for the soft classifier. Default is 7.")
    parser.add_argument("--a0", type=float, required=False, default=0.64, help="(Optional) a0 parameter for the soft classifier. Default is 0.64.")

    return parser


def main():
    logger.info(f"Starting {METHOD_NAME} denoising process.")
    parser = build_argparser()
    args = parser.parse_args()

    Filter = GBF(args.matlab_script_path)
    input_sar = read_image(args.input_path, key=KEY_INPUT_SAR)
    input_eo = read_image(args.input_path, key=KEY_INPUT_EO)
    output_sar = Filter.filter(
        input_sar, 
        input_eo, 
        args.L,
        args.window_size,
        args.lambda_S,
        args.lambda_RO,
        args.lambda_RS,
        args.N,
        args.gamma,
        args.a0
    )

    prepare_output_directory(args.output_path)
    save_image(args.output_path, output_sar, KEY_OUTPUT_SAR)
    logger.info(f"Completed. Output saved to {args.output_path}")


if __name__ == "__main__":
    main()

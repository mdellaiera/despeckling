import os
import numpy as np
import scipy
import argparse
import logging
import bm3d


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_argparser():
    """
    Build the argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Test BM3D."
    )
    parser.add_argument("--input_path", required=True, help="(Mandatory) Path to the .mat file containing the data.")
    parser.add_argument("--output_path", required=False, default="./results/output.mat", help="(Optional) Path to save the denoised output. Default is './results/output.mat'.")
    parser.add_argument("--sigma_psd", type=float, required=True, help="(Mandatory) PSD noise level.")

    return parser


def check_input_data(data):
    """Check if the input data is in the expected format."""
    if 'cout' not in data:
        raise ValueError("Input data must contain 'cout' key.")
    if not isinstance(data['cout'], np.ndarray):
        raise ValueError("'cout' must be a numpy array.")


def main():
    logger.info("Starting BM3D denoising process.")
    parser = build_argparser()
    args = parser.parse_args()

    # Prepare output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Load the file
    data = scipy.io.loadmat(args.input_path)

    I = data['cout']
    I_abs = np.log10(np.abs(I**2)+1)
    I_real = np.real(I)
    I_imag = np.imag(I)
    image = np.stack((I_real, I_imag), axis=-1)
    denoised_image = bm3d.bm3d(I_abs, sigma_psd=args.sigma_psd, stage_arg=bm3d.BM3DStages.ALL_STAGES)
    results = denoised_image

    scipy.io.savemat(args.output_path, {'cout': results})


if __name__ == "__main__":
    main()

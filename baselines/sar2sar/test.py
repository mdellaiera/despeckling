import os
import torch
import numpy as np
import scipy
import argparse
import logging
from deepdespeckling.utils.constants import PATCH_SIZE, STRIDE_SIZE
from deepdespeckling.sar2sar.sar2sar_denoiser import Sar2SarDenoiser
from deepdespeckling.model import Model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_argparser():
    """
    Build the argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Test SAR2SAR."
    )
    parser.add_argument("--input_path", required=True, help="(Mandatory) Path to the .mat file containing the data.")
    parser.add_argument("--output_path", required=False, default="./results/output.mat", help="(Optional) Path to save the denoised output. Default is './results/output.mat'.")

    return parser


class Sar2SarDenoiserFixed(Sar2SarDenoiser):
    """Class to share parameters beyond denoising functions
    """

    def __init__(self, **params):
        super().__init__(**params)

    def load_model(self, patch_size: int) -> Model:
        """Load model with given weights 

        Args:
            weights_path (str): path to weights  
            patch_size (int): patch size

        Returns:
            model (Model): model loaded with stored weights
        """
        model = Model(torch.device(self.device),
                      height=patch_size, width=patch_size)
        model.load_state_dict(torch.load(
            self.weights_path, map_location=torch.device("cpu"), weights_only=False)['model_state_dict'])

        return model


def main():
    logger.info("Starting SAR2SAR denoising process.")
    parser = build_argparser()
    args = parser.parse_args()

    # Prepare output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Load the file
    data = scipy.io.loadmat(args.input_path)

    I = data['cout']
    I_real = np.real(I)
    I_imag = np.imag(I)
    image = np.sqrt(I_real**2 + I_imag**2)
    denoiser = Sar2SarDenoiserFixed()
    denoised_image = denoiser.denoise_image(image, patch_size=PATCH_SIZE, stride_size=STRIDE_SIZE)
    results = denoised_image['denoised']

    scipy.io.savemat(args.output_path, {'cout': results})


if __name__ == "__main__":
    main()

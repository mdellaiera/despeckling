import os
import numpy as np
import argparse
import logging
import tensorflow as tf
import sys
sys.path.insert(0, '../scripts')
from baselines_utils import read_image, save_image, prepare_output_directory

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
    parser.add_argument("--output_path", required=False, default="./results/output.mat", help="(Optional) Path to save the denoised output. Default is './results/output.mat'.")

    return parser


def process_image(data: np.ndarray,
                  checkpoint_path: str,
                  libraries_path: str,
                  norm: float) -> np.ndarray:
    sys.path.insert(0, os.path.dirname(libraries_path))
    sys.path.insert(0, libraries_path)
    from Speckle2Void import Speckle2V

    tf.reset_default_graph()

    input = np.abs(data).reshape((1, data.shape[0], data.shape[1], 1))

    model = Speckle2V("./",
                    "./",
                    checkpoint_path,
                    batch_size=1,
                    patch_size=64,
                    model_name='speckle2void',
                    lr=1e-04, 
                    steps_per_epoch=2000,
                    k_penalty_tv=5e-05,
                    shift_list=[3,1],
                    prob = [0.9,0.1],
                    clip=500000,
                    norm=norm,
                    L_noise=1)    
    
    model.build_inference()
    model.load_weights()

    output = model.predict(input)
    return output.squeeze()


def main():
    logger.info(f"Starting {METHOD_NAME} denoising process.")
    parser = build_argparser()
    args = parser.parse_args()

    prepare_output_directory(args.output_path)
    data = read_image(args.input_path, 'sar')
    output = process_image(data, args.checkpoint_path, args.libraries_path, args.norm)
    save_image(args.output_path, output, 'sar_despeckled')
    logger.info(f"Completed. Output saved to {args.output_path}")


if __name__ == "__main__":
    main()
    
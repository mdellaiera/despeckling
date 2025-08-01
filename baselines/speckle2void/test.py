import os
import sys
import numpy as np
import scipy
import argparse
import logging
from DataGenerator import DataGenerator
import tensorflow as tf
from Speckle2Void import Speckle2V

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
sys.path.insert(0, './libraries')  #TODO: find the good path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_argparser():
    """
    Build the argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Test Speckle2Void."
    )
    parser.add_argument("--input_path", required=True, help="(Mandatory) Path to the .mat file containing the data.")
    parser.add_argument("--output_path", required=False, default="./results/output.mat", help="(Optional) Path to save the denoised output. Default is './results/output.mat'.")

    return parser


def check_input_data(data):
    """Check if the input data is in the expected format."""
    if 'cout' not in data:
        raise ValueError("Input data must contain 'cout' key.")
    if not isinstance(data['cout'], np.ndarray):
        raise ValueError("'cout' must be a numpy array.")


def main():
    logger.info("Starting Speckle2Void denoising process.")
    parser = build_argparser()
    args = parser.parse_args()

    # Prepare output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Load the file
    data = scipy.io.loadmat(args.input_path)

    I = data['cout']

    tf.reset_default_graph()
    batch_size=16

    dir_train = "../DataSet_SAR/TerraSAR_X/HDF5_SLC_DECORRELATED_2/"
    dir_test = "../DataSet_SAR/TerraSAR_X/HDF5_SLC_TEST_DECORRELATED_2/"

    file_checkpoint = 's2v_checkpoint/model.ckpt-299999'#None for the latest checkpoint

    model = Speckle2V(dir_train,
                    dir_test,
                    file_checkpoint,
                    batch_size=batch_size,
                    patch_size=64,
                    model_name='speckle2void',
                    lr=1e-04, 
                    steps_per_epoch=2000,
                    k_penalty_tv=5e-05,
                    shift_list=[3,1],
                    prob = [0.9,0.1],
                    clip=500000,
                    norm=100000,
                    L_noise=1)    
    
    model.build_inference()
    model.load_weights()
    datagen = DataGenerator()
    imgs = datagen.load_imgs_from_directory(directory = dir_test,filter='decorr*.mat',max_files=None)
    imgs = [img[:,0:1000,0:1000,:] for img in imgs]

    results = model.predict(imgs[0])

    # Save the cleaned image
    scipy.io.savemat(args.output_path, {'cout': results})


if __name__ == "__main__":
    main()
    
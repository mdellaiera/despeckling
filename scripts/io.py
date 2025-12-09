import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scripts.utils import c2ap

SUPPORTED_FORMATS = ('.npy', '.npz', '.mat')
KEY_INPUT_EO = 'eo'
KEY_INPUT_SAR = 'sar'
KEY_OUTPUT_SAR = 'sar_despeckled'


def read_image(input_path: str, key: str) -> np.ndarray:
    """Read an image from a file."""
    if input_path.endswith('.npy') or input_path.endswith('.npz'):
        data = np.load(input_path)
    elif input_path.endswith('.mat'):
        data = scipy.io.loadmat(input_path)
    else:
        raise ValueError(f"Unsupported file format. Use {SUPPORTED_FORMATS} files.")
    if key not in data:
        raise ValueError(f"Input file must contain '{key}' key.")
    return data[key]


def prepare_output_directory(output_path: str):
    """Prepare the output directory."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)


def save_image(output_path: str, image: np.ndarray, key: str):
    """Save an image to a file."""
    extension = os.path.splitext(output_path)[1]
    if extension not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported file format. Use {SUPPORTED_FORMATS} files.")
    plt.imsave(output_path.replace(extension, '.png'), c2ap(image)[0].squeeze(), cmap='gray')  # Preview
    if extension in ('.npy', '.npz'):
        np.savez_compressed(output_path, **{key: image})
    elif extension == '.mat':
        scipy.io.savemat(output_path, {key: image})

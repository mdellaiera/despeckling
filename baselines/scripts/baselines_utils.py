import os
import numpy as np
import scipy.io

KEY = 'data'


def read_image(input_path: str) -> np.ndarray:
    """Read an image from a file."""
    data = scipy.io.loadmat(input_path)
    if KEY not in data:
        raise ValueError(f"Input file must contain '{KEY}' key.")
    return data[KEY]


def prepare_output_directory(output_path: str):
    """Prepare the output directory."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)


def save_image(output_path: str, image: np.ndarray):
    """Save an image to a file."""
    scipy.io.savemat(output_path, {KEY: image})

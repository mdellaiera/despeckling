import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

KEY = 'data'


def c2ap(image: np.ndarray) -> np.ndarray:
    """Convert a complex image to amplitude."""
    amplitude = 20. * np.log1p(np.abs(image))
    phase = np.angle(image)
    return amplitude, phase


def ap2c(amplitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Convert amplitude and phase back to complex representation."""
    return (np.exp(amplitude / 20.) - 1) * np.exp(1j * phase)


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
    plt.imsave(output_path.replace('.mat', '.png'), c2ap(image)[0], cmap='gray')  # Preview
    scipy.io.savemat(output_path, {KEY: image})

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io


def c2ap(image: np.ndarray) -> np.ndarray:
    """Convert a complex image to amplitude."""
    if not np.iscomplexobj(image):
        return image, None
    amplitude = 20. * np.log1p(np.abs(image))
    phase = np.angle(image)
    return amplitude, phase


def ap2c(amplitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Convert amplitude and phase back to complex representation."""
    if phase is None:
        return amplitude
    return (np.exp(amplitude / 20.) - 1) * np.exp(1j * phase)


def read_image(input_path: str, key: str) -> np.ndarray:
    """Read an image from a file."""
    if input_path.endswith('.npy') or input_path.endswith('.npz'):
        data = np.load(input_path)
    elif input_path.endswith('.mat'):
        data = scipy.io.loadmat(input_path)
    else:
        raise ValueError("Unsupported file format. Use .npy or .mat files.")
    if key not in data:
        raise ValueError(f"Input file must contain '{key}' key.")
    return data[key]


def prepare_output_directory(output_path: str):
    """Prepare the output directory."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)


def save_image(output_path: str, image: np.ndarray, key: str):
    """Save an image to a file."""
    plt.imsave(output_path.replace('.mat', '.png'), c2ap(image)[0].squeeze(), cmap='gray')  # Preview
    scipy.io.savemat(output_path, {key: image})

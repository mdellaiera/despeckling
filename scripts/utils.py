import numpy as np
import jax.numpy as jnp
from typing import Tuple, Union, Dict


def c2ap(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a complex image to amplitude.
    
    Arguments:
        image (np.ndarray): A complex-valued image of shape (H, W) or (H, W, C).
    Returns:
        amplitude (np.ndarray): The amplitude representation of the image.
        phase (np.ndarray): The phase representation of the image, or None if the input is not complex.
    """
    amplitude = 20. * np.log1p(np.abs(image))
    if not np.iscomplexobj(image):
        return amplitude, None
    phase = np.angle(image)
    return amplitude, phase


def ap2c(amplitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Convert amplitude and phase back to complex representation.

    Arguments:
        amplitude (np.ndarray): The amplitude representation of the image.
        phase (np.ndarray): The phase representation of the image, or None.
    Returns:
        image (np.ndarray): A complex-valued image reconstructed from amplitude and phase if phase is provided,
                            otherwise returns the amplitude as is.
    """
    modulus = np.exp(amplitude / 20.) - 1
    if phase is None:
        return modulus
    return modulus * np.exp(1j * phase)


def rgb2gray(rgb: Union[np.ndarray, jnp.ndarray]) -> Union[np.ndarray, jnp.ndarray]:
    """Convert RGB image to grayscale.
    
    Arguments:
        rgb (Union[np.ndarray, jnp.ndarray]): An RGB image of shape (H, W, 3).
    Returns:
        gray (Union[np.ndarray, jnp.ndarray]): The grayscale representation of the image.
    """
    if isinstance(rgb, np.ndarray):
        return np.dot(rgb[..., :3], np.array([1./3.] * 3))
    return jnp.dot(rgb[..., :3], jnp.array([1./3.] * 3))


def T(sar: np.ndarray, expand_dims: bool = False) -> Tuple[np.ndarray, Dict]:
    sar = jnp.array(sar)
    sar = 20 * np.log1p(np.abs(sar))
    sar = (sar - sar.min()) / (sar.max() - sar.min())  # Normalize to [0, 1]
    if sar.ndim == 2 and expand_dims:
        sar = jnp.expand_dims(sar, axis=-1)  # (H, W) -> (H, W, 1)
    return sar, {'min': sar.min(), 'max': sar.max()}


def invT(sar: np.ndarray, params: Dict={'min': 0, 'max': 1}) -> np.ndarray:
    sar = jnp.array(sar)
    sar = sar * (params['max'] - params['min']) + params['min']
    sar = jnp.expm1(sar / 20)
    return sar


def standardize(image: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """Standardize image to zero mean and unit variance."""
    mean = np.mean(image)
    std = np.std(image)
    standardized_image = (image - mean) / std
    return standardized_image, {'standardize_mean': mean, 'standardize_std': std}


def clip(image: np.ndarray, vmin: float=0., vmax: float=1.) -> Tuple[np.ndarray, Dict]:
    """Clip image between vmin and vmax."""
    vmin = image.min()
    vmax = image.max()
    return np.clip(image, vmin, vmax), {'clip_min': vmin, 'clip_max': vmax}


class BaseFilter:
    """Base class for despeckling methods."""

    def  __init__(self):
        pass
    
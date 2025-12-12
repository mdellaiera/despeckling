import jax.numpy as jnp
from typing import List
from tqdm import tqdm
from scripts.parallelisation import extract_patches


def compute_enl(img: jnp.ndarray, window_size: int) -> jnp.ndarray:
    radius = window_size // 2
    img_padded = jnp.pad(img, ((radius, radius), (radius, radius)), mode='reflect')  # (H', W')

    # Extract patches
    patches = extract_patches(img_padded, window_size, (0, 0), img_padded.shape[:2])  # (H', W', k, k)

    # Compute local mean and variance
    local_mean = patches.mean(axis=(-2, -1))  # (H', W')
    local_var = patches.var(axis=(-2, -1))  # (H', W')

    # Compute ENL
    enl = local_mean**2 / (local_var + 1e-10)  # Avoid division by zero
    return enl[radius:-radius, radius:-radius]  # Unpad to get output of shape (H, W)


class ENLClassifier:
    """Soft classifier to linearly combine multiple filtered estimates."""

    def __init__(self):
        self.at = []
    
    def classify(self, sar: jnp.ndarray, N: List[int], gamma: float = 7, a0: float = 0.64) -> jnp.ndarray:
        """
        Compute the soft classification weight for SAR despeckling.

        Parameters:
        sar : jnp.ndarray
            Input SAR image of shape (H, W).
        N : List[int]
            List of equivalent look numbers for each class.
        gamma : float
            Parameter for the soft classification weight.
        a0 : float
            Parameter for the soft classification weight.

        Returns:
        jnp.ndarray
            The soft classification weight of shape (H, W).
        """
        self.at = []
        for n in tqdm(N):
            self.at.append(compute_enl(sar, window_size=n))
        at = jnp.mean(jnp.stack(self.at), axis=0)
        ft = 1. / (jnp.exp(-gamma * (at - a0)) + 1)
        return ft
    
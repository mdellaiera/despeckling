import jax.numpy as jnp
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

import jax.numpy as jnp


def gaussian_kernel_1d(radius: int) -> jnp.ndarray:
    """
    Create a 1D Gaussian filter kernel with the specified radius.
    The standard deviation is set to radius / 3.
    
    Parameters:
    radius: int
        Radius of the Gaussian filter.

    Returns:
    kernel: jnp.ndarray
        Gaussian filter kernel of shape (2*radius+1, 1).
    """
    if radius < 1:
        raise ValueError("Radius must be at least 1.")
    sigma = radius / 3
    x = jnp.arange(-radius, radius + 1)
    kernel = jnp.exp(-x * x / (2 * sigma * sigma))
    kernel = kernel / sum(kernel)
    return kernel.reshape(-1, 1)


def gaussian_kernel_2d(radius: int) -> jnp.ndarray:
    """
    Create a 2D Gaussian filter kernel with the specified radius.
    The standard deviation is set to radius / 3.
    
    Parameters:
    radius: int
        Radius of the Gaussian filter.
    
    Returns:
    kernel: jnp.ndarray
        2D Gaussian filter kernel of shape (2*radius+1, 2*radius+1).
    """
    gaussian_1d = gaussian_kernel_1d(radius)
    return gaussian_1d @ gaussian_1d.T

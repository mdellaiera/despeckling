import jax.numpy as jnp
from jax import vmap
from jax.scipy.signal import convolve as conv2
from skimage.color import rgb2lab as skimage_rgb2lab
from scripts.filters import gaussian_kernel_1d


def rgb2lab(img: jnp.ndarray) -> jnp.ndarray:
    """
    Convert an RGB image to Lab-CIE color space.
    
    Parameters:
    img: jnp.ndarray
        Input image of shape (H, W, 3).

    Returns:
    jnp.ndarray
        Image in Lab-CIE color space of shape (H, W, 3).
    """
    img_lab = skimage_rgb2lab(img)
    return img_lab


def rgb2gray(img: jnp.ndarray) -> jnp.ndarray:
    """
    Convert an RGB image to grayscale.
    
    Parameters:
    img: jnp.ndarray
        Input image of shape (H, W, 3).
    
    Returns:
    jnp.ndarray
        Grayscale image of shape (H, W, 1).
    """
    return jnp.dot(img[..., :3], jnp.ones(3) / 3)[..., None]


def compute_feature(img: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the feature vector for the given image.

    Parameters:
    img: jnp.ndarray
        Input image of shape (H, W, C). The first channel must correspond to the luminance.
        If the image is RGB, convert it in the Lab-CIE color space.
    Returns:
    feature_vector: jnp.ndarray
        Feature vector of shape (H, W, D).
    """
    assert img.ndim == 3, "Input image must be 3D (H, W, C)."

    # Define the spatial filters
    filter_first_order = jnp.array([0.5, 0, -0.5]).reshape(3, 1)
    filter_second_order = jnp.array([1, -2, 1]).reshape(3, 1)

    # Create the feature vector
    L = img[..., 0]
    feature_vector = jnp.dstack([
        conv2(L, filter_first_order, mode='same'),
        conv2(L, filter_first_order.T, mode='same'),
        conv2(L, filter_second_order, mode='same'),
        conv2(L, filter_second_order.T, mode='same'),
        conv2(conv2(L, filter_first_order, mode='same'), filter_first_order.T, mode='same')
    ])
    # Add the original image as the last channel
    feature_vector = jnp.concatenate([feature_vector, img], axis=-1)

    # Give each feature the same importance
    feature_vector / feature_vector.std(axis=(0, 1), keepdims=True)

    return feature_vector


def compute_average(tensor: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the average of each component of the tensor using a Gaussian filter.

    Parameters:
    tensor: jnp.ndarray
        Input tensor of shape (H, W, C).
    kernel: jnp.ndarray
        Gaussian filter kernel of shape (2*radius+1, 1).

    Returns:
    averaged_tensor: jnp.ndarray
        Averaged tensor of shape (H, W, C).
    """
    def monochannel_filter(channel):
        out = conv2(channel, kernel, mode='same')
        out = conv2(out, kernel.T, mode='same')
        return out

    averaged_tensor = vmap(monochannel_filter, in_axes=2, out_axes=2)(tensor)
    return averaged_tensor


def compute_covariance(tensor: jnp.ndarray, tensor_averaged: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the covariance matrix for each pixel in the tensor.

    Parameters:
    tensor: jnp.ndarray
        Input tensor of shape (H, W, C)
    tensor_averaged: 
        Tensor containing the average of each features of shape (H, W, C)
    kernel: jnp.ndarray
        Gaussian filter kernel of shape (2*radius+1, 1).
    Returns:
    covariance_matrix: jnp.ndarray
        The resulting covariance matrix of shape (H, W, C, C)
    """
    # Get all products t_i * t_j for all channel pairs (H, W, C, C)
    t_i = tensor[..., None]   # (H, W, C, 1)
    t_j = tensor[..., None, :]  # (H, W, 1, C)
    products = t_i * t_j       # (H, W, C, C)

    # Vectorize the Gaussian filtering across channel pairs using vmap
    def filter_2d(mat):
        out = conv2(mat, kernel, mode='same')
        out = conv2(out, kernel.T, mode='same')
        return out
    # Batch filtering over last two axes (C, C)
    products_filtered = vmap(
        vmap(filter_2d, in_axes=2, out_axes=2), in_axes=3, out_axes=3
    )(products)

    # Compute mean_i * mean_j (outer product for each pixel)
    mean_i = tensor_averaged[..., None]  # (H, W, C, 1)
    mean_j = tensor_averaged[..., None, :] # (H, W, 1, C)
    mean_prod = mean_i * mean_j           # (H, W, C, C)

    # Covariance by formula
    covariance_matrix = products_filtered - mean_prod  # (H, W, C, C)
    return covariance_matrix


def cholesky_decomposition(tensor: jnp.ndarray) -> jnp.ndarray:
    """
    Perform Cholesky decomposition on a batch of covariance matrices.

    Parameters:
    tensor: jnp.ndarray
        Covariance tensor of shape (H, W, C, C).

    Returns:
    L: jnp.ndarray
        Lower triangular matrix of shape (H, W, C(C+1)/2).
    """
    # First, we need to vmap over the last two batch dims (H, W) for cholesky
    # vmap twice, once over axis=0 (H), then axis=0 (W) of what's left
    def cholesky_decomposition_and_tril(x: jnp.ndarray) -> jnp.ndarray:
        indices = jnp.tril_indices(x.shape[-1])
        return jnp.linalg.cholesky(x)[indices]
    
    cholesky_2d = vmap(vmap(cholesky_decomposition_and_tril, in_axes=0), in_axes=0)
    L = cholesky_2d(tensor)
    return L


def add_averaged(tensor: jnp.ndarray, tensor_averaged: jnp.ndarray) -> jnp.ndarray:
    """
    Add the averaged tensor to the original tensor, along the last axis.

    Parameters:
    tensor: jnp.ndarray
        Input tensor of shape (H, W, C(C+1)/2).
    tensor_averaged: jnp.ndarray
        Averaged tensor of shape (H, W, C).

    Returns:
    descriptor: jnp.ndarray
        Descriptor tensor of shape (H, W, C(C+1)/2+C).
    """
    # Concatenate along the last axis
    descriptor = jnp.concatenate([tensor, tensor_averaged], axis=-1)
    return descriptor


def compute_texture_descriptor(img: jnp.ndarray, radius: int) -> jnp.ndarray:
    """
    Compute the texture descriptor for the given image.

    Parameters:
    img: jnp.ndarray
        Input image of shape (H, W).
    radius: int
        Radius for the Gaussian filter.
    Returns:
    descriptor: jnp.ndarray
        Texture descriptor of shape (H, W, C(C+1)/2+C).
    """
    feature = compute_feature(img)
    kernel = gaussian_kernel_1d(radius)
    feature_averaged = compute_average(feature, kernel)
    covariance = compute_covariance(feature, feature_averaged, kernel)
    cholesky = cholesky_decomposition(covariance)
    descriptor = add_averaged(cholesky, feature_averaged)

    return descriptor


def compute_similarity_map(S_p: jnp.ndarray, S: jnp.ndarray, sigma_d: float = 0.1):
    """
    Compute similarity map for a given descriptor tensor.

    ----------
    Parameters
    S_p : jnp.ndarray
        Descriptor vector for the pixel (x, y) for which to compute the similarity map.
    S : jnp.ndarray
        Descriptor tensor of shape (H, W, D).
    sigma_d : float
        Standard deviation for the descriptor space.
    Returns
    jnp.ndarray
        Similarity map of shape (H, W).
    """
    distsq = jnp.sum((S - S_p)**2, axis=-1)  # (H, W)
    similarity_map = jnp.exp(-distsq / (2 * sigma_d ** 2))
    return similarity_map

import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple, List


def extract_patches(
        tensor: jnp.ndarray, 
        kernel_size: int,
        start_index: Tuple[int, int],
        end_indices: Tuple[int, int]) -> jnp.ndarray:
    """
    Extracts sliding patches from a 2D or 3D tensor.
    This function is designed to be memory efficient by extracting patches only within the specified indices.

    Parameters:
    tensor: 
        Input tensor of size (H, W, D) or (H, W)
    kernel_size: int
        Size of the square kernel (must be odd).
    start_index: Tuple[int, int]
        Starting indices for the patch extraction.
    end_indices: Tuple[int, int]
        Ending indices for the patch extraction.
    Returns: 
    jnp.ndarray
        A tensor of size (H', W', kernel_size, kernel_size, D) or (H', W', kernel_size, kernel_size) containing the extracted patches,
        where H' and W' are the dimensions defined by the start and end indices.
    """
    D = tensor.shape[2:]  # Tuple corresponding to the size of the descriptor, which can be empty if input tensor is 2D
    pad = kernel_size // 2  # Integer, the pad size usually corresponds to half the size of the kernel

    # Pad the input tensor to handle borders. The last part '((0, 0),) * len(D)' allows to handle N-D tensors.
    tensor = jnp.pad(tensor, ((pad, pad), (pad, pad)) + ((0, 0),) * len(D), mode='reflect')

    # Create indices for the sliding window
    h_idx = jnp.arange(start_index[0], end_indices[0])
    w_idx = jnp.arange(start_index[1], end_indices[1])

    # Create a function to extract patches for each (i, j) pair
    def get_patch(i: int, j: int) -> jnp.ndarray:
        patch = jax.lax.dynamic_slice(operand=tensor, 
                                      start_indices=(i, j) + (0,)*len(D), 
                                      slice_sizes=(kernel_size, kernel_size) + D)
        return patch

    # Use vmap to vectorize the extraction of patches across the height and width indices
    patches = jax.vmap(
        lambda i: jax.vmap(lambda j: get_patch(i, j))(w_idx)
    )(h_idx)
    return patches 


def compute_indices_from_n_blocks(
        n_blocks: int, 
        H: int, 
        W: int,
        padding: int = 0
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Computes the starting and ending indices for extracting patches from a tensor.
    
    Returns:
    Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]
        Starting and ending indices for patch extraction.
    """
    H = H + 2 * padding
    W = W + 2 * padding

    # Compute the split points along each dimension
    h_edges = np.linspace(0, H, n_blocks + 1, dtype=int)
    w_edges = np.linspace(0, W, n_blocks + 1, dtype=int)

    # Generate all combinations of start and end indices
    start_indices = []
    end_indices = []
    for i in range(n_blocks):
        for j in range(n_blocks):
            start = (h_edges[i], w_edges[j])
            end = (h_edges[i+1], w_edges[j+1])
            start_indices.append(start)
            end_indices.append(end)
    return start_indices, end_indices

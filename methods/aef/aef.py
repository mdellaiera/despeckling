import os
import jax
import jax.numpy as jnp
from tqdm import tqdm
import logging
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../scripts')
from utils import extract_patches, compute_indices_from_n_blocks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SARDespeckling:
    """Despeckling of SAR images using AEF embeddings."""

    def filter(
            self, 
            sar: jnp.ndarray, 
            embeddings: jnp.ndarray, 
            sigma_distance: float, 
            radius_despeckling: int, 
            n_blocks: int = 10) -> jnp.ndarray:
        """
        Despeckle SAR image using texture descriptors.
        
        Parameters
        sar : jnp.ndarray
            Input SAR image of shape (H, W, 1).
        embeddings : jnp.ndarray
            Embeddings tensor of shape (H, W, D) where D is the number of descriptors.
        sigma_distance : float
            Standard deviation for the Gaussian kernel used in similarity computation.
        radius_despeckling : int
            Radius to consider neighboring pixels.
        n_blocks : int
            Number of blocks for processing the image in parallel.

        Returns
        jnp.ndarray
            Filtered SAR image of shape (H, W, 1).
        """
        H, W, _ = embeddings.shape
        kernel_size = 2 * radius_despeckling + 1  # k = 2 * r + 1

        # Pad the input arrays to handle borders
        pad_width = ((radius_despeckling, radius_despeckling), (radius_despeckling, radius_despeckling), (0, 0))
        embeddings_pad = jnp.pad(embeddings.copy(), pad_width, mode='reflect')  # (H+2*r, W+2*r, D)
        sar_pad = jnp.pad(sar.copy(), pad_width, mode='reflect')  # (H+2*r, W+2*r, 1)

        # Compute start and end indices for memory efficiency
        start_indices, end_indices = compute_indices_from_n_blocks(n_blocks, H, W, padding=radius_despeckling)

        sar_filtered = jnp.zeros_like(sar_pad)  # Initialize output tensor

        # Iterate over the blocks
        progress_bar = tqdm(total=len(start_indices), desc="Despeckling", unit="block")
        for start_index, end_index in zip(start_indices, end_indices):
            # Extract windows (patches) for all spatial locations
            embeddings_patches = extract_patches(embeddings_pad, kernel_size, start_index, end_index)   # (H', W', k, k, D)
            sar_patches = extract_patches(sar_pad, kernel_size, start_index, end_index)  # (H', W', k, k, 1)

            embeddings_centers = embeddings_patches[..., radius_despeckling, radius_despeckling, :][..., None, None, :]  # Centers are located at (r, r)

            # Compute similarity map
            difference = embeddings_patches - embeddings_centers # (H', W', k, k, D)
            distsq = jnp.sum((difference)**2, axis=-1) 
            similarity_map = jnp.exp(-distsq / (2 * sigma_distance ** 2))[..., None]
            Z = 1 / similarity_map.sum(axis=(-3, -2))  # (H', W')

            # Filtering
            update_block = (similarity_map * sar_patches).sum(axis=(-3, -2)) * Z  # (H', W', 1)

            # Update only the current block
            sar_filtered = jax.lax.dynamic_update_slice(sar_filtered, update_block, start_index + (0,))  
            progress_bar.update(1)
        progress_bar.close()
        
        # Remove padding
        return sar_filtered[radius_despeckling:-radius_despeckling, radius_despeckling:-radius_despeckling, :]
    
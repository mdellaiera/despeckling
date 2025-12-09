import numpy as np
import jax.numpy as jnp
from scripts.utils import BaseFilter
from scripts.despeckle import despeckle


class AEF(BaseFilter):
    """Wrapper around AEF filter."""

    def filter(
            self, 
            sar: np.ndarray, 
            embeddings: np.ndarray, 
            sigma_distance: float, 
            radius_despeckling: int, 
            n_blocks: int = 10) -> jnp.ndarray:
        """
        Despeckle SAR image using AEF embeddings.
        
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
        input_embeddings = jnp.array(embeddings, dtype=jnp.float32)
        input_sar = jnp.array(sar, dtype=jnp.float32)
        sar_filtered = despeckle(
            sar=input_sar,
            descriptor=input_embeddings,
            sigma_distance=sigma_distance,
            radius_despeckling=radius_despeckling,
            n_blocks=n_blocks
        )
        return np.array(sar_filtered)
    
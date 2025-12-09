import jax
import jax.numpy as jnp
from typing import List
import logging
from scripts.texture_descriptor import compute_texture_descriptor
from scripts.mubf import MUBF
from scripts.despeckle import despeckle
import time


class TBOG:
    """Despeckling of SAR images using texture descriptors."""

    def __init__(self):
        self.error = None
    
    def _check_inputs(self, sar: jnp.ndarray, eo: jnp.ndarray):
        """
        Check the validity of input tensors sar and eo.
        
        Parameters:
        sar : jnp.ndarray
            Input SAR image of shape (H, W, 1).
        eo : jnp.ndarray
            Input optical image of shape (H, W, 3) or (H, W, 1).
        """
        assert sar.ndim == 3 and sar.shape[-1] == 1, "SAR image should be (H, W, 1)"
        assert eo.ndim == 3 and eo.shape[-1] in [1, 3], "Optical image should be (H, W, 1) or (H, W, 3)"

    def filter(
        self,
        sar: jnp.ndarray,
        eo: jnp.ndarray,
        radius_descriptor: int,
        sigma_spatial: float,
        sigma_guides: List[float],  # [eo, sar]
        gamma_guides: List[float],  # [eo, sar]
        alpha: float,
        n_iterations: int,
        n_blocks_mubf: int,
        sigma_distance: float,
        radius_despeckling: int,
        n_blocks_despeckling: int) -> jnp.ndarray:
        """
        Run the despeckling process.

        Returns
        jnp.ndarray
            Filtered SAR image.
        """
        time_start_tbog = time.time()

        input_sar = jnp.array(sar, dtype=jnp.float32)[..., None]
        input_eo = jnp.array(eo, dtype=jnp.float32)
        self._check_inputs(input_sar, input_eo)

        logging.info("Computing texture descriptor...")
        time_start_descriptor = time.time()
        descriptor = compute_texture_descriptor(img=eo, radius=radius_descriptor)
        if jnp.isnan(descriptor).any():
            logging.info("Warning: NaN values found in texture descriptor. Replacing with zeros.")
            descriptor = jnp.nan_to_num(descriptor, nan=0.0)
        time_end_descriptor = time.time()
        logging.info(f"Texture descriptor computed in {time_end_descriptor - time_start_descriptor:.2f} seconds.")

        logging.info("Filtering texture descriptor with MUBF...")
        mubf = MUBF()
        time_start_mubf = time.time()
        descriptor_filtered, error = mubf.filter(
            target=descriptor, 
            guides=[input_eo.mean(axis=-1)[..., None], input_sar], 
            sigma_spatial=sigma_spatial, 
            sigma_guides=sigma_guides, 
            gamma_guides=gamma_guides,
            alpha=alpha, 
            n_iterations=n_iterations, 
            n_blocks=n_blocks_mubf
        )
        time_end_mubf = time.time()
        logging.info(f"Texture descriptor filtering completed in {time_end_mubf - time_start_mubf:.2f} seconds.")
        self.error = error
        
        logging.info("Despeckling SAR image...")
        time_start_despeckling = time.time()
        sar_filtered = despeckle(
            sar=input_sar, 
            descriptor=descriptor_filtered, 
            sigma_distance=sigma_distance, 
            radius_despeckling=radius_despeckling, 
            n_blocks=n_blocks_despeckling
        )
        time_end_despeckling = time.time()
        logging.info(f"Despeckling completed in {time_end_despeckling - time_start_despeckling:.2f} seconds.")
        
        time_end_tbog = time.time()
        logging.info(f"Total despeckling process completed in {time_end_tbog - time_start_tbog:.2f} seconds.")

        return sar_filtered
    
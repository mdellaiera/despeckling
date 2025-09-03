import jax
import jax.numpy as jnp
from typing import List
from tqdm import tqdm
import logging
from scripts.utils import extract_patches, compute_indices_from_n_blocks
from scripts.texture_descriptor import compute_texture_descriptor
from scripts.mubf import MUBF
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SARDespeckling:
    """Despeckling of SAR images using texture descriptors."""

    def __init__(self):
        self.error = None

    def _despeckle(self, sar: jnp.ndarray, descriptor: jnp.ndarray, sigma_distance: float, radius_despeckling: float, n_blocks: int = 10) -> jnp.ndarray:
        """
        Despeckle SAR image using texture descriptors.
        
        Parameters
        sar : jnp.ndarray
            Input SAR image of shape (H, W, 1).
        descriptor : jnp.ndarray
            Texture descriptor tensor of shape (H, W, D) where D is the number of descriptors.
        sigma_distance : float
            Standard deviation for the Gaussian kernel used in similarity computation.
        radius_despeckling : float
            Radius to consider neighboring pixels.
        n_blocks : int
            Number of blocks for processing the image in parallel.

        Returns
        jnp.ndarray
            Filtered SAR image of shape (H, W, 1).
        """
        H, W, _ = descriptor.shape
        kernel_size = 2 * radius_despeckling + 1  # k = 2 * r + 1

        # Pad the input arrays to handle borders
        pad_width = ((radius_despeckling, radius_despeckling), (radius_despeckling, radius_despeckling), (0, 0))
        descriptor_pad = jnp.pad(descriptor.copy(), pad_width, mode='reflect')  # (H+2*r, W+2*r, D)
        sar_pad = jnp.pad(sar.copy(), pad_width, mode='reflect')  # (H+2*r, W+2*r, 1)

        # Compute start and end indices for memory efficiency
        start_indices, end_indices = compute_indices_from_n_blocks(n_blocks, H, W, padding=radius_despeckling)

        sar_filtered = jnp.zeros_like(sar_pad)  # Initialize output tensor

        # Iterate over the blocks
        progress_bar = tqdm(total=len(start_indices), desc="Despeckling", unit="block")
        for start_index, end_index in zip(start_indices, end_indices):
            # Extract windows (patches) for all spatial locations
            descriptor_patches = extract_patches(descriptor_pad, kernel_size, start_index, end_index)   # (H', W', k, k, D)
            sar_patches = extract_patches(sar_pad, kernel_size, start_index, end_index)  # (H', W', k, k, 1)

            descriptor_centers = descriptor_patches[..., radius_despeckling, radius_despeckling, :][..., None, None, :]  # Centers are located at (r, r)

            # Compute similarity map
            difference = descriptor_patches - descriptor_centers # (H', W', k, k, D)
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
    
    def _check_inputs(self, sar: jnp.ndarray, opt: jnp.ndarray):
        """
        Check the validity of input tensors sar and opt.
        
        Parameters:
        sar : jnp.ndarray
            Input SAR image of shape (H, W, 1).
        opt : jnp.ndarray
            Input optical image of shape (H, W, 3) or (H, W, 1).
        """
        assert sar.ndim == 3 and sar.shape[-1] == 1, "SAR image should be (H, W, 1)"
        assert opt.ndim == 3 and opt.shape[-1] in [1, 3], "Optical image should be (H, W, 1) or (H, W, 3)"

    def _step1_compute_texture_descriptor(self, opt: jnp.ndarray, radius: float) -> jnp.ndarray:
        """
        Compute the texture descriptor for the optical image.

        Parameters
        opt : jnp.ndarray
            Input optical image of shape (H, W, 3) or (H, W, 1).
        radius : float
            Radius for the texture descriptor.

        Returns
        jnp.ndarray
            Texture descriptor of shape (H, W, D) where D is the number of descriptors.
        """
        logging.info("Computing texture descriptor...")
        time_start = time.time()
        desc = compute_texture_descriptor(img=opt, radius=radius)
        if jnp.isnan(desc).any():
            logging.info("Warning: NaN values found in texture descriptor. Replacing with zeros.")
            desc = jnp.nan_to_num(desc, nan=0.0)
        time_end = time.time()
        logging.info(f"Texture descriptor computed in {time_end - time_start:.2f} seconds.")
        return desc

    def _step2_filter_texture_descriptor(self, 
                                         target: jnp.ndarray, 
                                         guides: List[jnp.ndarray],
                                         sigma_spatial: float, 
                                         sigma_guides: List[float],
                                         alpha: float, 
                                         n_iterations: int, 
                                         n_blocks: int) -> jnp.ndarray:
        """
        Filter the texture descriptor using MUBF.

        Parameters
        descriptor : jnp.ndarray
            Texture descriptor of shape (H, W, D).
        opt : jnp.ndarray
            Input optical image of shape (H, W, 3) or (H, W, 1).
        sar : jnp.ndarray
            Input SAR image of shape (H, W, 1).
        sigma_spatial : float
            Spatial standard deviation for Gaussian kernel.
        sigma_luminance_opt : float
            Optical luminance standard deviation for Gaussian weighting.
        sigma_luminance_sar : float
            SAR luminance standard deviation for Gaussian weighting.
        alpha : float
            Scaling factor for the update step.
        n_iterations : int
            Number of iterations for the filtering process.
        n_blocks : int
            Number of blocks for processing the image in parallel.
        Returns
        Tuple[jnp.ndarray, List[float]]
            Filtered texture descriptor of shape (H, W, D) and a list of errors
            for each iteration.
        """
        logging.info("Filtering texture descriptor with MUBF...")
        mubf = MUBF()
        time_start = time.time()
        # opt.mean(axis=-1, keepdims=True)
        desc_filtered, error = mubf.filter(target=target, 
                                           guides=guides, 
                                           sigma_spatial=sigma_spatial, 
                                           sigma_guides=sigma_guides, 
                                           alpha=alpha, 
                                           n_iterations=n_iterations, 
                                           n_blocks=n_blocks)
        time_end = time.time()
        logging.info(f"Texture descriptor filtering completed in {time_end - time_start:.2f} seconds.")
        self.error = error
        return desc_filtered
    
    def _step3_despeckle(self, 
                         sar: jnp.ndarray, 
                         descriptor: jnp.ndarray,
                         sigma_distance: float, 
                         radius_despeckling: int, 
                         n_blocks: int) -> jnp.ndarray:
        """        
        Despeckle the SAR image using the filtered texture descriptor.

        Parameters
        sar : jnp.ndarray
            Input SAR image of shape (H, W, 1).
        descriptor : jnp.ndarray
            Filtered texture descriptor of shape (H, W, D).
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
        logging.info("Despeckling SAR image...")
        time_start = time.time()
        sar_filtered = self._despeckle(sar=sar, 
                                       descriptor=descriptor, 
                                       sigma_distance=sigma_distance, 
                                       radius_despeckling=radius_despeckling, 
                                       n_blocks=n_blocks)
        time_end = time.time()
        logging.info(f"Despeckling completed in {time_end - time_start:.2f} seconds.")
        return sar_filtered
    
    def get_error(self) -> List[float]:
        """
        Get the error list from the last DUBF step.

        Returns
        List[float]
            List of errors for each iteration.
        """
        if self.error is None:
            raise ValueError("No DUBF has been performed yet. Call run() first.")
        return self.error

    def run(self,
            sar: jnp.ndarray,
            opt: jnp.ndarray,
            radius_descriptor: float = 21,
            sigma_spatial: float = 5,
            sigma_luminance_opt: float = 0.1,
            sigma_luminance_sar: float = 0.1,
            alpha: float = 1.0,
            n_iterations: int = 30,
            n_blocks_dubf: int = 10,
            sigma_distance: float = 1.5,
            radius_despeckling: int = 50,
            n_blocks_despeckling: int = 100) -> jnp.ndarray:
        """
        Run the despeckling process.

        Returns
        jnp.ndarray
            Filtered SAR image.
        """
        time_start = time.time()
        self._check_inputs(sar, opt)

        # Compute the texture descriptor on the optical image
        descriptor = self._step1_compute_texture_descriptor(opt, radius=radius_descriptor)

        # Filter the texture descriptor using MUBF
        descriptor_filtered = self._step2_filter_texture_descriptor(
            descriptor=descriptor,
            opt=opt,
            sar=sar,
            sigma_spatial=sigma_spatial,
            sigma_luminance_opt=sigma_luminance_opt,
            sigma_luminance_sar=sigma_luminance_sar,
            alpha=alpha, 
            n_iterations=n_iterations, 
            n_blocks=n_blocks_dubf
        )
        
        # Despeckle the SAR image using the filtered texture descriptor
        sar_filtered = self._step3_despeckle(
            sar=sar, 
            descriptor_filtered=descriptor_filtered, 
            sigma_distance=sigma_distance, 
            radius_despeckling=radius_despeckling, 
            n_blocks=n_blocks_despeckling
        )
        
        time_end = time.time()
        logging.info(f"Total despeckling process completed in {time_end - time_start:.2f} seconds.")

        return sar_filtered
    
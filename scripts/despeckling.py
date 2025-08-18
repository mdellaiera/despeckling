import jax
import jax.numpy as jnp
from typing import Tuple, List, Callable
from tqdm import tqdm
import logging
from scripts.filters import gaussian_kernel_2d
from scripts.utils import extract_patches, compute_indices_from_n_blocks
from scripts.texture_descriptor import compute_texture_descriptor
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class _BaseFilter:
    """Base class for filters."""

    def __init__(self):
        pass

    def _check_inputs(self, *args, **kwargs):
        """
        Check the validity of input tensors.
        This method should be overridden in subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def _loop_over_blocks(self, start_indices: List[Tuple[int, int]], end_indices: List[Tuple[int, int]], 
                                    func: Callable, *args, **kwargs) -> jnp.ndarray:
        """
        Loop over blocks of the image and apply a function.

        Parameters:
        start_indices : List[Tuple[int, int]]
            List of starting indices for each block.
        end_indices : List[Tuple[int, int]]
            List of ending indices for each block.
        func : Callable
            Function to apply to each block.
        *args, **kwargs : Additional arguments to pass to the function.
        Returns:
        jnp.ndarray
            Resulting tensor after applying the function to each block.
        """
        return NotImplementedError
    
    def _loop_over_iterations(self, n_iterations: int, func: Callable, *args, **kwargs) -> List[float]:
        """
        Loop over a number of iterations and apply a function.

        Parameters:
        n_iterations : int
            Number of iterations to perform.
        func : Callable
            Function to apply in each iteration.
        *args, **kwargs : Additional arguments to pass to the function.
        Returns:
        List[float]
            List of errors or results from each iteration.
        """
        return NotImplementedError
    

class UBF(_BaseFilter):
    """Unnormalized Bilateral Filter (UBF)."""

    def __init__(self):
        pass

    def _check_inputs(self, descriptor: jnp.ndarray, luminance: jnp.ndarray):
        """
        Check the validity of input tensors descriptor and luminance.

        Parameters:
        descriptor : jnp.ndarray
            Input tensor of shape (H, W, D) where D is the number of descriptors, H is height, and W is width.
        luminance : jnp.ndarray
            Luminance tensor of shape (H, W, 1) used for weighting.
        """
        assert descriptor.ndim == 3, "descriptor should be (H, W, D)"
        assert luminance.ndim == 3 and luminance.shape[-1] == 1, "luminance should be (H, W, 1)"

    def _compute_luminance_weights(self, 
                                  luminance: jnp.ndarray, 
                                  radius: int, 
                                  sigma_luminance: float, 
                                  start_index: Tuple[int, int], 
                                  end_index: Tuple[int, int]) -> jnp.ndarray:
        """
        Compute the luminance weights.

        Parameters:
        luminance : jnp.ndarray
            The luminance map of shape (H, W, 1).
        radius : int
            The radius of the Gaussian kernel.
        sigma_luminance : float
            The standard deviation of the Gaussian kernel.
        start_index : Tuple[int, int]
            The starting index for the patch extraction.
        end_index : Tuple[int, int]
            The ending index for the patch extraction.

        Returns:
        jnp.ndarray
            The computed luminance weights.
        """
        kernel_size = 2 * radius + 1  # k = 2 * r + 1
        patches = extract_patches(luminance, kernel_size, start_index, end_index)   # (H', W', k, k, 1)
        centers = patches[..., radius, radius, :]  # (H', W', 1)
        centers = centers[..., None, None, :] # (H', W', 1, 1, 1)
        weights = jnp.exp(-(patches - centers)**2 / (2 * sigma_luminance ** 2))  # (H', W', k, k, 1)
        return weights
    
    def _compute_gaussian_weights(self, radius: int) -> jnp.ndarray:
        """
        Compute the Gaussian weights for the spatial domain.

        Parameters:
        radius : int
            The radius of the Gaussian kernel.

        Returns:
        jnp.ndarray
            The Gaussian weights of shape (k, k).
        """
        weights = gaussian_kernel_2d(radius)  # (k, k)
        return weights

    def filter(
        self,
        descriptor: jnp.ndarray,
        luminance: jnp.ndarray,
        sigma_spatial: float = 5,
        sigma_luminance: float = 0.05,
        alpha_update: float = 1.0,
        n_iterations: int = 30,
        n_blocks: int = 10
    ) -> Tuple[jnp.ndarray, List[float]]:
        """
        Apply UBF.

        ----------
        Parameters
        descriptor : jnp.ndarray
            Input tensor of shape (H, W, D) where D is the number of descriptors, H is height, and W is width.
        luminance : jnp.ndarray
            Luminance tensor of shape (H, W, 1) used for weighting.
        sigma_spatial : float
            Spatial standard deviation for Gaussian kernel.
        sigma_luminance : float
            Luminance standard deviation for Gaussian weighting.
        alpha_update : float
            Scaling factor for the update step.
        n_iterations : int
            Number of iterations for the filtering process.
        n_blocks : int
            Number of blocks for processing the image in parallel.
        Returns
        jnp.ndarray
            Filtered tensor of shape (H, W, D) after applying the unnormalized bilateral filter.
        """
        self._check_inputs(descriptor, luminance)

        H, W, _ = descriptor.shape
        radius = int(3 * sigma_spatial)  # r
        kernel_size = 2 * radius + 1  # k = 2 * r + 1

        # Pad the input arrays to handle borders
        pad_width = ((radius, radius), (radius, radius), (0, 0))
        descriptor_initial = jnp.pad(descriptor.copy(), pad_width, mode='reflect')  # (H+2*r, W+2*r, D)
        luminance_initial = jnp.pad(luminance.copy(), pad_width, mode='reflect')  # (H+2*r, W+2*r, 1)

        # Precompute Gaussian weights
        weights_spatial = self._compute_gaussian_weights(radius).reshape(1, 1, kernel_size, kernel_size, 1)  # (1, 1, k, k, 1)

        # Compute start and end indices for memory efficiency
        start_indices, end_indices = compute_indices_from_n_blocks(n_blocks, H, W, padding=radius)

        error = []
        for _ in tqdm(range(n_iterations), desc="UBF Iterations"):
            update = jnp.zeros_like(descriptor_initial)  # Initialize update tensor

            # Iterate over the blocks
            for start_index, end_index in zip(start_indices, end_indices):
                # Extract windows (patches) for all spatial locations
                patches = extract_patches(descriptor_initial, kernel_size, start_index, end_index)   # (H', W', k, k, D)
                weights_luminance = self._compute_luminance_weights(luminance_initial, radius, sigma_luminance, start_index, end_index)   # (H', W', k, k)

                # Combined weight
                weights = weights_spatial * weights_luminance  # (H', W', k, k)

                # Difference from center/target pixel
                centers = patches[..., radius, radius, :]  # Centers are located at (radius, radius)
                difference = patches - centers[..., None, None, :]  # (H', W', k, k, D)

                # Weighted sum over window
                update_block = (weights * difference).sum(axis=(-3, -2))  # (H', W', D)

                # Update only the current block
                update = jax.lax.dynamic_update_slice(update, update_block, start_index + (0,))  

            # Update
            descriptor_updated = descriptor_initial + alpha_update * update  # (H+2*r, W+2*r, D)

            error.append(float(jnp.linalg.norm(descriptor_updated - descriptor_initial)))
            descriptor_initial = descriptor_updated

        # Unpad to get output of shape (H, W, D)
        descriptor_filtered = descriptor_initial[radius:H+radius, radius:W+radius, :]  # (H, W, D)
        return descriptor_filtered, error


class DUBF(UBF):
    """Dual Unnormalized Bilateral Filter (DUBF)."""

    def __init__(self):
        pass

    def _check_inputs(self, descriptor: jnp.ndarray, opt: jnp.ndarray, sar: jnp.ndarray):
        """
        Check the validity of input tensors descriptor, opt, and sar.

        Parameters:
        descriptor : jnp.ndarray
            Input tensor of shape (H, W, D) where D is the number of descriptors, H is height, and W is width.
        opt : jnp.ndarray
            Optical luminance tensor of shape (H, W, 1) used for weighting.
        sar : jnp.ndarray
            SAR luminance tensor of shape (H, W, 1) used for weighting.
        """
        assert descriptor.ndim == 3, "descriptor should be (H, W, D)"
        assert opt.ndim == 3 and opt.shape[-1] == 1, 'opt should be (H, W, 1)'
        assert sar.ndim == 3 and sar.shape[-1] == 1, 'sar should be (H, W, 1)'

    def filter(
        self,
        descriptor: jnp.ndarray,
        opt: jnp.ndarray,
        sar: jnp.ndarray,
        sigma_spatial: float = 5,
        sigma_luminance_opt: float = 0.05,
        sigma_luminance_sar: float = 0.05,
        alpha_update: float = 1.0,
        n_iterations: int = 30,
        n_blocks: int = 10
    ) -> Tuple[jnp.ndarray, List[float]]:
        """
        Apply DUBF.

        ----------
        Parameters
        descriptor : jnp.ndarray
            Input tensor of shape (H, W, D) where D is the number of descriptors, H is height, and W is width.
        opt : jnp.ndarray
            Optical luminance tensor of shape (H, W, 1) used for weighting.
        sar : jnp.ndarray
            SAR luminance tensor of shape (H, W, 1) used for weighting.
        sigma_spatial : float
            Spatial standard deviation for Gaussian kernel.
        sigma_luminance_opt : float
            Optical luminance standard deviation for Gaussian weighting.
        sigma_luminance_sar : float
            SAR luminance standard deviation for Gaussian weighting.
        alpha_update : float
            Scaling factor for the update step.
        n_iterations : int
            Number of iterations for the filtering process.
        n_blocks : int
            Number of blocks for processing the image in parallel.
        Returns
        jnp.ndarray
            Filtered tensor of shape (H, W, D) after applying the unnormalized bilateral filter.
        """
        self._check_inputs(descriptor, opt, sar)

        H, W, _ = descriptor.shape
        radius = int(3 * sigma_spatial)  # r
        kernel_size = 2 * radius + 1  # k = 2 * r + 1

        # Pad the input arrays to handle borders
        pad_width = ((radius, radius), (radius, radius), (0, 0))
        descriptor_initial = jnp.pad(descriptor.copy(), pad_width, mode='reflect')  # (H+2*r, W+2*r, D)
        luminance_opt = jnp.pad(opt.copy(), pad_width, mode='reflect')  # (H+2*r, W+2*r, 1)
        luminance_sar = jnp.pad(sar.copy(), pad_width, mode='reflect')  # (H+2*r, W+2*r, 1)

        # Precompute Gaussian weights
        weights_spatial = self._compute_gaussian_weights(radius).reshape(1, 1, kernel_size, kernel_size, 1)  # (1, 1, k, k, 1)

        # Compute start and end indices for memory efficiency
        start_indices, end_indices = compute_indices_from_n_blocks(n_blocks, H, W, padding=radius)

        error = []
        for _ in tqdm(range(n_iterations), desc="DUBF Iterations"):
            update = jnp.zeros_like(descriptor_initial)  # Initialize update tensor

            # Iterate over the blocks
            for start_index, end_index in zip(start_indices, end_indices):
                # Extract windows (patches) for all spatial locations
                patches = extract_patches(descriptor_initial, kernel_size, start_index, end_index)   # (H', W', k, k, D)
                weights_luminance_opt = self._compute_luminance_weights(luminance_opt, radius, sigma_luminance_opt, start_index, end_index)   # (H', W', k, k)
                weights_luminance_sar = self._compute_luminance_weights(luminance_sar, radius, sigma_luminance_sar, start_index, end_index)   # (H', W', k, k)

                # Combined weight
                weights = weights_spatial * weights_luminance_opt * weights_luminance_sar  # (H, W, k, k, 1)

                # Difference from center/target pixel
                centers = patches[..., radius, radius, :]  # Centers are located at (radius, radius)
                difference = patches - centers[..., None, None, :]  # (H', W', k, k, D)

                # Weighted sum over window
                update_block = (weights * difference).sum(axis=(-3, -2))  # (H', W', D)

                # Update only the current block
                update = jax.lax.dynamic_update_slice(update, update_block, start_index + (0,))  

            # Update
            descriptor_updated = descriptor_initial + alpha_update * update

            error.append(float(jnp.linalg.norm(descriptor_updated - descriptor_initial)))
            descriptor_initial = descriptor_updated

        # Unpad to get output of shape (H, W, D)
        descriptor_filtered = descriptor_initial[radius:H+radius, radius:W+radius, :]
        return descriptor_filtered, error


class SARDespeckling:
    """Despeckling of SAR images using texture descriptors."""

    def __init__(self):
        self.error = None

    def _despeckle(self, sar: jnp.ndarray, descriptor: jnp.ndarray, sigma_distance: float, radius: float, n_blocks: int = 10) -> jnp.ndarray:
        """
        Despeckle SAR image using texture descriptors.
        
        Parameters
        sar : jnp.ndarray
            Input SAR image of shape (H, W, 1).
        descriptor : jnp.ndarray
            Texture descriptor tensor of shape (H, W, D) where D is the number of descriptors.
        sigma_distance : float
            Standard deviation for the Gaussian kernel used in similarity computation.
        radius : float
            Radius to consider neighboring pixels.
        n_blocks : int
            Number of blocks for processing the image in parallel.

        Returns
        jnp.ndarray
            Filtered SAR image of shape (H, W, 1).
        """
        H, W, _ = descriptor.shape
        kernel_size = 2 * radius + 1  # k = 2 * r + 1

        # Pad the input arrays to handle borders
        pad_width = ((radius, radius), (radius, radius), (0, 0))
        descriptor_pad = jnp.pad(descriptor.copy(), pad_width, mode='reflect')  # (H+2*r, W+2*r, D)
        sar_pad = jnp.pad(sar.copy(), pad_width, mode='reflect')  # (H+2*r, W+2*r, 1)

        # Compute start and end indices for memory efficiency
        start_indices, end_indices = compute_indices_from_n_blocks(n_blocks, H, W, padding=radius)

        sar_filtered = jnp.zeros_like(sar_pad)  # Initialize output tensor

        # Iterate over the blocks
        progress_bar = tqdm(total=len(start_indices), desc="Despeckling", unit="block")
        for start_index, end_index in zip(start_indices, end_indices):
            # Extract windows (patches) for all spatial locations
            descriptor_patches = extract_patches(descriptor_pad, kernel_size, start_index, end_index)   # (H', W', k, k, D)
            sar_patches = extract_patches(sar_pad, kernel_size, start_index, end_index)  # (H', W', k, k, 1)

            descriptor_centers = descriptor_patches[..., radius, radius, :][..., None, None, :]  # Centers are located at (radius, radius)

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
        return sar_filtered[radius:-radius, radius:-radius, :]
    
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

    def _step2_filter_texture_descriptor(self, descriptor: jnp.ndarray, opt: jnp.ndarray, sar: jnp.ndarray,
                                         sigma_spatial: float, sigma_luminance_opt: float, sigma_luminance_sar: float,
                                         alpha_update: float, n_iterations: int, n_blocks: int) -> jnp.ndarray:
        """
        Filter the texture descriptor using DUBF.

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
        alpha_update : float
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
        logging.info("Filtering texture descriptor with DUBF...")
        dubf = DUBF()
        time_start = time.time()
        desc_filtered, error = dubf.filter(descriptor=descriptor, 
                                           opt=opt.mean(axis=-1, keepdims=True), 
                                           sar=sar, 
                                           sigma_spatial=sigma_spatial, 
                                           sigma_luminance_opt=sigma_luminance_opt, 
                                           sigma_luminance_sar=sigma_luminance_sar, 
                                           alpha_update=alpha_update, 
                                           n_iterations=n_iterations, 
                                           n_blocks=n_blocks)
        time_end = time.time()
        logging.info(f"Texture descriptor filtering completed in {time_end - time_start:.2f} seconds.")
        self.error = error
        return desc_filtered
    
    def _step3_despeckle(self, sar: jnp.ndarray, descriptor: jnp.ndarray,
                         sigma_distance: float, radius: int, n_blocks: int) -> jnp.ndarray:
        """        
        Despeckle the SAR image using the filtered texture descriptor.

        Parameters
        sar : jnp.ndarray
            Input SAR image of shape (H, W, 1).
        descriptor : jnp.ndarray
            Filtered texture descriptor of shape (H, W, D).
        sigma_distance : float
            Standard deviation for the Gaussian kernel used in similarity computation.
        radius : int
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
                                       radius=radius, 
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
            alpha_update: float = 1.0,
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

        # Filter the texture descriptor using DUBF
        descriptor_filtered = self._step2_filter_texture_descriptor(
            descriptor=descriptor,
            opt=opt,
            sar=sar,
            sigma_spatial=sigma_spatial,
            sigma_luminance_opt=sigma_luminance_opt,
            sigma_luminance_sar=sigma_luminance_sar,
            alpha_update=alpha_update, 
            n_iterations=n_iterations, 
            n_blocks=n_blocks_dubf
        )
        
        # Despeckle the SAR image using the filtered texture descriptor
        sar_filtered = self._step3_despeckle(
            sar=sar, 
            descriptor_filtered=descriptor_filtered, 
            sigma_distance=sigma_distance, 
            radius=radius_despeckling, 
            n_blocks=n_blocks_despeckling
        )
        
        time_end = time.time()
        logging.info(f"Total despeckling process completed in {time_end - time_start:.2f} seconds.")

        return sar_filtered
    
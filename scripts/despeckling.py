import jax
import jax.numpy as jnp
from typing import Tuple, List, Callable
from tqdm import tqdm
from scripts.filters import gaussian_kernel_2d
from scripts.utils import extract_patches, compute_indices_from_n_blocks, distanceL2


class UBF:
    """Unnormalized Bilateral Filter (UBF)."""

    def __init__(self, distance_luminance_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = distanceL2):
        """
        Initialize the UBF class with distance functions for spatial and luminance distances.

        Parameters:
        distance_luminance_fn : Callable
            Function to compute luminance distance between two tensors. Might change based on the domain of the input images.
        """
        assert callable(distance_luminance_fn), "distance_luminance_fn must be callable"
        self.distance_luminance_fn = distance_luminance_fn
        self.distance_luminance_fn = distance_luminance_fn

    def _check_inputs(self, S: jnp.ndarray, L: jnp.ndarray):
        """
        Check the validity of input tensors S and L.
        
        Parameters:
        S : jnp.ndarray
            Input tensor of shape (H, W, D) where D is the number of descriptors, H is height, and W is width.
        L : jnp.ndarray
            Luminance tensor of shape (H, W, 1) used for weighting.
        """
        assert S.ndim == 3, "S should be (H, W, D)"
        assert L.ndim == 3 and L.shape[-1] == 1, "L should be (H, W, 1)"

    def _compute_luminance_weights(self, 
                                  luminance: jnp.ndarray, 
                                  radius: int, 
                                  sigma_l: float, 
                                  start_index: Tuple[int, int], 
                                  end_index: Tuple[int, int]) -> jnp.ndarray:
        """
        Compute the luminance weights.

        Parameters:
        luminance : jnp.ndarray
            The luminance map of shape (H, W, 1).
        radius : int
            The radius of the Gaussian kernel.
        sigma_l : float
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
        weights = jnp.exp(-self.distance_luminance_fn(patches, centers) / (2 * sigma_l ** 2))  # (H', W', k, k, 1)
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
        S: jnp.ndarray,
        L: jnp.ndarray,
        sigma_s: float = 2,
        sigma_l: float = 0.05,
        alpha_ubf: float = 1.0,
        n_iter: int = 1,
        n_blocks: int = 1
    ) -> Tuple[jnp.ndarray, List[float]]:
        """
        Apply UBF.

        ----------
        Parameters
        S : jnp.ndarray
            Input tensor of shape (H, W, D) where D is the number of descriptors, H is height, and W is width.
        L : jnp.ndarray
            Luminance tensor of shape (H, W, 1) used for weighting.
        sigma_s : float
            Spatial standard deviation for Gaussian kernel.
        sigma_l : float
            Luminance standard deviation for Gaussian weighting.
        alpha_ubf : float
            Scaling factor for the update step.
        n_iter : int
            Number of iterations for the filtering process.
        n_blocks : int
            Number of blocks for processing the image in parallel.
        Returns
        jnp.ndarray
            Filtered tensor of shape (H, W, D) after applying the unnormalized bilateral filter.
        """
        self._check_inputs(S, L)

        H, W, _ = S.shape
        radius = int(3 * sigma_s)  # r
        kernel_size = 2 * radius + 1  # k = 2 * r + 1

        # Pad the input arrays to handle borders
        pad_width = ((radius, radius), (radius, radius), (0, 0))
        S0 = jnp.pad(S.copy(), pad_width, mode='reflect')  # (H+2*r, W+2*r, D)
        L0 = jnp.pad(L.copy(), pad_width, mode='reflect')  # (H+2*r, W+2*r, 1)

        # Precompute Gaussian weights
        Gs = self._compute_gaussian_weights(radius).reshape(1, 1, kernel_size, kernel_size, 1)  # (1, 1, k, k, 1)

        # Compute start and end indices for memory efficiency
        start_indices, end_indices = compute_indices_from_n_blocks(n_blocks, H, W, padding=radius)

        error = []
        for _ in tqdm(range(n_iter), desc="UBF Iterations"):
            update = jnp.zeros_like(S0)  # Initialize update tensor

            # Iterate over the blocks
            for start_index, end_index in zip(start_indices, end_indices):
                # Extract windows (patches) for all spatial locations
                S_patches = extract_patches(S0, kernel_size, start_index, end_index)   # (H', W', k, k, D)
                Gl = self._compute_luminance_weights(L0, radius, sigma_l, start_index, end_index)   # (H', W', k, k)

                # Combined weight
                w = Gs * Gl  # (H', W', k, k)

                # Difference from center/target pixel
                S_centers = S_patches[..., radius, radius, :]  # Centers are located at (radius, radius)
                diff = S_patches - S_centers[..., None, None, :]  # (H', W', k, k, D), 

                # Weighted sum over window
                update_block = (w * diff).sum(axis=(-3, -2))  # (H', W', D)

                # Update only the current block
                update = jax.lax.dynamic_update_slice(update, update_block, start_index + (0,))  

            # Update
            S1 = S0 + alpha_ubf * update  # (H+2*r, W+2*r, D)

            error.append(float(jnp.linalg.norm(S1 - S0)))
            S0 = S1

        # Unpad to get output of shape (H, W, D)
        S_filtered = S0[radius:H+radius, radius:W+radius, :]  # (H, W, D)
        return S_filtered, error


class DUBF(UBF):
    """Dual Unnormalized Bilateral Filter (DUBF)."""

    def __init__(self, distance_luminance_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = distanceL2):
        """
        Initialize the DUBF class with distance functions for spatial and luminance distances.

        Parameters:
        distance_luminance_fn : Callable
            Function to compute luminance distance between two tensors. Might change based on the domain of the input images.
        """
        super().__init__(distance_luminance_fn)

    def _check_inputs(self, S: jnp.ndarray, L_opt: jnp.ndarray, L_sar: jnp.ndarray):
        """
        Check the validity of input tensors S, L_opt, and L_sar.
        
        Parameters:
        S : jnp.ndarray
            Input tensor of shape (H, W, D) where D is the number of descriptors, H is height, and W is width.
        L_opt : jnp.ndarray
            Optical luminance tensor of shape (H, W, 1) used for weighting.
        L_sar : jnp.ndarray
            SAR luminance tensor of shape (H, W, 1) used for weighting.
        """
        assert S.ndim == 3, "S should be (H, W, D)"
        assert L_opt.ndim == 3 and L_opt.shape[-1] == 1, 'L_opt should be (H, W, 1)'
        assert L_sar.ndim == 3 and L_sar.shape[-1] == 1, 'L_sar should be (H, W, 1)'

    def filter(
        self,
        S: jnp.ndarray,
        L_opt: jnp.ndarray,
        L_sar: jnp.ndarray,
        sigma_s: float = 2,
        sigma_l_opt: float = 0.05,
        sigma_l_sar: float = 0.05,
        alpha_ubf: float = 1.0,
        n_iter: int = 1,
        n_blocks: int = 1
    ) -> Tuple[jnp.ndarray, List[float]]:
        """
        Apply DUBF.

        ----------
        Parameters
        S : jnp.ndarray
            Input tensor of shape (H, W, D) where D is the number of descriptors, H is height, and W is width.
        L_opt : jnp.ndarray
            Optical luminance tensor of shape (H, W, 1) used for weighting.
        L_sar : jnp.ndarray
            SAR luminance tensor of shape (H, W, 1) used for weighting.
        sigma_s : float
            Spatial standard deviation for Gaussian kernel.
        sigma_l_opt : float
            Optical luminance standard deviation for Gaussian weighting.
        sigma_l_sar : float
            SAR luminance standard deviation for Gaussian weighting.
        alpha_ubf : float
            Scaling factor for the update step.
        n_iter : int
            Number of iterations for the filtering process.
        n_blocks : int
            Number of blocks for processing the image in parallel.
        Returns
        jnp.ndarray
            Filtered tensor of shape (H, W, D) after applying the unnormalized bilateral filter.
        """
        self._check_inputs(S, L_opt, L_sar)

        H, W, _ = S.shape
        radius = int(3 * sigma_s)  # r
        kernel_size = 2 * radius + 1  # k = 2 * r + 1

        # Pad the input arrays to handle borders
        pad_width = ((radius, radius), (radius, radius), (0, 0))
        S0 = jnp.pad(S.copy(), pad_width, mode='reflect')  # (H+2*r, W+2*r, D)
        L0_opt = jnp.pad(L_opt.copy(), pad_width, mode='reflect')  # (H+2*r, W+2*r, 1)
        L0_sar = jnp.pad(L_sar.copy(), pad_width, mode='reflect')  # (H+2*r, W+2*r, 1)

        # Precompute Gaussian weights
        Gs = self._compute_gaussian_weights(radius).reshape(1, 1, kernel_size, kernel_size, 1)  # (1, 1, k, k, 1)

        # Compute start and end indices for memory efficiency
        start_indices, end_indices = compute_indices_from_n_blocks(n_blocks, H, W, padding=radius)

        error = []
        for _ in tqdm(range(n_iter), desc="DUBF Iterations"):
            update = jnp.zeros_like(S0)  # Initialize update tensor

            # Iterate over the blocks
            for start_index, end_index in zip(start_indices, end_indices):
                # Extract windows (patches) for all spatial locations
                S_patches = extract_patches(S0, kernel_size, start_index, end_index)   # (H', W', k, k, D)
                Gl_opt = self._compute_luminance_weights(L0_opt, radius, sigma_l_opt, start_index, end_index)   # (H', W', k, k)
                Gl_sar = self._compute_luminance_weights(L0_sar, radius, sigma_l_sar, start_index, end_index)   # (H', W', k, k)

                # Combined weight
                w = Gs * Gl_opt * Gl_sar  # (H, W, k, k, 1)

                # Difference from center/target pixel
                S_centers = S_patches[..., radius, radius, :]  # Centers are located at (radius, radius)
                diff = S_patches - S_centers[..., None, None, :]  # (H', W', k, k, D)

                # Weighted sum over window
                update_block = (w * diff).sum(axis=(-3, -2))

                # Update only the current block
                update = jax.lax.dynamic_update_slice(update, update_block, start_index + (0,))  

            # Update
            S1 = S0 + alpha_ubf * update

            error.append(float(jnp.linalg.norm(S1 - S0)))
            S0 = S1

        # Unpad to get output of shape (H, W, D)
        S_filtered = S0[radius:H+radius, radius:W+radius, :]
        return S_filtered, error


def despeckle():
    return NotImplementedError("Despeckling function is not implemented yet.")

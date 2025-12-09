import jax
import jax.numpy as jnp
from typing import Tuple, List
from tqdm import tqdm
import logging
from scripts.filters import gaussian_kernel_2d
from scripts.parallelisation import extract_patches, compute_indices_from_n_blocks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MUBF:
    """Multimodal Unnormalized Bilateral Filter."""

    def _check_inputs(self, target: jnp.ndarray, guides: List[jnp.ndarray]) -> None:
        """
        Check the size of input tensors.

        Parameters:
        Target : jnp.ndarray
            Input tensor of shape (H, W, D) where D is the number of descriptors, H is height, and W is width.
        Guides : List[jnp.ndarray]
            List of guide tensors, each of shape (H, W, 1).
        """
        assert target.ndim == 3, "Target should be (H, W, D)"
        for guide in guides:
            assert guide.ndim == 3 and guide.shape[-1] == 1, "Guides should be (H, W, 1)"
    
    def _initialize_inputs(self, target: jnp.ndarray, guides: List[jnp.ndarray], radius: int) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
        """
        Pad the input tensors.

        Parameters:
        target : jnp.ndarray
            Input tensor of shape (H, W, D).
        guides : List[jnp.ndarray]
            List of guide tensors, each of shape (H, W, 1).
        radius : int
            The radius for padding the input tensors.
        Returns:
        Tuple[jnp.ndarray, List[jnp.ndarray]]
            Padded target tensor of shape (H+2*r, W+2*r, D) and list of padded guide tensors, each of shape (H+2*r, W+2*r, 1).
        """
        pad_width = ((radius, radius), (radius, radius), (0, 0))
        target_padded = jnp.pad(target.copy(), pad_width, mode='reflect')  # (H+2*r, W+2*r, D)
        guides_padded = [jnp.pad(guide.copy(), pad_width, mode='reflect') for guide in guides]  # (H+2*r, W+2*r, 1)
        return target_padded, guides_padded

    def _compute_gaussian_weights(self, radius: int) -> jnp.ndarray:
        """
        Compute the Gaussian weights for the spatial domain.

        Parameters:
        radius : int
            The radius of the Gaussian kernel.
        Returns:
        jnp.ndarray
            The Gaussian weights of shape (1, 1, k, k, 1) where k = 2 * r + 1.
        """
        kernel_size = 2 * radius + 1  # k = 2 * r + 1
        return gaussian_kernel_2d(radius).reshape(1, 1, kernel_size, kernel_size, 1)  # (1, 1, k, k, 1)

    def _compute_guide_weights(self, 
                               guide: jnp.ndarray, 
                               sigma: float, 
                               radius: int, 
                               start_index: Tuple[int, int], 
                               end_index: Tuple[int, int]) -> jnp.ndarray:
        """
        Compute the luminance weights.

        Parameters:
        guide : jnp.ndarray
            The guide map of shape (H, W, 1).
        radius : int
            The radius of the Gaussian kernel.
        sigma : float
            The standard deviation of the guide Gaussian kernel.
        start_index : Tuple[int, int]
            The starting index for the patch extraction.
        end_index : Tuple[int, int]
            The ending index for the patch extraction.

        Returns:
        jnp.ndarray
            The computed guide weights.
        """
        kernel_size = 2 * radius + 1  # k = 2 * r + 1
        patches = extract_patches(guide, kernel_size, start_index, end_index)   # (H', W', k, k, 1)
        centers = patches[..., radius, radius, :]  # (H', W', 1)
        centers = centers[..., None, None, :] # (H', W', 1, 1, 1)
        weights = jnp.exp(-(patches - centers)**2 / (2 * sigma ** 2))  # (H', W', k, k, 1)
        return weights
    
    def _compute_weights(self, 
                         guides: List[jnp.ndarray], 
                         sigmas: List[float],
                         gammas: List[float],
                         radius: int, 
                         gaussian_weights: jnp.ndarray,
                         start_index: Tuple[int, int], 
                         end_index: Tuple[int, int]) -> jnp.ndarray:
        """ 
        Compute the Gaussian weights for the guides.

        Parameters:
        guides : List[jnp.ndarray]
            List of guide tensors, each of shape (H, W, 1).
        sigmas : List[float]
            List of standard deviations for the Gaussian weights corresponding to each guide.
        gammas : List[float]
            List of scaling factors for each guide.
        radius : int
            The radius of the Gaussian kernel.
        gaussian_weights : jnp.ndarray
            Precomputed Gaussian weights of shape (1, 1, k, k, 1).
        start_index : Tuple[int, int]
            The starting index for the patch extraction.
        end_index : Tuple[int, int]
            The ending index for the patch extraction.
        Returns:
        jnp.ndarray
            The computed weights of shape (H', W', k, k) where H' and W' are the dimensions of the patches.
        """
        weights = gaussian_weights.copy()
        for guide, sigma, gamma in zip(guides, sigmas, gammas):
            weights *= gamma * self._compute_guide_weights(guide, sigma, radius, start_index, end_index)
        return weights
    
    def _update(self, target: jnp.ndarray, update: jnp.ndarray, alpha: float) -> Tuple[jnp.ndarray, float]:
        target_updated = target + alpha * update  # (H+2*r, W+2*r, D)
        error = float(jnp.linalg.norm(target_updated - target))

        return target_updated, error
    
    def _loop_over_blocks(self, 
                          start_index: Tuple[int, int], 
                          end_index: Tuple[int, int], 
                          target: jnp.ndarray,
                          update: jnp.ndarray,
                          guides: List[jnp.ndarray],
                          sigmas: List[float],
                          gammas: List[float],
                          gaussian_weights: jnp.ndarray,
                          kernel_size: int) -> jnp.ndarray:
        """
        Loop over blocks of the target.

        Parameters:
        start_index : Tuple[int, int]
            Starting index for the block.
        end_index : Tuple[int, int]
            Ending index for the block.
        target : jnp.ndarray
            Target tensor of shape (H, W, D) to be updated.
        update : jnp.ndarray
            Update tensor of shape (H, W, D) to accumulate updates.
        guides : List[jnp.ndarray]
            List of guide tensors, each of shape (H, W, 1).
        sigmas : List[float]
            List of standard deviations for the Gaussian weights corresponding to each guide.
        gammas : List[float]
            List of scaling factors for each guide.
        gaussian_weights : jnp.ndarray
            Precomputed Gaussian weights of shape (1, 1, k, k, 1).
        kernel_size : int
            Size of the Gaussian kernel (k = 2 * r + 1).
        """
        # Extract windows (patches) for all spatial locations
        radius = kernel_size // 2
        patches = extract_patches(target, kernel_size, start_index, end_index)   # (H', W', k, k, D)
        weights = self._compute_weights(guides, sigmas, gammas, radius, gaussian_weights, start_index, end_index)   # (H', W', k, k)

        # Difference from center/target pixel
        centers = patches[..., radius, radius, :]  # Centers are located at (radius, radius)
        difference = patches - centers[..., None, None, :]  # (H', W', k, k, D)

        # Weighted sum over window
        update_block = (weights * difference).sum(axis=(-3, -2))  # (H', W', D)

        # Update only the current block
        update = jax.lax.dynamic_update_slice(update, update_block, start_index + (0,))  

        return update

    def filter(
        self,
        target: jnp.ndarray,
        guides: List[jnp.ndarray],
        sigma_spatial: float = 5,
        sigma_guides: List[float] = [0.05, 0.05],
        gamma_guides: List[float] = [1.0, 1.0],
        alpha: float = 1.0,
        n_iterations: int = 30,
        n_blocks: int = 10
    ) -> Tuple[jnp.ndarray, List[float]]:
        """
        Apply filter.

        ----------
        Parameters
        target : jnp.ndarray
            Input tensor of shape (H, W, D) where D is the number of descriptors, H is height, and W is width.
        guides : List[jnp.ndarray]
            List of guide tensors, each of shape (H, W, 1).
        sigma_spatial : float
            Spatial standard deviation for Gaussian kernel.
        sigma_guides : List[float]
            List of standard deviations for Gaussian weights corresponding to each guide.
        gamma_guides : List[float]
            List of scaling factors for each guide.
        alpha : float
            Scaling factor for the update step.
        n_iterations : int
            Number of iterations for the filtering process.
        n_blocks : int
            Number of blocks for processing the image in parallel.
        Returns
        jnp.ndarray
            Filtered tensor of shape (H, W, D) after applying the unnormalized bilateral filter.
        List[float]
            List of errors for each iteration.
        """
        self._check_inputs(target, guides)

        H, W, _ = target.shape
        radius = int(3 * sigma_spatial)  # r
        kernel_size = 2 * radius + 1  # k = 2 * r + 1

        target, guides = self._initialize_inputs(target, guides, radius)

        # Precompute Gaussian weights
        gaussian_weights = self._compute_gaussian_weights(radius)  # (1, 1, k, k, 1)

        # Compute start and end indices for memory efficiency
        start_indices, end_indices = compute_indices_from_n_blocks(n_blocks, H, W, padding=radius)

        errors = []
        for _ in tqdm(range(n_iterations), desc="Iterations"):
            update = jnp.zeros_like(target)  # Initialize update tensor

            # Iterate over the blocks
            for start_index, end_index in zip(start_indices, end_indices):
                update = self._loop_over_blocks(
                    start_index=start_index,
                    end_index=end_index,
                    target=target,
                    update=update,
                    guides=guides,
                    sigmas=sigma_guides,
                    gammas=gamma_guides,
                    gaussian_weights=gaussian_weights,
                    kernel_size=kernel_size
                )

            # Update
            target, error = self._update(target=target, update=update, alpha=alpha)
            errors.append(error)

        # Unpad to get output of shape (H, W, D)
        target = target[radius:-radius, radius:-radius, :]  # (H, W, D)
        return target, errors
    
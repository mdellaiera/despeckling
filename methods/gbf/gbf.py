import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Tuple
from scripts.mubf import MUBF
from scripts.enl import ENLClassifier
from methods.sarbm3d.sarbm3d import SARBM3D
from scripts.parallelisation import extract_patches


class BF(MUBF):
    """Bilateral Filter (BF)."""

    # def _compute_luminance_weights(self, 
    #                                 luminance: jnp.ndarray, 
    #                                 radius: int, 
    #                                 lmbda: float,
    #                                 euclidian_distance: bool = True) -> jnp.ndarray:
    #     """
    #     Compute the luminance weights.

    #     Parameters:
    #     luminance : jnp.ndarray
    #         The luminance map of shape (H, W).
    #     radius : int
    #         The radius of the Gaussian kernel.
    #     lmbda : float
    #         The weight for the luminance kernel.
    #     euclidian_distance : bool
    #         If True, use Euclidean distance for luminance weighting. Else, use the SAR-domain range distance.

    #     Returns:
    #     jnp.ndarray
    #         The computed luminance weights of shape (H, W, k, k).
    #     """
    #     eps = 1e-10
    #     kernel_size = 2 * radius + 1  # k = 2 * r + 1
    #     patches = extract_patches(luminance, kernel_size)   # (H, W, k, k)
    #     centers = patches[..., radius, radius]  # (H, W)
    #     centers = centers[..., None, None] # (H, W, 1, 1)

    #     if euclidian_distance:
    #         distance = (patches - centers)**2  # (H, W, k, k)
    #     else:
    #         distance = jnp.log2(patches / (centers + eps) - centers / (patches + eps))  # (H, W, k, k)

    #     weights = jnp.exp(-lmbda * distance)  # (H, W, k, k)
    #     return weights

    # def _compute_gaussian_weights(self, radius: int, lmbda: float) -> jnp.ndarray:
    #     """
    #     Compute the normalized Gaussian weights for the spatial domain.

    #     Parameters:
    #     radius : int
    #         The radius of the Gaussian kernel.
    #     lmbda : float
    #         The weight for the Gaussian kernel.

    #     Returns:
    #     jnp.ndarray
    #         The normalized Gaussian weights of shape (k, k).
    #     """
    #     x = jnp.arange(-radius, radius + 1)
    #     kernel = jnp.exp(-lmbda * (x * x))
    #     kernel = kernel / sum(kernel)
    #     kernel = kernel.reshape(-1, 1)
    #     return kernel @ kernel.T

    def lmbda2sigma(self, lmbda: float) -> float:
        return jnp.sqrt(0.5 / lmbda)
    
    def lmbda2radius(self, lmbda: float) -> int:
        return int(jnp.ceil(3 * self.lmbda2sigma(lmbda)))
    
    def _update(self, target: jnp.ndarray, update: jnp.ndarray, alpha: float) -> Tuple[jnp.ndarray, float]:
        error = float(jnp.linalg.norm(update - target))
        return update, error  # No update in BF, just return the filtered result.

    def _loop_over_blocks(
            self, 
            start_index: Tuple[int, int], 
            end_index: Tuple[int, int], 
            target: jnp.ndarray,
            update: jnp.ndarray,
            guides: List[jnp.ndarray],
            sigmas: List[float],
            gammas: List[float],
            gaussian_weights: jnp.ndarray,
            kernel_size: int,
            euclidian_distance: bool) -> jnp.ndarray:
        radius = kernel_size // 2
        patches = extract_patches(target, kernel_size, start_index, end_index)

        w_S = gaussian_weights
        w_RO = self._compute_guide_weights(guides[0], sigmas[0], radius, start_index, end_index, True)
        w_RS = self._compute_guide_weights(guides[1], sigmas[1], radius, start_index, end_index, euclidian_distance) 

        # Combine weights
        w = w_S * w_RO * w_RS  # (H', W', k, k), corresponds to w(s, t) in Eq. 1, computed as Eq. 3
        w = w / (w.sum(axis=(-2, -1), keepdims=True) + 1e-10)

        # Filter
        # sar_filtered = (w * patches).sum(axis=(-2, -1))  # (H', W')

        # Weighted sum over window
        update_block = (w * patches).sum(axis=(-3, -2))  # (H', W', D)

        # Update only the current block
        update = jax.lax.dynamic_update_slice(update, update_block, start_index + (0,))  

        return update
    

class GBF:
    """
    From the article Optical-Driven Nonlocal SAR Despeckling by Verdoliva et al. published in 
    IEEE GEOSCIENCE AND REMOTE SENSING LETTERS, VOL. 12, NO. 2, FEBRUARY 2015
    """

    def __init__(self):
        self.uO = None  # Output from GBF
        self.uS = None  # Output from SARBM3D
        self.ft = None  # Soft classification weight
        self.at: List = None 

    def filter(self, 
               sar: jnp.ndarray, 
               eo: jnp.ndarray, 
               matlab_script_path: str,
               L: int = 1,
            #    window_size: int = 31, 
               lambda_S: float = 0.005, 
               lambda_RO: float = 0.02, 
               lambda_RS: float = 0.1,
               N: List[int] = np.arange(7, 63, 2),
               gamma: float = 7,
               a0: float = 0.64,
               n_blocks: int = 10,
               euclidian_distance: bool = True) -> jnp.ndarray:
        input_eo = jnp.array(eo, dtype=jnp.float32)
        input_sar = jnp.array(sar, dtype=jnp.float32)

        gbf = BF()
        sarbm3d = SARBM3D(matlab_script_path=matlab_script_path)
        classifier = ENLClassifier()

        sigma_spatial = gbf.lmbda2sigma(lambda_S)
        sigma_guides = [gbf.lmbda2sigma(lambda_RO), gbf.lmbda2sigma(lambda_RS)]

        self.uO = gbf.filter(
            input_sar, 
            input_eo, 
            sigma_spatial=sigma_spatial, 
            sigma_guides=sigma_guides, 
            n_iterations=1, 
            n_blocks=n_blocks, 
            euclidian_distance=euclidian_distance
        )
        self.uS = sarbm3d.filter(input_sar, L=L)
        self.ft = classifier.classify(input_sar, N, gamma, a0)
        self.at = classifier.at

        return self.ft * self.uO + (1 - self.ft) * self.uS
    
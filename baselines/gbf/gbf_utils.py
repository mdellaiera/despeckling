import os
import numpy as np
import jax
import jax.numpy as jnp
from typing import List
from tqdm import tqdm
import matlab.engine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_patches(tensor: jnp.ndarray, kernel_size: int) -> jnp.ndarray:
        """
        Extracts sliding patches from a 2D tensor.

        Parameters:
        tensor: 
            Input tensor of size (H, W)
        kernel_size: int
            Size of the square kernel (must be odd).
        Returns: 
        jnp.ndarray
            A tensor of size (H, W, kernel_size, kernel_size) containing the extracted patches.
        """
        assert kernel_size % 2 == 1, "Kernel size must be odd."
        assert tensor.ndim < 3, "Input tensor must be 2D."

        H, W = tensor.shape  
        pad = kernel_size // 2  
        tensor = jnp.pad(tensor, ((pad, pad), (pad, pad)), mode='reflect')

        # Create indices for the sliding window
        h_idx = jnp.arange(0, H)
        w_idx = jnp.arange(0, W)

        # Create a function to extract patches for each (i, j) pair
        def get_patch(i: int, j: int) -> jnp.ndarray:
            patch = jax.lax.dynamic_slice(operand=tensor, 
                                        start_indices=(i, j), 
                                        slice_sizes=(kernel_size, kernel_size))
            return patch

        # Use vmap to vectorize the extraction of patches across the height and width indices
        patches = jax.vmap(
            lambda i: jax.vmap(lambda j: get_patch(i, j))(w_idx)
        )(h_idx)
        return patches 


class GBF:
    """Guided Bilateral Filter (GBF)."""

    def _compute_luminance_weights(self, 
                                    luminance: jnp.ndarray, 
                                    radius: int, 
                                    lmbda: float,
                                    euclidian_distance: bool = True) -> jnp.ndarray:
        """
        Compute the luminance weights.

        Parameters:
        luminance : jnp.ndarray
            The luminance map of shape (H, W).
        radius : int
            The radius of the Gaussian kernel.
        lmbda : float
            The weight for the luminance kernel.
        euclidian_distance : bool
            If True, use Euclidean distance for luminance weighting. Else, use the SAR-domain range distance.

        Returns:
        jnp.ndarray
            The computed luminance weights of shape (H, W, k, k).
        """
        eps = 1e-10
        kernel_size = 2 * radius + 1  # k = 2 * r + 1
        patches = extract_patches(luminance, kernel_size)   # (H, W, k, k)
        centers = patches[..., radius, radius]  # (H, W)
        centers = centers[..., None, None] # (H, W, 1, 1)

        if euclidian_distance:
            distance = (patches - centers)**2  # (H, W, k, k)
        else:
            distance = jnp.log2(patches / (centers + eps) - centers / (patches + eps))  # (H, W, k, k)

        weights = jnp.exp(-lmbda * distance)  # (H, W, k, k)
        return weights

    def _compute_gaussian_weights(self, radius: int, lmbda: float) -> jnp.ndarray:
        """
        Compute the normalized Gaussian weights for the spatial domain.

        Parameters:
        radius : int
            The radius of the Gaussian kernel.
        lmbda : float
            The weight for the Gaussian kernel.

        Returns:
        jnp.ndarray
            The normalized Gaussian weights of shape (k, k).
        """
        x = jnp.arange(-radius, radius + 1)
        kernel = jnp.exp(-lmbda * (x * x))
        kernel = kernel / sum(kernel)
        kernel = kernel.reshape(-1, 1)
        return kernel @ kernel.T

    def filter(self, 
               sar: jnp.ndarray, 
               opt: jnp.ndarray, 
               window_size: int = 31, 
               lambda_S: float = 0.005, 
               lambda_RO: float = 0.02, 
               lambda_RS: float = 0.1) -> jnp.ndarray:
        """
        Apply filter.

        Parameters
        sar : jnp.ndarray
            Input tensor of shape (H, W).
        opt : jnp.ndarray
            Luminance tensor of shape (H, W).
        window_size : int
            Size of the square kernel (must be odd).
        lambda_S : float
            Weight for spatial Gaussian kernel.
        lambda_RO : float
            Weight for optical luminance kernel.
        lambda_RS : float
            Weight for SAR luminance kernel.

        Returns
        jnp.ndarray
            Filtered tensor of shape (H, W).
        """
        radius = int(window_size // 2)  # r
        H, W = sar.shape

        # Pad the input arrays to handle borders
        pad_width = ((radius, radius), (radius, radius))
        sar_padded = jnp.pad(sar.copy(), pad_width, mode='reflect')  # (H', W'), with H' = H + 2*r and W' = W + 2*r
        opt_padded = jnp.pad(opt.copy(), pad_width, mode='reflect')  # (H', W')

        # Extract patches from the SAR image
        sar_patches = extract_patches(sar_padded, window_size)  # (H', W', k, k), corresponds to v(s) in Eq. 1

        # Precompute Gaussian weights
        w_S = lambda_S * self._compute_gaussian_weights(radius, lambda_S).reshape(1, 1, window_size, window_size)  # (1, 1, k, k)
        w_RO = lambda_RO * self._compute_luminance_weights(opt_padded, radius, lambda_RO, True)   # (H', W', k, k)
        w_RS = lambda_RS * self._compute_luminance_weights(sar_padded, radius, lambda_RS, True)  # (H', W', k, k)

        # Combine weights
        w = w_S * w_RO * w_RS  # (H', W', k, k), corresponds to w(s, t) in Eq. 1, computed as Eq. 3
        w = w / (w.sum(axis=(-2, -1), keepdims=True) + 1e-10)

        # Filter
        sar_filtered = (w * sar_patches).sum(axis=(-2, -1))  # (H', W')

        # Unpad to get output of shape (H, W)
        return sar_filtered[radius:H+radius, radius:W+radius]  # (H, W)
    

class SARBM3D:
    """Wrapper around SARBM3D filter."""

    def __init__(self, matlab_script_path: str):
        self.matlab_script_path = matlab_script_path
        self._set_library_path()

    def _set_library_path(self) -> None:
        os.environ['LD_LIBRARY_PATH'] = os.path.join(
            os.path.dirname(os.path.abspath(self.matlab_script_path)), "lib_opencv210/glnxa64"
        ) + ":" + os.environ.get('LD_LIBRARY_PATH', '')

    def _preprocess_input(self, sar: jnp.ndarray) -> jnp.ndarray:
        return np.log1p(np.abs(sar**2))

    def filter(self, sar: jnp.ndarray, L: int) -> jnp.ndarray:
        input = self._preprocess_input(sar)
        
        eng = matlab.engine.start_matlab()
        eng.addpath(os.path.dirname(self.matlab_script_path), nargout=0)
        eng.eval("cd('{}')".format(os.path.dirname(self.matlab_script_path)), nargout=0)
        try:
            y = eng.SARBM3D_v10(matlab.double(input.tolist()), matlab.double(L), nargout=2)
            output = jnp.array(y[0]).reshape(input.shape)
            eng.quit()
        except Exception as e:
            eng.quit()
            raise RuntimeError(f"Error during execution: {e}")
        return output
    

class SoftClassifier:
    """Soft classifier to linearly combine multiple filtered estimates."""

    def __init__(self):
        pass

    def _compute_enl(self, sar: jnp.ndarray, window_size: int) -> float:
        """
        Compute the Equivalent Number of Looks (ENL) for a given SAR image.

        Parameters:
        sar : jnp.ndarray
            Input image of shape (H, W).
        window_size : int
            Size of the window for local statistics.

        Returns:
        float
            The ENL value.
        """
        radius = window_size // 2
        sar_padded = jnp.pad(sar, ((radius, radius), (radius, radius)), mode='reflect')  # (H', W')

        # Extract patches
        patches = extract_patches(sar_padded, window_size)  # (H', W', k, k)

        # Compute local mean and variance
        local_mean = patches.mean(axis=(-2, -1))  # (H', W')
        local_var = patches.var(axis=(-2, -1))  # (H', W')

        # Compute ENL
        enl = local_mean**2 / (local_var + 1e-10)  # Avoid division by zero
        enl = enl[radius:-radius, radius:-radius]  # Unpad to get output of shape (H, W)
        return enl  # Return the average ENL over the image
    
    def compute_weight(self, sar: jnp.ndarray, N: List[int], gamma: float = 7, a0: float = 0.64) -> jnp.ndarray:
        """
        Compute the soft classification weight for SAR despeckling.

        Parameters:
        sar : jnp.ndarray
            Input SAR image of shape (H, W).
        N : List[int]
            List of equivalent look numbers for each class.
        gamma : float
            Parameter for the soft classification weight.
        a0 : float
            Parameter for the soft classification weight.

        Returns:
        jnp.ndarray
            The soft classification weight of shape (H, W).
        """
        at = jnp.zeros_like(sar)
        for n in tqdm(N):
            at += self._compute_enl(sar, window_size=n)
        at = at / len(N)

        ft = jnp.exp(-gamma * (at - a0)) + 1
        ft = 1 / ft
        return ft
    

class SARDespeckling:
    """
    From the article Optical-Driven Nonlocal SAR Despeckling by Verdoliva et al. published in 
    IEEE GEOSCIENCE AND REMOTE SENSING LETTERS, VOL. 12, NO. 2, FEBRUARY 2015
    """

    def __init__(self):
        self.uO = None  # Output from ODUBF
        self.uS = None  # Output from SARBM3D
        self.ft = None  # Soft classification weight

    def filter(self, 
               sar: jnp.ndarray, 
               opt: jnp.ndarray, 
               matlab_script_path: str,
               L: int = 1,
               window_size: int = 31, 
               lambda_S: float = 0.005, 
               lambda_RO: float = 0.02, 
               lambda_RS: float = 0.1,
               N: List[int] = np.arange(7, 63, 2),
               gamma: float = 7,
               a0: float = 0.64) -> jnp.ndarray:
        gbf = GBF()
        sarbm3d = SARBM3D(matlab_script_path=matlab_script_path)
        soft_classifier = SoftClassifier()

        logging.info("Applying GBF...")
        self.uO = gbf.filter(sar, opt, window_size, lambda_S, lambda_RO, lambda_RS)
        logging.info("Applying SARBM3D...")
        self.uS = sarbm3d.filter(sar, L=L)
        logging.info("Computing soft classification weight...")
        self.ft = soft_classifier.compute_weight(sar, N, gamma, a0)

        return self.ft * self.uO + (1 - self.ft) * self.uS
    
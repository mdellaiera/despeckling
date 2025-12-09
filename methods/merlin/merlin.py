import numpy as np
import jax.numpy as jnp
import sys
from typing import Union
from scripts.utils import BaseFilter


class MERLIN(BaseFilter):
    """Wrapper around MERLIN filter."""

    def __init__(self, project_path: str):
        self.project_path = project_path

    def filter(self, sar: Union[np.ndarray, jnp.ndarray]) -> Union[np.ndarray, jnp.ndarray]:
        sys.path.insert(0, self.project_path)
        from deepdespeckling.utils.constants import PATCH_SIZE, STRIDE_SIZE
        from deepdespeckling.merlin.merlin_denoiser import MerlinDenoiser

        input = np.stack((np.real(sar), np.imag(sar)), axis=-1)
        denoiser = MerlinDenoiser(model_name='spotlight', symetrise=False)
        output = denoiser.denoise_image(input, patch_size=PATCH_SIZE, stride_size=STRIDE_SIZE)['denoised']['full']

        if isinstance(sar, jnp.ndarray):
            return jnp.array(output)
        return output
    
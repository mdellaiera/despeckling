import numpy as np
import jax.numpy as jnp
import torch
import sys
from typing import Union
from scripts.utils import BaseFilter


class SAR2SAR(BaseFilter):
    """Wrapper around SAR2SAR filter."""

    def __init__(self, project_path: str):
        self.project_path = project_path

    def filter(self, sar: Union[np.ndarray, jnp.ndarray]) -> Union[np.ndarray, jnp.ndarray]:
        sys.path.insert(0, self.project_path)
        from deepdespeckling.utils.constants import PATCH_SIZE, STRIDE_SIZE
        from deepdespeckling.sar2sar.sar2sar_denoiser import Sar2SarDenoiser
        from deepdespeckling.model import Model

        class Sar2SarDenoiserFixed(Sar2SarDenoiser):
            """Class to share parameters beyond denoising functions
            """

            def __init__(self, **params):
                super().__init__(**params)

            def load_model(self, patch_size: int) -> Model:
                """Load model with given weights 

                Args:
                    weights_path (str): path to weights  
                    patch_size (int): patch size

                Returns:
                    model (Model): model loaded with stored weights
                """
                model = Model(torch.device(self.device),
                            height=patch_size, width=patch_size)
                model.load_state_dict(torch.load(
                    self.weights_path, map_location=torch.device("cpu"), weights_only=False)['model_state_dict'])

                return model

        input = np.sqrt(np.real(sar)**2 + np.imag(sar)**2)
        denoiser = Sar2SarDenoiserFixed()
        output = denoiser.denoise_image(input, patch_size=PATCH_SIZE, stride_size=STRIDE_SIZE)['denoised']

        if isinstance(sar, jnp.ndarray):
            return jnp.array(output)
        return output
    
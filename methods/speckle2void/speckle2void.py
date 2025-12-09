import os
import numpy as np
import jax.numpy as jnp
import tensorflow as tf
import sys
from typing import Union
from scripts.utils import BaseFilter


class Speckle2Void(BaseFilter):
    """Wrapper around Speckle2Void filter."""

    def __init__(self, checkpoint_path: str, libraries_path: str):
        self.checkpoint_path = checkpoint_path
        self.libraries_path = libraries_path

    def filter(self, sar: Union[np.ndarray, jnp.ndarray], norm: float) -> Union[np.ndarray, jnp.ndarray]:
        sys.path.insert(0, os.path.dirname(self.libraries_path))
        sys.path.insert(0, self.libraries_path)
        from Speckle2Void import Speckle2V

        tf.reset_default_graph()

        input = np.abs(sar).reshape((1, sar.shape[0], sar.shape[1], 1))

        model = Speckle2V(
            "./",
            "./",
            self.checkpoint_path,
            batch_size=1,
            patch_size=64,
            model_name='speckle2void',
            lr=1e-04, 
            steps_per_epoch=2000,
            k_penalty_tv=5e-05,
            shift_list=[3,1],
            prob = [0.9,0.1],
            clip=500000,
            norm=norm,
            L_noise=1)    
        
        model.build_inference()
        model.load_weights()

        output = model.predict(input)
        output = output.squeeze()
        
        if isinstance(sar, jnp.ndarray):
            return jnp.array(output)
        return output
    
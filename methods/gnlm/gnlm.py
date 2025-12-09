import os
import numpy as np
import jax.numpy as jnp
import matlab.engine
from typing import Union
from scripts.utils import BaseFilter, c2ap, ap2c


class GNLM(BaseFilter):
    """Wrapper around GNLM filter."""

    def __init__(self, matlab_script_path: str):
        self.matlab_script_path = matlab_script_path
        self._set_library_path()

    def _set_library_path(self) -> None:
        os.environ['LD_LIBRARY_PATH'] = os.path.join(
            os.path.dirname(os.path.abspath(self.matlab_script_path)), "lib_opencv210/glnxa64"
        ) + ":" + os.environ.get('LD_LIBRARY_PATH', '')

    def filter(
            self, 
            sar: Union[np.ndarray, jnp.ndarray], 
            eo: Union[np.ndarray, jnp.ndarray],
            L: int,
            stack_size: int,
            sharpness: float,
            balance: float,
            th_sar: float,
            block_size: int,
            win_size: int,
            stride: int) -> Union[np.ndarray, jnp.ndarray]:
        amplitude, phase = c2ap(sar)
        
        eng = matlab.engine.start_matlab()
        eng.addpath(os.path.dirname(self.matlab_script_path), nargout=0)
        eng.eval("cd('{}')".format(os.path.dirname(self.matlab_script_path)), nargout=0)
        try:
            y = eng.guidedNLMeans(
                matlab.double(amplitude.tolist()), 
                matlab.double(L), 
                matlab.double(eo.tolist()),
                matlab.double(stack_size),
                matlab.double(sharpness), 
                matlab.double(balance), 
                matlab.double(th_sar), 
                matlab.double(block_size), 
                matlab.double(win_size), 
                matlab.double(stride), 
                nargout=2)
            eng.quit()
        except Exception as e:
            eng.quit()
            raise RuntimeError(f"Error during execution: {e}")
        
        if isinstance(sar, jnp.ndarray):
            output = jnp.array(y[0]).reshape(amplitude.shape)
        else:
            output = np.array(y[0]).reshape(amplitude.shape) 

        return ap2c(output, phase)
    
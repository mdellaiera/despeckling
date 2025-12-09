import os
import numpy as np
import jax.numpy as jnp
import matlab.engine
from typing import Union
from scripts.utils import BaseFilter, c2ap, ap2c


class FANS(BaseFilter):
    """Wrapper around FANS filter."""

    def __init__(self, matlab_script_path: str):
        self.matlab_script_path = matlab_script_path
        self._set_library_path()

    def _set_library_path(self) -> None:
        os.environ['LD_LIBRARY_PATH'] = os.path.join(
            os.path.dirname(os.path.abspath(self.matlab_script_path)), "lib_opencv210/glnxa64"
        ) + ":" + os.environ.get('LD_LIBRARY_PATH', '')

    def filter(self, sar: Union[np.ndarray, jnp.ndarray], L: int) -> np.ndarray:
        amplitude, phase = c2ap(sar)

        eng = matlab.engine.start_matlab()
        eng.addpath(os.path.dirname(self.matlab_script_path), nargout=0)
        eng.eval("cd('{}')".format(os.path.dirname(self.matlab_script_path)), nargout=0)
        try:
            y = eng.FANS(matlab.double(amplitude.tolist()), matlab.double(L), nargout=2)
            eng.quit()
        except Exception as e:
            eng.quit()
            raise RuntimeError(f"Error during execution: {e}")

        if isinstance(sar, jnp.ndarray):
            output = jnp.array(y[0]).reshape(amplitude.shape)
        else:
            output = np.array(y[0]).reshape(amplitude.shape) 
            
        return ap2c(output, phase)

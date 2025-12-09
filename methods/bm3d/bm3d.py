import numpy as np
import bm3d
from scripts.utils import BaseFilter, c2ap, ap2c


class BM3D(BaseFilter):
    """Wrapper around BM3D filter."""

    def filter(self, sar: np.ndarray, sigma_psd: float) -> np.ndarray:
        amplitude, phase = c2ap(sar)
        filtered = bm3d.bm3d(amplitude, sigma_psd=sigma_psd, stage_arg=bm3d.BM3DStages.ALL_STAGES)
        return ap2c(filtered, phase)

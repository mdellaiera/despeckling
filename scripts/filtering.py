import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, List, Tuple
from tqdm import tqdm
from owl.cv.filters import FirstOrderDerivative, GaussianFilter2D


def compute_effective_number_of_looks(x: torch.Tensor) -> torch.Tensor:
    return x.mean()**2 / (x.var() + 1e-8)


def compute_noise_variation_coefficient(x: torch.Tensor) -> torch.Tensor:
    L = compute_effective_number_of_looks(x)
    assert L > 1, 'Effective number of looks must be greater than 1'
    return torch.sqrt(1 / L)


def compute_threshold(x: torch.Tensor) -> torch.Tensor:
    L = compute_effective_number_of_looks(x)
    assert L > 1, 'Effective number of looks must be greater than 1'
    return torch.sqrt(1 / (1 + 2 / L))


def compute_local_mean(x: torch.Tensor, window_size: int) -> torch.Tensor:
    x_ = x.unsqueeze(0).unsqueeze(0) if x.dim() == 2 else x.unsqueeze(0)
    mean = F.avg_pool2d(x_, window_size, stride=1, padding=window_size//2, count_include_pad=False)
    return mean.view_as(x)


def compute_local_variance(x: torch.Tensor, window_size: int) -> torch.Tensor:
    mean = compute_local_mean(x, window_size)
    mean_sq = compute_local_mean(x**2, window_size)  # Equivalent to unbiased=False in torch.var()
    return mean_sq - mean**2 

def compute_distance_matrix(window_size: int) -> torch.Tensor:
    X, Y = torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing='ij')
    return torch.sqrt((X - window_size // 2)**2 + (Y - window_size // 2)**2)


def compute_coefficient_of_variation(x: torch.Tensor, window_size: int) -> torch.Tensor:
    mean = compute_local_mean(x, window_size)
    std = compute_local_variance(x, window_size).sqrt()
    return std / (mean + 1e-8)


def compute_enl(x: torch.Tensor, window_size: int) -> torch.Tensor:
    mean = compute_local_mean(x, window_size)
    var = compute_local_variance(x, window_size)
    return mean**2 / (var + 1e-8)


def compute_multiscale_enl(x: torch.Tensor, window_size: Union[List, int]) -> torch.Tensor:
    window_size = [window_size] if isinstance(window_size, int) else window_size
    enl = torch.zeros_like(x)
    for w in window_size:
        enl += compute_enl(x, w)
    return enl / len(window_size)


def compute_soft_classification(x: torch.Tensor, a0: float, gamma: float, window_size: Union[List, int]) -> torch.Tensor:
    at = compute_enl(x, window_size)
    return F.sigmoid(-gamma * (at - a0))


def compute_conduction_coefficient(grad: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + (grad / (sigma * np.sqrt(2) + 1e-8))**2)


def _apply_frost_filter(x: torch.Tensor, window_size: int, alpha: float) -> torch.Tensor:
    # pad = window_size // 2
    # x_pad = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    # y = torch.zeros_like(x)

    # D = compute_distance_matrix(window_size)
    # C = compute_coefficient_of_variation(x_pad, window_size)
    # H, W = x.shape[-2:]
    # progress_bar = tqdm(total=H*W, desc='Applying Frost filter')
    
    # for i in range(H):
    #     for j in range(W):
    #         # Compute indices
    #         i_left, i_right = i, i + window_size
    #         j_left, j_right = j, j + window_size

    #         # Compute kernel
    #         C_neighborhood = C[:, i_left:i_right, j_left:j_right]
    #         kernel = torch.exp(-alpha * D * C_neighborhood)
    #         kernel /= kernel.sum()

    #         # Apply filter
    #         x_neighborhood = x_pad[:, i_left:i_right, j_left:j_right]
    #         y[:, i, j] = (kernel * x_neighborhood).sum()

    #         # Update progress bar
    #         progress_bar.update(1)
    device = x.device
    _, H, W = x.shape
    pad = window_size // 2

    # Prepare input for unfold (N, C, H, W)
    x_ = x.unsqueeze(0).unsqueeze(0) if x.dim() == 2 else x.unsqueeze(0)
    x_pad = F.pad(x_, (pad, pad, pad, pad), mode='reflect')  # (1, 1, H+2p, W+2p)
    unfold = torch.nn.Unfold(kernel_size=window_size)
    patches = unfold(x_pad)  # (1, window_size*window_size, H*W)
    patches = patches.squeeze(0)  # (window_size*window_size, H*W)

    # Compute coefficient of variation for each window center
    local_mean = compute_local_mean(x, window_size).squeeze()
    local_var = compute_local_variance(x, window_size).squeeze()
    C = torch.sqrt(local_var.clamp(min=1e-12)) / (local_mean + 1e-12)  # (H, W)
    C = C.reshape(-1)  # (H*W,)

    # Compute distance matrix
    D = compute_distance_matrix(window_size).reshape(1, -1)  # (1, window_size*window_size)

    # Compute per-patch kernels (vectorized!)
    # Each window has its own C (H*W,), but share the D (window)
    kernels = torch.exp(-alpha * D * C.unsqueeze(1))  # (H*W, window_size*window_size)
    kernels = kernels / (kernels.sum(dim=1, keepdim=True) + 1e-12)  # normalize each kernel

    # Apply filter: weighted sum for each patch
    filtered = (kernels * patches.t()).sum(dim=1)  # (H*W,)

    # Reshape to image
    y = filtered.reshape(H, W).view_as(x)
    return y


def _apply_lee_filter(x: torch.Tensor, window_size: int, relative_variance: float = 0.05) -> torch.Tensor:
    local_mean = compute_local_mean(x, window_size)
    local_variance = compute_local_variance(x, window_size)
    weights = local_variance / (local_mean**2 * relative_variance + local_variance + 1e-8)
    y = local_mean + weights * (x - local_mean)
    return y


def _apply_enhanced_lee_filter(x: torch.Tensor, window_size: int, k: float, Cu: float, Cmax: float) -> torch.Tensor:  #TODO: check if correct
    mu = compute_local_mean(x, window_size)
    C = compute_coefficient_of_variation(x, window_size)
    weights = torch.exp(-k * (C - Cu) / (Cmax - C)).clamp(0, 1)
    y = mu + weights * (x - mu)
    return y


def _apply_kuan_filter(x: torch.Tensor, window_size: int, Cu: float = 0.523) -> torch.Tensor:
    local_mean = compute_local_mean(x, window_size)
    if Cu is None:
        Cu = compute_noise_variation_coefficient(x)
    Ci = compute_coefficient_of_variation(x, window_size)
    weights = (1 - (Cu / (Ci + 1e-8))**2) / (1 + Cu**2)
    y = local_mean + weights * (x - local_mean)
    return y


def _apply_improved_kuan(x: torch.Tensor, window_size: int, lmbda: float, dt: float) -> torch.Tensor:
    """
    From the article "An Improved Kuan Algorithm for Despeckling of SAR Images".
    """
    kernel_north = FirstOrderDerivative()
    kernel_south = kernel_north.M()  # Mirror
    kernel_west = kernel_north.T()
    kernel_east = kernel_south.T()

    grad_north = kernel_north.apply(x)
    grad_south = kernel_south.apply(x)
    grad_west = kernel_west.apply(x)
    grad_east = kernel_east.apply(x)

    local_variance = compute_local_variance(x, window_size=7).sqrt()

    c_north = compute_conduction_coefficient(grad_north, local_variance)
    c_south = compute_conduction_coefficient(grad_south, local_variance)
    c_west = compute_conduction_coefficient(grad_west, local_variance)
    c_east = compute_conduction_coefficient(grad_east, local_variance)

    Cu = compute_noise_variation_coefficient(x)
    Cmax = compute_threshold(x)
    Ci = compute_coefficient_of_variation(x, window_size)

    local_mean = compute_local_mean(x, window_size)
    weights = (1 - (Cu / (Ci + 1e-8))**2) / (1 + Cu**2)
    weights = weights.clamp(0, 1)

    mask_low = Ci <= Cu
    mask_high = Ci >= Cmax
    mask_medium = (Ci > Cu) & (Ci < Cmax)

    x_low = (local_mean + weights * (x - local_mean)) * mask_low
    x_high = x * mask_high
    x_medium = (x + lmbda * dt * (c_north * grad_north + c_south * grad_south + c_east * grad_east + c_west * grad_west)) * mask_medium

    return x_low + x_high + x_medium


def unnormalized_bilateral_filter(S: torch.Tensor, 
                                  L: torch.Tensor, 
                                  sigma_s: float = 2, 
                                  sigma_l: float = 0.05, 
                                  alpha_ubf: float = 1.0, 
                                  n_iter: int = 1) -> Tuple[torch.Tensor, List[float]]:
    """
    Unnormalized Bilateral Filter (UBF) for descriptor maps.
    ----------
    Parameters
    S : torch.Tensor
        Input tensor of shape (D, H, W) where D is the number of descriptors, H is height, and W is width.
    L : torch.Tensor
        Luminance tensor of shape (1, H, W) used for weighting.
    sigma_s : float
        Spatial standard deviation for Gaussian kernel.
    sigma_l : float
        Luminance standard deviation for Gaussian weighting.
    alpha_ubf : float
        Scaling factor for the update step.
    n_iter : int
        Number of iterations for the filtering process.
    Returns
    torch.Tensor
        Filtered tensor of shape (D, H, W) after applying the unnormalized bilateral filter.
    """
    #TODO: allow learning rate schedule for alpha_ubf
    assert S.dim() == 3, 'Input tensor S must have shape (D, H, W)'
    assert L.dim() == 3 and L.shape[0] == 1, 'Luminance tensor L must have shape (1, H, W)'

    D, H, W = S.shape
    radius = int(3 * sigma_s)

    S0 = F.pad(S.clone(), [radius, radius, radius, radius], mode='reflect')  # (D, H+2*radius, W+2*radius)
    L0 = F.pad(L.clone(), [radius, radius, radius, radius], mode='reflect')  # (1, H+2*radius, W+2*radius)

    kernel_size = 2*radius+1
    Gs = GaussianFilter2D(sigma=sigma_s, radius=radius).kernel.to(S.device) # (kernel_size, kernel_size)
    error = []

    # Prepare to unfold for fast windowed ops
    # For each channel in S, unfold to (D, H*W, kernel_size*kernel_size)
    unfold = torch.nn.Unfold(kernel_size=kernel_size)
    # For batch-unfold, add a batch dimension: (1, D, H, W)
    for _ in tqdm(range(n_iter)):
        S0_ = S0.unsqueeze(0)  # (1, D, H+2r, W+2r)
        L0_ = L0.unsqueeze(0)  # (1, 1, H+2r, W+2r)

        # For each spatial location, extract the local window
        S_patches = unfold(S0_).reshape(D, kernel_size*kernel_size, H, W)  # (D, window, H, W)
        L_patches = unfold(L0_).reshape(1, kernel_size*kernel_size, H, W)  # (1, window, H, W)

        center_inds = kernel_size**2 // 2
        # Get center values (target pixel at each location)
        S_centers = S0[:, radius:H+radius, radius:W+radius]        # (D, H, W)
        L_centers = L0[0, radius:H+radius, radius:W+radius]        # (H, W)

        # Compute luminance gaussian weights for all windows at once
        # (1, window, H, W) - (H, W) => (1, window, H, W)
        L_diff = L_patches - L_centers.unsqueeze(0).flatten(1).unsqueeze(1).reshape(1, 1, H, W)
        Gl = torch.exp(-(L_diff**2) / (2 * sigma_l ** 2))  # (1, window, H, W)
        w = Gs.flatten().view(1, -1, 1, 1) * Gl            # (1, window, H, W)

        # For each channel, difference from center
        # (D, window, H, W) - (D, 1, H, W)
        diff = S_patches - S_centers.unsqueeze(1)          # (D, window, H, W)

        # Weighted sum: sum over window dimension
        update = (w * diff).sum(dim=1)                     # (D, H, W)

        S1 = S0.clone()
        S1[:, radius:H+radius, radius:W+radius] += alpha_ubf * update

        error.append(torch.norm(S1 - S0).item())
        S0 = S1

    return S0[:, radius:H+radius, radius:W+radius], error


def dual_unnormalized_bilateral_filter(S: torch.Tensor, 
                                  L_opt: torch.Tensor, 
                                  L_sar: torch.Tensor, 
                                  sigma_s: float = 2, 
                                  sigma_l_opt: float = 0.05, 
                                  sigma_l_sar: float = 0.05, 
                                  alpha_ubf: float = 1.0, 
                                  n_iter: int = 1) -> Tuple[torch.Tensor, List[float]]:
    """
    Unnormalized Bilateral Filter (UBF) for descriptor maps.
    ----------
    Parameters
    S : torch.Tensor
        Input tensor of shape (D, H, W) where D is the number of descriptors, H is height, and W is width.
    L_opt : torch.Tensor
        Optical luminance tensor of shape (1, H, W) used for weighting.
    L_sar : torch.Tensor
        SAR luminance tensor of shape (1, H, W) used for weighting.
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
    Returns
    torch.Tensor
        Filtered tensor of shape (D, H, W) after applying the unnormalized bilateral filter.
    """
    assert S.dim() == 3, 'Input tensor S must have shape (D, H, W)'
    assert L_opt.dim() == 3 and L_opt.shape[0] == 1, 'Optical luminance tensor L must have shape (1, H, W)'
    assert L_sar.dim() == 3 and L_sar.shape[0] == 1, 'SAR luminance tensor L must have shape (1, H, W)'

    D, H, W = S.shape
    radius = int(3 * sigma_s)

    S0 = F.pad(S.clone(), [radius, radius, radius, radius], mode='reflect')  # (D, H+2*radius, W+2*radius)
    L0_opt = F.pad(L_opt.clone(), [radius, radius, radius, radius], mode='reflect')  # (1, H+2*radius, W+2*radius)
    L0_sar = F.pad(L_sar.clone(), [radius, radius, radius, radius], mode='reflect')  # (1, H+2*radius, W+2*radius)

    kernel_size = 2*radius+1
    Gs = GaussianFilter2D(sigma=sigma_s, radius=radius).kernel.to(S.device) # (kernel_size, kernel_size)
    error = []

    # Prepare to unfold for fast windowed ops
    # For each channel in S, unfold to (D, H*W, kernel_size*kernel_size)
    unfold = torch.nn.Unfold(kernel_size=kernel_size)
    # For batch-unfold, add a batch dimension: (1, D, H, W)
    for _ in tqdm(range(n_iter)):
        S0_ = S0.unsqueeze(0)  # (1, D, H+2r, W+2r)
        L0_opt_ = L0_opt.unsqueeze(0)  # (1, 1, H+2r, W+2r)
        L0_sar_ = L0_sar.unsqueeze(0)  # (1, 1, H+2r, W+2r)

        # For each spatial location, extract the local window
        S_patches = unfold(S0_).reshape(D, kernel_size*kernel_size, H, W)  # (D, window, H, W)
        L_opt_patches = unfold(L0_opt_).reshape(1, kernel_size*kernel_size, H, W)  # (1, window, H, W)
        L_sar_patches = unfold(L0_sar_).reshape(1, kernel_size*kernel_size, H, W)  # (1, window, H, W)

        center_inds = kernel_size**2 // 2
        # Get center values (target pixel at each location)
        S_centers = S0[:, radius:H+radius, radius:W+radius]        # (D, H, W)
        L_opt_centers = L0_opt[0, radius:H+radius, radius:W+radius]        # (H, W)
        L_sar_centers = L0_sar[0, radius:H+radius, radius:W+radius]        # (H, W)

        # Compute luminance gaussian weights for all windows at once
        # (1, window, H, W) - (H, W) => (1, window, H, W)
        L_opt_diff = L_opt_patches - L_opt_centers.unsqueeze(0).flatten(1).unsqueeze(1).reshape(1, 1, H, W)
        L_sar_diff = L_sar_patches - L_sar_centers.unsqueeze(0).flatten(1).unsqueeze(1).reshape(1, 1, H, W)
        Gl_opt = torch.exp(-(L_opt_diff**2) / (2 * sigma_l_opt ** 2))  # (1, window, H, W)
        Gl_sar = torch.exp(-(L_sar_diff**2) / (2 * sigma_l_sar ** 2))
        w = Gs.flatten().view(1, -1, 1, 1) * Gl_opt * Gl_sar            # (1, window, H, W)

        # For each channel, difference from center
        # (D, window, H, W) - (D, 1, H, W)
        diff = S_patches - S_centers.unsqueeze(1)          # (D, window, H, W)

        # Weighted sum: sum over window dimension
        update = (w * diff).sum(dim=1)                     # (D, H, W)

        S1 = S0.clone()
        S1[:, radius:H+radius, radius:W+radius] += alpha_ubf * update

        error.append(torch.norm(S1 - S0).item())
        S0 = S1

    return S0[:, radius:H+radius, radius:W+radius], error


class SARFilter:
    
    def __init__(self, window_size: int, filter_type: str, **kwargs):
        self.window_size = window_size
        self.filter_type = filter_type
        self.kwargs = kwargs

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.filter_type == 'frost':
            return _apply_frost_filter(x, self.window_size, **self.kwargs)
        elif self.filter_type == 'lee':
            return _apply_lee_filter(x, self.window_size, **self.kwargs)
        elif self.filter_type == 'enhanced_lee':
            return _apply_enhanced_lee_filter(x, self.window_size, **self.kwargs)
        elif self.filter_type == 'kuan':
            return _apply_kuan_filter(x, self.window_size, **self.kwargs)
        elif self.filter_type == 'improved_kuan':
            return _apply_improved_kuan(x, self.window_size, **self.kwargs)
        elif self.filter_type == 'unnormalized_bilateral':
            return unnormalized_bilateral_filter(x, **self.kwargs)
        else:
            raise ValueError(f'Filter type "{self.filter_type}" not recognized')

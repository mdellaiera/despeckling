# SAR Despeckling Toolkit

This repository contains multiple despeckling baselines and the new method proposed in our article.

## Mamba Installatation (Recommended)

Install mamba following the installation procedure at https://github.com/conda-forge/miniforge.

```bash
$ wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
$ bash Miniforge3-$(uname)-$(uname -m).sh
$ ~/miniforge3/bin/conda init
```

## Conda Installation

If you prefer to install conda instead of mamba, you can refer to the following.

```bash
$ mkdir -p ~/miniconda3
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O
~/miniconda3/miniconda.sh
$ bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
$ rm ~/miniconda3/miniconda.sh
$ source ~/miniconda3/bin/activate
$ conda init
```

## Download the project from GitHub

```bash
$ git clone https://github.com/mdellaiera/despeckling.git
$ cd despeckling
$ mamba env create -f environment.yml
```

The environment created allows to use our method.

## Alternative methods

The baselines folder includes code or Python wrappers that allow you to reuse methodologies from the literature. Each method can be run using its corresponding environment. For detailed descriptions of each non-deep learning method, please refer to the excellent article [A Tutorial on Speckle Reduction in Synthetic Aperture Radar Images](https://ieeexplore.ieee.org/document/6616053). In addition, an overview of deep learning models can be found at [Deep Learning Methods For Synthetic Aperture Radar Image Despeckling: An Overview Of Trends And Perspectives](https://ieeexplore.ieee.org/document/9416740).

Matlab must be installed on your system to use this method. Python can call matlab code through the Matlab engine. Please follow the installation procedure at https://pypi.org/project/matlabengine/.


| Methods | Input type | Optical-Guided | Deep learning | Article | Code | 
| :------ | :--------: | :------------: | :-----------: | :-----: |:----:|
| PPB     | Amplitude  | No | No | [Iterative Weighted Maximum Likelihood Denoising With Probabilistic Patch-Based Weights](https://ieeexplore.ieee.org/document/5196737) | [Author personal webpage](https://www.charles-deledalle.fr/pages/ppb.php) | 
| BM3D    | Ampltide | No | No | [Image denoising by sparse 3D transform-domain collaborative ltering](https://ieeexplore.ieee.org/document/4271520) | [University website](https://webpages.tuni.fi/foi/GCF-BM3D/) |
| SAR-BM3D | Amplitude | No | No | [A Nonlocal SAR Image Denoising Algorithm Based on LLMMSE Wavelet Shrinkage](https://ieeexplore.ieee.org/document/5989862) | [University website](https://www.grip.unina.it/download/prog/SAR-BM3D/) |
| FANS   | Amplitude | No | No | [Fast Adaptive Nonlocal SAR Despeckling](https://ieeexplore.ieee.org/document/6564458) | [University website](https://www.grip.unina.it/download/prog/FANS/) |
| SAR2SAR | Amplitude | No | Yes | [SAR2SAR: a semi-supervised despeckling algorithm for SAR images](https://ieeexplore.ieee.org/document/9399231) | [GitHub](https://github.com/hi-paris/deepdespeckling) |
| MERLIN   | Complex data | No | Yes | [As if by magic: self-supervised training of deep despeckling networks with MERLIN](https://arxiv.org/abs/2110.13148) | [GitHub](https://github.com/hi-paris/deepdespeckling) |
| Speckle2void | Amplitude | No | Yes | [Speckle2Void: Deep Self-Supervised SAR Despeckling with Blind-Spot Convolutional Neural Networks](https://arxiv.org/abs/2007.02075) | [GitHub](https://github.com/diegovalsesia/speckle2void) |
| GBF    | Amplitude | Yes | No | [SAR despeckling guided by an optical image](https://ieeexplore.ieee.org/document/6947286) | No code available online, it has been re-implemented based on the article |
| GNLM      | Amplitude | Yes | No | [Guided patch-wise nonlocal SAR despeckling](https://arxiv.org/abs/1811.11872) | [University website](https://github.com/grip-unina/GNLM) |
| Fuse-MERLIN   | Complex data | Yes | Yes | [SELF-SUPERVISED LEARNING OF MULTI-MODAL COOPERATION FOR SAR DESPECKLING](https://telecom-paris.hal.science/hal-04676452v1/document) | Please contact the main author at ... to obtain filtered images |

## Results


# Despeckling
SAR Despeckling Toolkit

This repository contains the multiple despeckling baselines and the new method proposed in our article.

## Mamba Installatation (Recommended)

Install mamba following the installation procedure at https://github.com/conda-forge/miniforge.

```bash
$ wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
$ bash Miniforge3-$(uname)-$(uname -m).sh
$ ~/miniforge3/bin/conda init
```

## Conda Installation

If you prefer to install Conda instead of mamba, you can refer to the following.

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

## Test the different methods

### Mulog

```bash
$ mamba env create -f env_mulog.yml
```

Currently not working

### BM3D

```bash
$ mamba env create -f env_bm3d.yml
```

### SAR-BM3D

Matlab must be installed on your system to use this method. Python can call matlab code through the Matlab engine. Please follow the installation procedure at https://pypi.org/project/matlabengine/.

```bash
$ mamba env create -f env_sarbm3d.yml
```

### GNLM - Guided Non-Local Means

Currently not working

https://github.com/grip-unina/GNLM

### MERLIN

MERLIN is part of the deepdespeckling GitHub project.

```bash
$ git clone https://github.com/hi-paris/deepdespeckling
```

### Fuse-MERLIN




### SAR2SAR

SAR2SAR is part of the deepdespeckling GitHub project.

```bash
$ git clone https://github.com/hi-paris/deepdespeckling
```

### Speckle2void

```bash
$ git clone https://github.com/diegovalsesia/speckle2void
```

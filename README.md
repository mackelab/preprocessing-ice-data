# Preprocessing Ice Data for "Simulation-Based Inference of Surface Accumulation and Basal Melt Rates of an Antarctic Ice shelf from Isochronal Layers"

This repository contains research code for "Simulation-Based Inference of Surface Accumulation and Basal Melt Rates of an Antarctic Ice shelf from Isochronal Layers".
The repository used here is used to preprocess ice shelf thickness and velocity data from the real world, and to create synthetic ice shelves, in order to extract realistic transects. These transects are used in the main [repository](https://github.com/mackelab/sbi-ice) for the paper.
The picked IRH transect data required to run `calculate_flowtube.m`is found [here](https://nc-geophysik.guz.uni-tuebingen.de/index.php/s/wH5zqPaBGZAPRyD).

## Installation

This repository requires an installation of the finite element solver [firedrake](https://www.firedrakeproject.org), and the ice flow solver [icepack](https://icepack.github.io) to be installed in this environment. We direct users to follow the [installation instructions](https://www.firedrakeproject.org/download.html) of firedrake first.

Once a working firedrake environment is available on your machine, activate the virtual and environment and install this repository with:

```
  git clone https://github.com/mackelab/preprocessing-ice-data.git
  cd preprocessing_ice_data
  pip install -e setup.py
```


## Usage

Create and spin up synthetic ice shelves using `python preproc/synthetic_flowlines.py`. Configurations for the synthetic ice shelf can be controlled with `configs/`.

Loading and processing of data from Ekström Ice Shelf can be done with `python preproc/loading_flowlines.py`. Note that you need access to a version of a [Bedmachine dataset](https://nsidc.org/data/nsidc-0756/versions/3) and an [its_live velocity map](https://its-live.jpl.nasa.gov).

Creating the flow tube coordinates for Ekström ice shelves is done with `preproc/calculate_flowtube.m`.

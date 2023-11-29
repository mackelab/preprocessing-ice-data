from setuptools import find_packages, setup

# Package meta-data.
NAME = "preprocessing_ice_data"
URL = "https://github.com/mackelab/preprocessing_ice_data"
EMAIL = "guy.moss@student.uni-tuebingen.de"
AUTHOR = "Guy Moss"
REQUIRES_PYTHON = ">=3.9.0"

REQUIRED = ["scipy", "numpy", "matplotlib","pandas","hydra"]

setup(
    name=NAME,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(),
    install_requires=REQUIRED,
)
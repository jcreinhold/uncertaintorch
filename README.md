uncertaintorch
=======================
<!--
[![Build Status](https://travis-ci.org/jcreinhold/uncertaintorch.svg?branch=master)](https://travis-ci.org/jcreinhold/uncertaintorch)
[![Coverage Status](https://coveralls.io/repos/github/jcreinhold/uncertaintorch/badge.svg?branch=master)](https://coveralls.io/github/jcreinhold/uncertaintorch?branch=master)
[![Documentation Status](https://readthedocs.org/projects/uncertaintorch/badge/?version=latest)](http://uncertaintorch.readthedocs.io/en/latest/)
[![Docker Cloud Build Status](https://img.shields.io/docker/cloud/build/jcreinhold/uncertaintorch.svg)](https://hub.docker.com/r/jcreinhold/uncertaintorch/)-->
[![Python Versions](https://img.shields.io/badge/python-3.6%20%7C%203.7-blue.svg)](https://www.python.org/downloads/release/python-360/)
<!--[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2669612.svg)](https://doi.org/10.5281/zenodo.2669612)-->

This package contains deep neural network-based (pytorch) modules to synthesize or segment magnetic resonance (MR) and computed 
tomography (CT) brain images with uncertainty estimates.

** Note that this is an **alpha** release. If you have feedback or problems, please submit an issue (it is very appreciated) **

This package was developed by [Jacob Reinhold](https://jcreinhold.github.io) and the other students and researchers of the 
[Image Analysis and Communication Lab (IACL)](http://iacl.ece.jhu.edu/index.php/Main_Page).

[Link to main Gitlab Repository](https://gitlab.com/jcreinhold/uncertaintorch)

Requirements
------------

- matplotlib
- nibabel >= 2.3.1
- [niftidataset](https://github.com/jcreinhold/niftidataset) >= 0.1.4
- numpy >= 1.15.4
- pillow >= 5.3.0
- torch >= 1.2.0
- torchvision >= 0.2.1

Installation
------------

    pip install git+git://github.com/jcreinhold/uncertaintorch.git

Test Package
------------

Unit tests can be run from the main directory as follows:

    nosetests -v tests

Citation
--------

If you use the `uncertaintorch` package in an academic paper, please use the following citation:

    @misc{reinhold2020,
        author       = {Jacob Reinhold},
        title        = {{uncertaintorch}},
        year         = 2019,
        doi          = {10.5281/zenodo.2669612},
        version      = {0.3.2},
        publisher    = {Zenodo},
        url          = {https://doi.org/10.5281/zenodo.2669612}
    }
    
Relevant Papers
---------------

[1] J. Reinhold, Y. He, Y. Chen, D. Gao, J. Lee, J. Prince, A. Carass.
    ``Validating uncertainty in medical image translation.''
    2020 IEEE 17th International Symposium on Biomedical Imaging (ISBI 2020).
    IEEE, 2020.

[2] J. Reinhold, Y. He, Y. Chen, D. Gao, J. Lee, J. Prince, A. Carass.
    ``Finding novelty with uncertainty.''
    Medical Imaging 2020: Image Processing,
    International Society for Optics and Photonics, 2020.

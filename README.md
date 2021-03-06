uncertaintorch
=======================
[![Python Versions](https://img.shields.io/badge/python-3.6%20%7C%203.7-blue.svg)](https://www.python.org/downloads/release/python-360/)

This package contains deep neural network-based (pytorch) modules to synthesize or segment magnetic resonance (MR) and computed 
tomography (CT) brain images with uncertainty estimates.

The models in this repo were used to generate the experimental results in the two papers [[1](https://arxiv.org/abs/2002.04626),[2](https://arxiv.org/abs/2002.04626)].

This package was developed by [Jacob Reinhold](https://jcreinhold.github.io) of the
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

    @inproceedings{reinhold2020validating,
      title={Validating uncertainty in medical image translation},
      author={Reinhold, Jacob C and He, Yufan and Han, Shizhong and Chen, Yunqiang and Gao, Dashan and Lee, Junghoon and Prince, Jerry L and Carass, Aaron},
      booktitle={2020 IEEE 17th International Symposium on Biomedical Imaging (ISBI)},
      pages={95--98},
      year={2020},
      organization={IEEE}
    }
   
Relevant Papers
---------------

[1] J. Reinhold, Y. He, Y. Chen, D. Gao, J. Lee, J. Prince, A. Carass.
    ``Validating uncertainty in medical image translation).''
    2020 IEEE 17th International Symposium on Biomedical Imaging (ISBI 2020).
    IEEE, 2020.

[2] J. Reinhold, Y. He, Y. Chen, D. Gao, J. Lee, J. Prince, A. Carass.
    ``Finding novelty with uncertainty.''
    Medical Imaging 2020: Image Processing,
    International Society for Optics and Photonics, 2020.

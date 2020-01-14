#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
setup

Module installs the uncertaintorch package
Can be run via command: python setup.py install (or develop)

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Jan 14, 2020
"""

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

version = '0.0.1'

args = dict(
    name='uncertaintorch',
    version=version,
    description="pytorch-based uncertainty in segmentation and synthesis",
    long_description=readme,
    author='Jacob Reinhold',
    author_email='jacob.reinhold@jhu.edu',
    url='https://gitlab.com/jcreinhold/uncertaintorch',
    license=license,
    packages=find_packages(exclude=('tests')),
    keywords="mr image-synthesis uncertainty-quantification image-segmentation",
    dependency_links=[f'git+git://github.com/jcreinhold/niftidataset.git@master#egg=niftidataset-0.1.4']
)

setup(install_requires=['matplotlib',
                        'nibabel>=2.3.1',
                        'niftidataset>=0.1.4',
                        'numpy>=1.15.4',
                        'pillow>=5.3.0'
                        'torch>=1.1.0',
                        'torchvision>=0.2.2'], **args)

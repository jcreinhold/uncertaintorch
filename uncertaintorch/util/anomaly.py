#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
anomaly

holds functions to create synthetic anomalies

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: January 1, 2020
"""

__all__ = ['block', 'gaussian_kernel']

import argparse
import logging
import os
import sys
import warnings

import nibabel as nib
import numpy as np
import scipy.ndimage.filters as fi
from scipy.ndimage.morphology import binary_erosion

from niftidataset import glob_imgs, split_filename


def gaussian_kernel(im, mask, ht=1, ksz=7, n_erode=4):
    g = np.zeros_like(im)
    msk = binary_erosion(mask, iterations=n_erode)
    mask_idxs = np.where(msk > 0)
    c = np.random.randint(0, len(mask_idxs[0]))  # choose the set of idxs to use
    h, w, d = [m[c] for m in mask_idxs]  # pull out the chosen idxs (2D)
    g[h, w, d] = 1
    g = fi.gaussian_filter(g, ksz)
    g = ht * (g - g.min()) / (g.max() - g.min())
    return g


def block(im, mask, ht=1, sz=7, n_erode=4):
    g = np.zeros_like(im)
    msk = binary_erosion(mask, iterations=n_erode)
    mask_idxs = np.where(msk > 0)
    c = np.random.randint(0, len(mask_idxs[0]))  # choose the set of idxs to use
    h, w, d = [m[c] for m in mask_idxs]  # pull out the chosen idxs (2D)
    g[h-sz//2:h+sz//2+1, w-sz//2:w+sz//2+1, d-sz//2:d+sz//2+1] = ht
    return g

# --- script functions ---

def arg_parser():
    parser = argparse.ArgumentParser(description='create synthetic anomalies '
                                                 'for a directory of nifti images')
    required = parser.add_argument_group('Required')
    required.add_argument('-i', '--img-dir', type=str, required=True,
                          help='path to directory with images to be processed')
    required.add_argument('-o', '--out-dir', type=str, required=True,
                          help='output directory for preprocessed files')

    options = parser.add_argument_group('Options')
    options.add_argument('-m', '--mask-dir', type=str, default=None,
                          help='directory to output the corresponding img files')
    options.add_argument('-n', '--n-anom', type=int, default=5,
                         help='number of anomalous images to create per image')
    options.add_argument('-sz', '--size', type=int, default=7,
                          help='size of anomaly (either kernel size or side length)')
    options.add_argument('-ht', '--height', type=float, default=1,
                          help='`height` of the anomaly (i.e., the peak value of the anomaly)')
    options.add_argument('-ne', '--n-erode', type=int, default=4,
                         help='number of times to erode mask before determining where to '
                              'randomly place the desired anomaly')
    options.add_argument('-op', '--operation', type=str, default='add', choices=('mult', 'add', 'set'),
                          help='how to apply the anomaly to the images (set means `set the value to`)')
    options.add_argument('--type', type=str, default='block', choices=('block','gaussian'),
                         help='create gaussian kernel anomalies or block anomalies')
    options.add_argument('-v', '--verbosity', action="count", default=0,
                         help="increase output verbosity (e.g., -vv is more than -v)")
    return parser


def main(args=None):
    args = arg_parser().parse_args(args)
    if args.verbosity == 1:
        level = logging.getLevelName('INFO')
    elif args.verbosity >= 2:
        level = logging.getLevelName('DEBUG')
    else:
        level = logging.getLevelName('WARNING')
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=level)
    logger = logging.getLogger(__name__)
    try:
        # grab the file names for the images of interest
        img_fns = glob_imgs(args.img_dir)

        # define and get the brain masks for the images, if defined
        if args.mask_dir is None:
            mask_fns = [None] * len(img_fns)
        else:
            mask_fns = glob_imgs(args.mask_dir)
            if len(img_fns) != len(mask_fns):
                raise ValueError(f'Number of images and masks must be equal, Images: {len(img_fns)}, Masks: {len(mask_fns)}')

        # create output directory if it doesn't exist
        if not os.path.exists(args.out_dir):
            logger.info(f'{args.out_dir} does not exist, creating it.')
            os.mkdir(args.out_dir)
        anom_mask_dir = os.path.join(args.out_dir, 'mask')
        anom_img_dir = os.path.join(args.out_dir, 'img')
        if not os.path.exists(anom_mask_dir):
            os.mkdir(anom_mask_dir)
        if not os.path.exists(anom_img_dir):
            os.mkdir(anom_img_dir)

        # handle height parameter
        if args.height == 0. and args.operation != 'set':
            logger.warning('If height is zero, a `set` anomaly is created')
            args.operation = 'set'
        ht = args.height if args.operation != 'set' else 1.

        # add anomalies and save the results
        for i, (img_fn, mask_fn) in enumerate(zip(img_fns, mask_fns), 1):
            _, base, ext = split_filename(img_fn)
            logger.info('Adding synthetic anomaly to image: {} ({:d}/{:d})'.format(base, i, len(img_fns)))
            for j in range(args.n_anom):
                img = nib.load(img_fn)
                data = img.get_fdata()
                mask = nib.load(mask_fn).get_fdata() if mask_fn is not None else data > 0
                if args.type == 'block':
                    anom = block(data, mask, ht=ht, sz=args.size, n_erode=args.n_erode)
                elif args.type == 'gaussian':
                    anom = gaussian_kernel(data, mask, ht=ht, ksz=args.size, n_erode=args.n_erode)
                else:
                    raise ValueError('`type` can only be block or gaussian')
                anom_mask_fn = os.path.join(anom_mask_dir, base + f'_{j}.nii.gz')
                anom_mask = (anom > 0.5) if args.type == 'gaussian' else (anom > 0)
                nib.Nifti1Image(anom_mask.astype(np.float32), img.affine, img.header).to_filename(anom_mask_fn)
                if args.operation == 'mult':
                    if args.type == 'block':
                        anom[anom == 0] = 1
                    elif args.type == 'gaussian':
                        anom += 1
                    else:
                        raise Exception('something went wrong')
                    fake = data * anom
                elif args.operation == 'add':
                    fake = data + anom
                elif args.operation == 'set':
                    fake = data.copy()
                    fake[anom > 0] = args.height
                else:
                    raise ValueError('`operation` can only be add, mult, or set')
                anom_img_fn = os.path.join(anom_img_dir, base + f'_{j}.nii.gz')
                nib.Nifti1Image(fake, img.affine, img.header).to_filename(anom_img_fn)
        return 0
    except Exception as e:
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

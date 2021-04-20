"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

Author: Ruchi Chauhan
"""
import kornia
import torch
import random
import numpy as np


def kornia_affine(im, parameter, aug_type, data_type='data'):
    '''
    Get rotation by given angle or scale by given factor
    along axis-0 using kornia.
    (See https://kornia.readthedocs.io/en/latest/geometry.transform.html)
    '''
    center = torch.ones(1, 2).cuda()
    center[..., 0] = im.shape[1]//2
    center[..., 1] = im.shape[2]//2
    if aug_type == 'rotate':
        scale = torch.ones(1).cuda()
        angle = parameter*scale
    elif aug_type == 'scale':
        scale = torch.Tensor([parameter]).cuda()
        angle = 0*scale
        # vol_warped = kornia.scale(vol[:, 0, :, :, :], scale, center)
    if data_type == 'data':
        interpolation = 'bilinear'
    elif data_type == 'label':
        interpolation = 'nearest'
    M = kornia.get_rotation_matrix2d(center, angle, scale)
    _, h, w = im.shape
    im_warped = kornia.warp_affine(im[None, :, :, :].float(), M.cuda(),
                                   dsize=(h, w), flags=interpolation)
    # vol_warped = vol_warped[:, None, :, :, :]
    return im_warped[0]


def kornia_translate(im, choice, data_type='data'):
    '''
    Random translation using kornia translate transform.
    Additional function needed for construction of translation Tensor.
    Choice has format str(magnitude)+'axis' where axis = x or y.
    '''
    axis = choice[-1]
    transMag = int(choice[:-1])
    transVal = torch.zeros(im.shape[0], 2)
    if axis == 'x':
        transVal[:, 0] = transMag
    elif axis == 'y':
        transVal[:, 1] = transMag
    if data_type == 'data':
        interpolation = 'bilinear'
    elif data_type == 'label':
        interpolation = 'nearest'
    M = kornia.geometry.transform.affwarp._compute_translation_matrix(transVal)
    _, _, h, w = im.shape
    vol_warped = kornia.warp_affine(im.float(), M.cuda(), dsize=(h, w),
                                    flags=interpolation)
    return vol_warped


def augment_gaussian_noise(data_sample, noise_variance=(0, 0.1)):
    '''
    Modified function from batchgenerators to process cuda tensor.
    '''
    if noise_variance[0] == noise_variance[1]:
        variance = noise_variance[0]
    else:
        variance = random.uniform(noise_variance[0], noise_variance[1])
    data_sample = data_sample + torch.Tensor(np.random.normal(0.0,
                                             variance, size=data_sample.shape)
                                             ).cuda()
    return data_sample

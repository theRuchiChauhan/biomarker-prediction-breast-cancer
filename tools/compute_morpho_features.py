"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

Compute morphological features.
Author: Ruchi Chauhan.
"""
import os
import time
import multiprocessing
import histomicstk as htk
from PIL import Image
import pandas as pd
import numpy as np


def get_feats(name):
    im_mask = np.array(Image.open(os.path.join(mask_dir, name)))
    im_mask = im_mask.astype('uint8')
    feats_morph = htk.features.compute_morphometry_features(im_mask)
    # feats_fsd = htk.features.compute_fsd_features(im_mask)
    # feats_grad = htk.features.compute_gradient_features(im_mask, im)
    # feats_haralick = htk.features.compute_haralick_features(im_mask, im)
    names = pd.DataFrame(np.array([name]), columns=['patch_names'])
    # feats_patch_all = pd.concat([names, feats_fsd, feats_grad, feats_haralick],
    #                             1, sort=False)
    feats_patch_all = pd.concat([names, feats_morph], 1, sort=False)
    return feats_patch_all


mask_dir = 'CleanedCode_26Jul20/allMasks_L0'
img_dir = 'colorNormPatches/tst_L0'

pool = multiprocessing.Pool(processes=8)
t1 = time.time()
rets = pool.map(get_feats, os.listdir(mask_dir))
feats_all = pd.concat(rets, 0, sort=False)
feats_all.to_csv('htk_feats_morpho.csv', index=False)
t2 = time.time()
print('For all patches, time is '+str(t2-t1)+' seconds.')

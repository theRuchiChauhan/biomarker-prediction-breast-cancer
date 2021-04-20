"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

Code to Extract L0 patches from L1 patches
Author: Ruchi Chauhan
Date: 6 Aug 2020
----------------------------------------------------------
python3 LevelConvertor.py tp53Patches_90 tp53Patches_90_L0
"""
from PIL import Image
import numpy as np
import sys,os

L1folderPath = sys.argv[1]
L0folderPath = sys.argv[2]

imageList = os.listdir(L1folderPath)
for img in imageList:
    patch = Image.open(L1folderPath+'/'+img)
    largePatch = patch.resize((1024,1024), Image.LANCZOS)

    part1 = Image.fromarray(np.array(largePatch)[:512, :512])
    part2 = Image.fromarray(np.array(largePatch)[:512, 512:])
    part3 = Image.fromarray(np.array(largePatch)[512:, :512])
    part4 = Image.fromarray(np.array(largePatch)[512:, 512:])
    patchName = img[:-4]
    part1.save(L0folderPath + '/' + patchName + '_a.png')
    part2.save(L0folderPath + '/' + patchName + '_b.png')
    part3.save(L0folderPath + '/' + patchName + '_c.png')
    part4.save(L0folderPath + '/' + patchName + '_d.png')
    
    print(f'{img} done')

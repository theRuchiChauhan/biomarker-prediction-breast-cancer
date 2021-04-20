"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

put TP53 & noTP53 high probability patches in two separate folder, read the names from list.
task number_of_patches
"""
import pandas as pd
import shutil, os
import sys

task = sys.argv[1] #ER
numPatches = int(sys.argv[2]) #20

subTasks = ['pos','neg']

sourcePath = '../colorNormPatches/tst_L0/'

for i in subTasks:

    filePath = "morph_matlab_code/Sept_5/"+task+"_"+i+"DiscSorted.csv"
    destFolName = i+"_"+task+"_discPatches"

    if (1-os.path.isdir(destFolName)):
        os.mkdir(destFolName)

    patchList = pd.read_csv(filePath, header=None).iloc[:,0].tolist()
    patchList = patchList[:numPatches]
    # pdb.set_trace()
    for i,entry in enumerate(patchList,1):

        if 'normal' in entry:
            entry = entry.rsplit('-',1)[0] ## remove -normal
        newName = '_'.join(entry.split('/'))

        shutil.copy2(sourcePath+entry,destFolName+'/'+newName)
        if i%10 == 0:
            print(f'{i} done')

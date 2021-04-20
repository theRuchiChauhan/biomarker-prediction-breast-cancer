"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

Tools here generate config files i.e. lists of slides, paths, label mapping
for trn, tst, val and also loads them into memory.
Author: Ruchi Chauhan
"""
import os
import sys


def collect_patches(path, category, slide_list):
    patch_list = []
    for slide in slide_list:
        patch_list += [slide+'/'+patch_name for patch_name in
                       os.listdir(os.path.join(path, category, slide))
                       if (patch_name.endswith('_cn.png') and not
                           patch_name.endswith('_rotated.png'))]
    return patch_list


def save_file_lists(task):
    '''
    Generate slide list and patch list for trn, val, tst
    for task norm_vs_cancer or mutation.
    '''
    for category in categories:
        f1 = open('file_lists/'+task+'_'+category+'_patch_list.txt', 'a')
        f2 = open('file_lists/'+task+'_'+category+'_slide_list.txt', 'a')
        if task == 'normal_vs_cancer':
            slide_list_cat = os.listdir(os.path.join(normal_vs_cancer_path,
                                                     category))
            patch_list = collect_patches(normal_vs_cancer_path, category,
                                         slide_list_cat)
        elif task == 'mutation':
            slide_list_cat = os.listdir(os.path.join(mut_data_path,
                                                     category))
            # slide_list_cat_norm_vs_cancer = os.listdir(os.path.join(
            #     normal_vs_cancer_path, category))
            # slide_list_cat = list(set(slide_list_cat_mut_all) - set(
            # slide_list_cat_norm_vs_cancer))
            patch_list = collect_patches(mut_data_path, category,
                                         slide_list_cat)
        for name in patch_list:
            f1.write(name+'\n')
        for name in slide_list_cat:
            f2.write(name+'\n')
        f1.close()
        f2.close()


if __name__ == '__main__':
    mut_data_path = sys.argv[1]
    normal_vs_cancer_path = sys.argv[2]
    categories = ['trn', 'val', 'tst']
    save_file_lists('mutation')  # 'normal_vs_cancer')

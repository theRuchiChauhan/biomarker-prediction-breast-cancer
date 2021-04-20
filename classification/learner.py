"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

Main Classification module
Author: Ruchi Chauhan
"""

from collections import namedtuple, defaultdict
import pdb
import os

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import sklearn
import sklearn.metrics
import imageio
import numpy as np
import pandas as pd
from tqdm import trange

import utils
from manage_config import mut_data_path, normal_vs_cancer_path


class DataLoader(object):
    def __init__(self, data_type, task, class_name='', sample_size=''):
        '''
        data_type should be trn, val, or tst respectively for train, val
        and test. Same will be used as prefix in all files.
        '''
        self.task = task
        self.data_type = data_type
        if task == 'normal_vs_cancer':
            self.file_path = normal_vs_cancer_path
            self.class_ID = 0
        elif task == 'mutation':
            self.file_path = mut_data_path
            if class_name:
                self.class_ID = self.get_class_ID(class_name)
            else:
                raise AssertionError('Class name not specified for mutation'
                                     ' prediction.')
        elif task == 'external_test':
            # case when pred on mutation data using norm vs cancer model
            self.file_path = mut_data_path
            self.class_ID = 0
        patch_list, slide_list = self.get_file_list(task, data_type,
                                                    class_name, model_class)
        self.label_dict = self.gen_label_dict(task, data_type, slide_list)
        self.sample_size = sample_size
        # if data_type == 'trn' and sample_size:
        #     patch_list, slide_list = self.undersample_data(patch_list,
        #                                                    sample_size)
        # elif data_type == 'trn' and not sample_size:
        #     print("Warning: Sample size for undersampling not given")
        self.patch_list = patch_list
        self.slide_list = slide_list

    def undersample_data(self, patch_list, sample_size):
        patch_list_neg = []
        patch_list_pos = []
        final_slide_list = []
        for patch_name in patch_list:
            slide_name = patch_name.split('/')[0]
            if self.label_dict[slide_name]:
                patch_list_pos.append(patch_name)
            else:
                patch_list_neg.append(patch_name)
        selected_patch_list_pos = np.random.choice(np.array(patch_list_pos),
                                                   (sample_size,),
                                                   replace=False)
        selected_patch_list_neg = np.random.choice(np.array(patch_list_neg),
                                                   (sample_size,),
                                                   replace=False)
        final_patch_list = np.concatenate((selected_patch_list_pos,
                                           selected_patch_list_neg), 0)
        for patch_name in final_patch_list:
            final_slide_list.append(patch_name.split('/')[0])
        final_slide_list = list(set(final_slide_list))
        return final_patch_list, final_slide_list

    def get_file_list(self, task, data_type, class_name, model_class):
        ''' Loads slide list and list of patches with slide name. '''
        if task == 'external_test':
            task = 'mutation'
            data_type = 'all'
        if class_name == 'subtype':
            class_name = 'hist_subtype'
        if model_class == 'subtype':
            model_class = 'hist_subtype'
        # slide_list_model = pd.read_csv(('file_lists/23Aug/'+task+'_selected_L0_'
        #                                + data_type + '_slide_list_redist_'
        #                                + model_class + '.txt'),
        #                                delimiter='\n', header=None).values
        # slide_list_class = pd.read_csv(('file_lists/23Aug/'+task+'_selected_L0_'
        #                                + data_type + '_slide_list_redist_'
        #                                + class_name + '.txt'),
        #                                delimiter='\n', header=None).values
        # slide_list_class = set(tuple(slide_list_class.reshape((-1,)).tolist()))
        # slide_list_model = set(tuple(slide_list_model.reshape((-1,)).tolist()))
        # slide_list = slide_list_class - slide_list_model
        # slide_list = np.array(list(slide_list))
        patch_list_model = pd.read_csv(('file_lists/23Aug/'+task+'_selected_L0_'
                                        + data_type + '_patch_list_redist_'
                                        + model_class + '.txt'),
                                       delimiter='\n', header=None).values
        patch_list_class = pd.read_csv(('file_lists/23Aug/'+task+'_selected_L0_'
                                        + data_type + '_patch_list_redist_'
                                        + class_name + '.txt'),
                                       delimiter='\n', header=None).values
        patch_list_class = set(tuple(patch_list_class.reshape((-1,)).tolist()))
        patch_list_model = set(tuple(patch_list_model.reshape((-1,)).tolist()))
        patch_list = patch_list_class - patch_list_model
        patch_list = np.array(list(patch_list))
        slide_list = []
        for patch_name in patch_list:
            slide_name = patch_name.split('/')[0]
            slide_list.append(slide_name)
        slide_list = np.array(slide_list)
        # slide_list = pd.read_csv('file_lists/mutation_selected_'+data_type
        #                          + '_slide_list.txt', header=None).values

        # patch_list = pd.read_csv('file_lists/mutation_selected_L0_'+data_type
        #                          + '_patch_list.txt', header=None).values
        slide_list = np.reshape(slide_list, (-1,))
        patch_list = np.reshape(patch_list, (-1,))
        return patch_list, slide_list

    def gen_label_dict(self, task, data_type, slide_list):
        ''' Loads labels and maps them to slide names in a dict. '''
        slide_lbl_dict = {}
        if task == 'external_test':
            task = 'normal_vs_cancer'
        if task == 'normal_vs_cancer':
            for slide_name in slide_list:
                # slide name for normal is of format TCGA-A7-A4SA-11A-02
                if len(slide_name.split('-')) > 3:
                    slide_lbl_dict[slide_name] = 0
                else:
                    slide_lbl_dict[slide_name] = 1

        elif task == 'mutation':
            lbl_data = pd.read_csv('file_lists/curated_708_5genes.csv').values
            for slide_name in slide_list:
                label = lbl_data[lbl_data[:, 0] == slide_name+'-01',
                                 self.class_ID]
                slide_lbl_dict[slide_name] = label

        return slide_lbl_dict

    def get_class_ID(self, class_name):
        class_dict = np.array(['name', 'CDH1', 'GATA3', 'KMT2C', 'PIK3CA',
                               'TP53', 'subtype'])
        class_id = np.where(class_dict == class_name)[0][0]
        return class_id

    # def get_num_samples(self):
        # , aug_list_0=['normal'], aug_list_1=['normal']):
        # if self.data_type == 'trn':
        # return len(self.get_aug_list(aug_list_0, aug_list_1,
        # random_aug=True))
        # else:
        #     return len(self.patch_list)
        # return len(self.patch_list)

    def get_num_batches(self, batch_size, aug_list_0=['normal'],
                        aug_list_1=['normal']):
        if self.data_type == 'trn':
            num_samples = 2*self.sample_size  # self.get_num_samples()
        else:
            num_samples = len(self.patch_list)
        if num_samples % batch_size == 0:
            num_batches = num_samples // batch_size
        else:
            num_batches = (num_samples // batch_size) + 1
        return num_batches

    def get_label_dist(self):
        class0_num = 0
        class1_num = 0
        for patch_name in self.patch_list:
            slide_name = patch_name.split('/')[0]
            if self.label_dict[slide_name] == 1:
                class1_num += 1
            else:
                class0_num += 1
        # import pdb; pdb.set_trace()
        return (class0_num, class1_num)

    def get_aug_list(self, aug_list_0, aug_list_1, random_aug=False):
        aug_file_list = []
        if self.data_type == 'trn':
            patch_list, slide_list = self.undersample_data(self.patch_list,
                                                           self.sample_size)
        else:
            patch_list = self.patch_list
        for patch_name in patch_list:
            # if self.task == 'external_test':
            #     slide_name = patch_name.split('/')[1]
            # else:
            slide_name = patch_name.split('/')[0]
            if random_aug:
                if self.label_dict[slide_name] == 1:
                    aug = np.random.choice(aug_list_1)
                elif self.label_dict[slide_name] == 0:
                    aug = np.random.choice(aug_list_0)
                aug_file_list += [patch_name+'-'+aug]
            else:
                if self.label_dict[slide_name] == 1:
                    aug_file_list += [patch_name+'-'+aug for aug in aug_list_1]
                elif self.label_dict[slide_name] == 0:
                    aug_file_list += [patch_name+'-'+aug for aug in aug_list_0]
        return aug_file_list

    def reset_counters(self):
        image_batch = []
        label_batch = []
        count_patch = 0
        file_name_batch = []
        return image_batch, label_batch, count_patch, file_name_batch

    def data_gen(self, batch_size, aug_list_0=['normal'],
                 aug_list_1=['normal']):
        ''' Generator to provide image and label batches. '''
        file_list = self.get_aug_list(aug_list_0, aug_list_1, random_aug=True)
        if self.data_type == 'trn':
            file_list = np.random.permutation(file_list)
        # else:
        #     file_list = self.get_aug_list(['normal'], ['normal'])
        label_dict = self.label_dict
        dataBuffer = ['']
        num_samples = len(file_list)
        checkpoint = num_samples // batch_size  # for saving left out samples
        leftover_samples = num_samples % batch_size
        while True:
            image_batch, label_batch, count_patch,\
                file_name_batch = self.reset_counters()
            num_batches = 0
            for file_name in file_list:
                # data_type = self.data_type
                slide_name, patch_name_w_aug = file_name.split('/')
                patch_name = '-'.join(patch_name_w_aug.split('-')[:-1])
                aug_type = patch_name_w_aug.split('-')[-1]
                if aug_type == 'normal' or file_name != dataBuffer[0]:
                    patch = imageio.imread(os.path.join(self.file_path,
                                                        slide_name
                                                        + '/' + patch_name))
                    patch = torch.Tensor(patch).cuda()
                    # pytorch wants channels first
                    patch = patch.permute(2, 1, 0)
                    dataBuffer = []
                    dataBuffer.append(file_name)
                    dataBuffer.append(patch)
                if aug_type != 'normal':
                    patch = utils.augment(dataBuffer[1], aug_type)
                # Normalisation
                patch = (patch - torch.mean(patch))/torch.std(patch)
                # patch = (patch - torch.min(patch))/(torch.max(patch)
                #                                     - torch.min(patch))
                if task == 'mutation':
                    label = label_dict[slide_name][0]
                else:
                    label = label_dict[slide_name]
                label = torch.Tensor([label]).long()
                image_batch.append(patch)
                label_batch.append(label)
                file_name_batch.append(file_name)
                count_patch += 1
                if count_patch == batch_size:
                    image_batch_tensor = torch.stack(image_batch)
                    label_batch_tensor = torch.stack(label_batch)
                    yield image_batch_tensor, label_batch_tensor,\
                        file_name_batch

                    image_batch, label_batch, count_patch,\
                        file_name_batch = self.reset_counters()
                    num_batches += 1

                elif num_batches == checkpoint and count_patch \
                        == leftover_samples:
                    yield torch.stack(image_batch), torch.stack(label_batch),\
                            file_name_batch


class Aggregator():
    def __init__(self, data_type, task, data_loader_obj):
        super(Aggregator, self).__init__()
        self.data_type = data_type
        self.task = task
        patch_list, slide_list = data_loader_obj.get_file_list(task, data_type)
        self.slide_dict = self.get_slide_dict(patch_list, slide_list)
        self.label_dict = data_loader_obj.label_dict
        self.file_path = data_loader_obj.file_path
        self.slide_list = slide_list

    def get_slide_dict(self, patch_list, slide_list):
        slide_dict = defaultdict(list)
        for slide_name in slide_list:
            for patch_name in patch_list:
                patch_slide_name = patch_name.split('/')[0]
                # patch_slide_name = patch_name.split('/')[1]
                if patch_slide_name == slide_name:
                    slide_dict[slide_name].append(patch_name)
        return slide_dict

    def maxVoting(self, hard_preds):
        count_0votes = torch.sum(torch.cat(hard_preds) == 0)
        count_1votes = torch.sum(torch.cat(hard_preds) == 1)
        ratio_0 = count_0votes.float()/(count_0votes+count_1votes)
        ratio_1 = count_1votes.float()/(count_0votes+count_1votes)
        soft_pred = torch.Tensor([ratio_0, ratio_1])
        hard_pred = torch.argmax(soft_pred.unsqueeze(0), 1)
        return soft_pred, hard_pred

    def avgProbability(self, soft_preds, nPatches):
        avg_soft_pred = torch.sum(torch.cat(soft_preds), 0)/nPatches
        hard_pred = torch.argmax(avg_soft_pred.unsqueeze(0), 1)
        return avg_soft_pred, hard_pred

    def getMetrics(self, soft_pred_list, pred_list, label_list):
        acc = utils.globalAcc(pred_list, label_list)
        F1 = sklearn.metrics.f1_score(label_list, pred_list, labels=None)
        auc, auprc, auc_params, prc_params = utils.AUC(soft_pred_list,
                                                       label_list)
        return acc, F1, auc, auprc

    def get_slide_wise_pred(self, model, slide_name, threshold=0.9):
        patch_list = self.slide_dict[slide_name]
        label = self.label_dict[slide_name]
        model.eval()
        soft_pred_list = []
        pred_list = []
        selected_patches = []
        for patch in patch_list:
            img = imageio.imread(os.path.join(self.file_path, patch))
            img = torch.Tensor(img).cuda()
            img = (img - torch.mean(img))/torch.std(img)
            img = img.permute(2, 1, 0)
            img = img.unsqueeze(0)
            soft_pred = F.softmax(model.forward(img), 1).detach().cpu()
            soft_pred_list.append(soft_pred)
            hard_pred = torch.argmax(soft_pred, 1)
            pred_list.append(hard_pred)
            if label and soft_pred[0, 1].item() > threshold:
                selected_patches.append(patch)
        return soft_pred_list, pred_list, selected_patches

    def aggregate(self):
        predList_maxVoting = []
        predList_avgProb = []
        labelList = []
        softPredList_maxVoting = []
        softPredList_avgProb = []
        allSoftPredList = []
        allSelectedList = []
        for i, slide_name in enumerate(self.slide_list, 1):
            nPatches = len(self.slide_dict[slide_name])
            if nPatches == 0:
                continue
            soft_pred_list, pred_list,\
                selected_patches = self.get_slide_wise_pred(model,
                                                            slide_name,
                                                            0.9)
            label = self.label_dict[slide_name]
            labelList.append(torch.Tensor([label]).long())
            allSelectedList += selected_patches
            allSoftPredList.append(soft_pred_list)
            softPredSlide_maxVoting, hardPredSlide_maxVoting = self.maxVoting(
                pred_list)
            softPredList_maxVoting.append(softPredSlide_maxVoting)
            predList_maxVoting.append(hardPredSlide_maxVoting)
            softPredSlide_avgProb, hardPredSlide_avgProb = self.avgProbability(
                soft_pred_list, nPatches)
            softPredList_avgProb.append(softPredSlide_avgProb)
            predList_avgProb.append(hardPredSlide_avgProb)
            if i % 10 == 0:
                print(str(i)+' slides predicted.')
        labelList = torch.cat(labelList)
        # np.savetxt('selected_patches.txt', allSelectedList,
        #            delimiter='\n')
        softPredList_maxVoting = torch.stack(softPredList_maxVoting, 0)
        softPredList_avgProb = torch.stack(softPredList_avgProb, 0)
        predList_maxVoting = torch.Tensor(predList_maxVoting)
        predList_avgProb = torch.Tensor(predList_avgProb)
        acc_maxVoting, f1_maxVoting, auc_maxVoting,\
            auc_prc_maxVoting = self.getMetrics(softPredList_maxVoting,
                                                predList_maxVoting,
                                                labelList)
        acc_avgProb, f1_avgProb, auc_avgProb, auc_prc_avgProb\
            = self.getMetrics(softPredList_avgProb, predList_avgProb,
                              labelList)
        print("Using Max Voting - Acc : %.3f ; F1 : %.3f ; AUC(ROC) : %.3f ;"
              "AUC(PRC) : %.3f" % (acc_maxVoting.item(), f1_maxVoting,
                                   auc_maxVoting, auc_prc_maxVoting))
        print("Using Avg Probability - Acc : %.3f ; F1 : %.3f ; AUC(ROC) : "
              "%.3f; AUC(PRC) : %.3f" % (acc_avgProb.item(), f1_avgProb,
                                         auc_avgProb, auc_prc_avgProb))


def runModel(datagen, model, optimizer, class_wts, process, batch_size,
             n_batches, loss_wts):
    '''
    process : 'trn', 'val' or 'tst'
    '''
    running_loss = 0
    pred_list = []
    label_list = []
    soft_pred_list = []
    all_file_list = []
    with trange(n_batches, desc=process, ncols=100) as t:
        for m in range(n_batches):
            data, labels, filenames = datagen.__next__()
            labels_one_hot = utils.get_one_hot(labels).cuda()
            if process == 'trn':
                optimizer.zero_grad()
                model.train()
                pred, aux_pred = model.forward(data)
                pred = F.softmax(pred, 1)
                aux_pred = F.softmax(aux_pred, 1)
                loss = 0
                for i in range(2):
                    loss += loss_wts[0] * utils.weightedBCE(class_wts[i],
                                                            pred[:, i],
                                                            (labels_one_hot
                                                            [:, i]))\
                            + loss_wts[1] * utils.weightedBCE(class_wts[i],
                                                              aux_pred[:, i],
                                                              (labels_one_hot
                                                              [:, i]))
                loss.backward()
                if torch.isnan(loss):
                    pdb.set_trace()
                optimizer.step()
            elif process == 'val' or process == 'tst':
                model.eval()
                with torch.no_grad():
                    pred = F.softmax(model.forward(data), 1)
                    loss = utils.weightedBCE(class_wts[0], pred[:, 0],
                                             labels_one_hot[:, 0])\
                        + utils.weightedBCE(class_wts[1], pred[:, 1],
                                            labels_one_hot[:, 1])
            running_loss += loss
            hard_pred = torch.argmax(pred, 1)
            pred_list.append(hard_pred.cpu())
            soft_pred_list.append(pred.detach().cpu())
            label_list.append(labels.cpu())
            all_file_list += filenames
            t.set_postfix(loss=running_loss.item()/(float(m+1) * batch_size))
            t.update()
        finalLoss = running_loss/(float(m+1) * batch_size)
        # if process != 'trn':
        #     pred_list, soft_pred_list, label_list = utils.test_time_aug(
        #                                                     all_file_list,
        #                                                     soft_pred_list,
        #                                                     label_list, 3)
        acc = utils.globalAcc(pred_list, label_list)
        if not isinstance(pred_list, torch.Tensor):
            f1 = sklearn.metrics.f1_score(torch.cat(label_list),
                                          torch.cat(pred_list), labels=None)
        else:
            f1 = sklearn.metrics.f1_score(label_list, pred_list, labels=None)
        auroc, auprc, fpr_tpr_arr, precision_recall_arr = utils.AUC(
            soft_pred_list, label_list)
        metrics = Metrics(finalLoss, acc, f1, auroc, auprc, fpr_tpr_arr,
                          precision_recall_arr)
        utils.save_preds(soft_pred_list, pred_list, label_list,
                         all_file_list, args.savename, process)
        return metrics


if __name__ == '__main__':
    # Load user inputs
    task = 'mutation'  # 'normal_vs_cancer'
    # class_name = 'CDH1'
    Metrics = namedtuple('Metrics', ['Loss', 'Acc', 'F1', 'AUROC', 'AUPRC',
                                     'fpr_tpr_arr', 'precision_recall_arr'])
    parser = utils.get_argparser()
    args = parser.parse_args()
    class_name = args.classname
    model_class = args.modelClassName
    # Load records for checkpoint logging
    if not args.savename:
        print("Warning! Savename unspecified. No logging will take place."
              "Model will not be saved.")
        best_val_record = None
        log_file = None
    else:
        best_val_record, log_file = utils.init_logging(args.savename)
        with open(os.path.join('logs', best_val_record), 'r') as status_file:
            best_val = float(status_file.readline().strip('\n').split()[-1])
    # Set up data loaders
    aug_list_0 = ['normal', 'mirror', 'rotated', 'gaussNoise']
    aug_list_1 = ['normal', 'mirror', 'rotated', 'gaussNoise']
    trn_data_loader = DataLoader('trn', task, class_name=class_name,
                                 sample_size=16000)
    trn_data_gen = trn_data_loader.data_gen(args.batchSize, aug_list_0,
                                            aug_list_1)
    trn_num_batches = trn_data_loader.get_num_batches(args.batchSize,
                                                      aug_list_0,
                                                      aug_list_1)
    val_data_loader = DataLoader('val', task, class_name=class_name)
    val_data_gen = val_data_loader.data_gen(args.batchSize)
    val_num_batches = val_data_loader.get_num_batches(args.batchSize)
    tst_data_loader = DataLoader('tst', task, class_name=class_name)
    tst_data_gen = tst_data_loader.data_gen(args.batchSize)
    tst_num_batches = tst_data_loader.get_num_batches(args.batchSize)

    samples_per_cls = trn_data_loader.get_label_dist()
    # print(samples_per_cls)
    # samples_per_cls = val_data_loader.get_label_dist()
    # print(samples_per_cls)
    # samples_per_cls = tst_data_loader.get_label_dist()
    # print(samples_per_cls)

    # Init loss function, model and optimizer
    loss_wts = tuple(map(float, args.lossWeights.split(',')))
    class_wts = utils.get_class_balanced_wt(0.9999, samples_per_cls)
    # class_wts = torch.Tensor([1, 1]).cuda()
    model = torchvision.models.inception_v3(pretrained=True, progress=True,
                                            num_classes=2, aux_logits=True,
                                            init_weights=False).cuda()
    model = nn.DataParallel(model)
    # model = torch.load('FINAL/models/tp53_wSelectedPatches_goodModel.pt')
    if args.loadModelFlag:
        success_flag = utils.load_model(args.loadModelFlag, model,
                                        args.savename)
        if success_flag == 0:
            print("Error! Model could not be loaded")
        elif success_flag == 1:
            print("Model loaded successfully")
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learningRate,
                                weight_decay=args.weightDecay)

    # Learning
    for epoch_num in range(args.initEpochNum, args.initEpochNum+args.nEpochs):
        trn_metrics = runModel(trn_data_gen, model, optimizer, class_wts,
                               'trn', args.batchSize, trn_num_batches,
                               loss_wts=loss_wts)
        utils.log_metrics(epoch_num, trn_metrics, 'trn', log_file,
                          args.savename)
        torch.save(model.state_dict(), args.savename+'.pt')
        val_metrics = runModel(val_data_gen, model, optimizer, class_wts,
                               'val', args.batchSize, val_num_batches, None)
        utils.log_metrics(epoch_num, val_metrics, 'val', log_file,
                          args.savename)
        if best_val_record and val_metrics.AUROC > best_val:
            best_val = utils.save_chkpt(best_val_record, best_val,
                                        val_metrics, model, args.savename)
    tst_metrics = runModel(tst_data_gen, model, optimizer, class_wts, 'tst',
                           args.batchSize, tst_num_batches, None)
    utils.log_metrics(0, tst_metrics, 'tst', log_file, args.savename)
    # val_aggregator = Aggregator('val', task, val_data_loader)
    # val_aggregator.aggregate()
    # tst_aggregator = Aggregator('tst', task, tst_data_loader)
    # tst_aggregator.aggregate()

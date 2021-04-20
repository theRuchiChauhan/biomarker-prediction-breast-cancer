import os
import argparse

import numpy as np
import pandas as pd
import torch
import sklearn
import kornia

from augmentTools import kornia_affine, augment_gaussian_noise

#   --------------- Handling models & inputs --------------


def save_preds(soft_pred, hard_pred, label_list, file_list, savename,
               data_type):
    '''
    Save probabilities, predictions, labels along with img names
    to a csv.
    '''
    hard_pred = torch.cat(hard_pred).reshape(-1, 1)
    file_list = np.array(file_list)
    file_list = np.reshape(file_list, (-1, 1))
    soft_pred = torch.cat(soft_pred)
    label_list = torch.cat(label_list).reshape(-1, 1)
    all_data = np.concatenate((file_list, soft_pred.numpy(),
                               hard_pred.numpy(), label_list.numpy()), axis=1)
    np.savetxt(data_type+'_pred_'+savename+'.csv', all_data,
               delimiter=',', fmt='%s')
    print("Saved preds")


def log_metrics(epoch_num, metrics, process, logfile, savename):
    '''
    Print metrics to terminal and save to logfile in a proper format.
    '''
    line = 'Epoch num. {epoch_num:d} \t {process} Loss : {loss_val:.7f};'\
           '{process} Acc : {acc:.3f} ; {process} F1 : {f1:.3f} ;'\
           '{process} AUROC : {auroc:.3f} ; {process} AUPRC : '\
           '{auprc:.3f}\n'.format(epoch_num=epoch_num, process=process,
                                  loss_val=metrics.Loss, acc=metrics.Acc,
                                  f1=metrics.F1, auroc=metrics.AUROC,
                                  auprc=metrics.AUPRC)
    print(line.strip('\n'))
    if logfile:
        with open(os.path.join('logs', logfile), 'a') as f:
            f.write(line)
    np.savetxt('logs/fpr_tpr_' + savename.split('.')[0] + '.csv',
               metrics.fpr_tpr_arr, delimiter=',')
    np.savetxt('logs/precision_recall_' + savename.split('.')[0] + '.csv',
               metrics.precision_recall_arr, delimiter=',')


def load_model(load_model_flag, model, savename):
    '''
    Load saved weights. load_model_flag: main, chkpt or None.
    Sends abort signal if saved model does not exist.
    '''
    try:
        savename = 'FINAL/models/' + savename
        if load_model_flag == 'main':
            model.load_state_dict(torch.load(savename+'.pt'))
            print('Loaded right')
        elif load_model_flag == 'chkpt':
            model.load_state_dict(torch.load('chkpt_'+savename+'.pt'))
        success_flag = 1
    except FileNotFoundError:
        print('Model does not exist! Aborting...')
        success_flag = 0
    return success_flag


def save_chkpt(best_val_record, best_val, metrics, model, savename):
    '''
    Save checkpoint model
    '''
    diff = metrics.AUROC - best_val
    best_val = metrics.AUROC
    with open(os.path.join('logs', best_val_record), 'w') as statusFile:
        statusFile.write('Best AUROC so far: '+str(best_val))
    torch.save(model.state_dict(), 'chkpt_'+savename+'.pt')
    print('Model checkpoint saved since AUROC has improved by '+str(diff))
    return best_val


def init_logging(savename):
    '''
    Create files for storing best metric value and logs if not existing
    already. Returns names of the files.
    '''
    best_val_record = 'best_val_'+savename+'.txt'
    logfile = 'log_'+savename+'.txt'
    if not os.path.exists(os.path.join('logs', best_val_record)):
        os.system('echo "Best F1 so far: 0.0" > ' +
                  os.path.join('logs', best_val_record))
    if not os.path.exists(os.path.join('logs', logfile)):
        os.system('touch '+os.path.join('logs', logfile))
    return best_val_record, logfile


def get_argparser():
    '''
    Set options for argument parser to take hyperparameters.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelClassName", help="name of class for model",
                        type=str)
    parser.add_argument("--classname", help="name of mutation class, ex. CDH1",
                        type=str, required=True)
    parser.add_argument("--savename",  help="Name by which model will be"
                        " saved. Names of logging files depend on this",
                        type=str)
    parser.add_argument("--initEpochNum", help="Serial number of starting"
                        " epoch, for display.", type=int, default=1)
    parser.add_argument("--nEpochs", help="Number of epochs", type=int,
                        default=10)
    parser.add_argument("--batchSize", help="Batch Size", type=int, default=12)
    parser.add_argument("-wd", "--weightDecay", help="Weight decay for"
                        "optimizer", type=float, default='1e-5')
    parser.add_argument("-lr", "--learningRate", help="Learning rate",
                        type=float, default='1e-4')
    parser.add_argument("-lwts", "--lossWeights", help="Weights for main and"
                        "auxiliary loss. Pass as a string in format wt1,wt2"
                        "such that wt1+wt2=1", type=str, default='0.8,0.2')
    parser.add_argument("-loadflg", "--loadModelFlag", help="Whether and "
                        "which model to load. main, chkpt or None"
                        "(not passed)", type=str)
    return parser

#   --------------- Metrics & data tools --------------


def test_time_aug(file_list, soft_pred_list, label_list, nAug):
    final_pred_list = []
    final_label_list = []
    final_soft_pred_list = []
    file_list = [('-').join(name.split('-')[:-1]) for name in file_list]
    soft_pred_list = torch.cat(soft_pred_list)
    label_list = torch.cat(label_list).reshape(-1,)
    df = pd.DataFrame({'file_names': file_list,
                       'soft_pred_list_0': soft_pred_list[:, 0],
                       'soft_pred_list_1': soft_pred_list[:, 1],
                       'label_list': label_list})
    df.groupby(['file_names'])
    for i in range(0, len(file_list), nAug):
        preds = df.iloc[i:i+nAug, 1:-1]
        labels = df.iloc[i, -1]
        agg_pred = np.mean(preds.values, 0)
        final_label_list.append(labels)
        final_soft_pred_list.append(agg_pred)
        final_pred_list.append(np.argmax(agg_pred))
    return (torch.Tensor(np.array(final_pred_list)), torch.Tensor(
            np.array(final_soft_pred_list)), torch.Tensor(
            np.array(final_label_list)))


def get_one_hot(y_arr):
    ''' One Hot encoding for softmax. '''
    y_OH = torch.FloatTensor(y_arr.shape[0], 2)
    y_OH.zero_()
    y_OH.scatter_(1, y_arr, 1)
    return y_OH


def AUC(soft_pred_list, label_list):
    """ Use the probabilities and labels to compute AUROC and AUPRC
    Also returns fpr, tpr which may be used for plotting ROC
    """
    if isinstance(soft_pred_list, list):
        soft_pred_list = np.concatenate(soft_pred_list, 0)
    if isinstance(label_list, list):
        label_list = np.concatenate(label_list, 0)
    fpr, tpr, threshold = sklearn.metrics.roc_curve(label_list,
                                                    soft_pred_list[:, 1],
                                                    pos_label=1)
    auc_roc = sklearn.metrics.auc(fpr, tpr)
    precision, recall, threshold = sklearn.metrics.precision_recall_curve(
        label_list, soft_pred_list[:, 1], pos_label=1)
    auc_prc = sklearn.metrics.auc(recall, precision)
    fpr_tpr_arr = np.array([fpr, tpr])
    precision_recall_arr = np.array([precision, recall])
    return auc_roc, auc_prc, fpr_tpr_arr, precision_recall_arr


def globalAcc(pred_list, label_list):
    ''' Compute accuracy for all batches at once. '''
    if not isinstance(pred_list, torch.Tensor):
        pred_list = torch.cat(pred_list)
    if not isinstance(label_list, torch.Tensor):
        label_list = torch.cat(label_list)
        label_list = label_list[:, 0]
    acc = torch.sum(pred_list == label_list).float()/(pred_list.shape[0])
    return acc


def augment(im, aug_type):
    ''' Augment images as per given augmentation type. '''
    if aug_type == 'normal':
        im = im
    elif aug_type == 'rotated':
        rot_angle = np.random.choice([-90, 90, 180])
        im = kornia_affine(im, rot_angle, 'rotate')
    elif aug_type == 'hsv':
        adjustFactor = np.random.choice([-0.1, -0.075, -0.05, 0.05, 0.075,
                                         0.1])
        hue, sat, value = torch.chunk(kornia.rgb_to_hsv(im), chunks=3, dim=-3)
        adjust_mat = (torch.ones(1, 512, 512)*adjustFactor).cuda()
        hueNew = hue + hue*adjust_mat.cuda()
        hueNew = torch.clamp(hueNew, -2*np.pi, 2*np.pi)
        satNew = sat + sat*adjust_mat.cuda()
        satNew = torch.clamp(satNew, -2*np.pi, 2*np.pi)
        new_im = torch.cat([hueNew, satNew, value], dim=-3)
        im = kornia.hsv_to_rgb(new_im)
    elif aug_type == 'gauss_noise':
        im = augment_gaussian_noise(im)
    elif aug_type == 'mirror':
        im = torch.flip(im, [-2])
    return im

#   --------------- Loss functions --------------

# class BCELossCustom(nn.Module):
#     '''
#     Weighted Binary cross entropy loss. Tested.
#     '''
#     def __init__(self, weight):
#         super(BCELossCustom, self).__init__()
#         self.weight = weight

#     def forward(self, inputs, target):
#         normVal = 1e-24
#         loss = -((self.weight[1] * target) * inputs.clamp(min=normVal).log()
#                  + self.weight[0] * (1 - target)
#                  * (1 - inputs).clamp(min=normVal).log()).mean()
#         return loss


def weightedBCE(weight, pred, target):
    norm_val = 1e-24
    weights = 1 + (weight - 1) * target
    loss = -((weights * target) * pred.clamp(min=norm_val).log() + (1 - target)
             * (1 - pred).clamp(min=norm_val).log()).sum()
    return loss


def get_class_balanced_wt(beta, samples_per_cls, nClasses=2):
    '''
    As per https://towardsdatascience.com/handling-class-imbalanced-data
    -using-a-loss-specifically-made-for-it-6e58fd65ffab
    '''
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * nClasses
    return torch.Tensor(weights).cuda()

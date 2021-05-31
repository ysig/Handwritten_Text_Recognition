from params import *
from network import *
from data.Preprocessing import img_resize

import os
import sys
import torch as torch
from tqdm import tqdm
from torch.autograd import Variable
from skimage import io as img_io
from data.IAM_dataset import iam_main_loader

import copy

params = BaseOptions().parser()


"""
Copied from https://github.com/him4318/Transformer_ocr/blob/master/src/data/evaluation.py
Tool to metrics calculation through data and label (string and string).
 * Calculation from Optical Character Recognition (OCR) metrics with editdistance.
"""

import editdistance
import numpy as np


def ocr_metrics(predicts, ground_truth, k=4):
    """Calculate Character Error Rate (CER), Word Error Rate (WER) and Sequence Error Rate (SER)"""

    if len(predicts) == 0 or len(ground_truth) == 0:
        return (1, 1, 1)

    cer, wer, ser = [], [], []

    for (pd, gt) in zip(predicts, ground_truth):
        pd_cer, gt_cer = list(pd), list(gt)
        dist = editdistance.eval(pd_cer, gt_cer)
        cer.append(dist / (max(len(pd_cer), len(gt_cer))))

        pd_wer, gt_wer = pd.split(' '), gt.split(' ')
        dist = editdistance.eval(pd_wer, gt_wer)
        wer.append(dist / (max(len(pd_wer), len(gt_wer))))

        # pd_ser, gt_ser = [pd], [gt]
        # dist = editdistance.eval(pd_ser, gt_ser)
        # ser.append(dist / (max(len(pd_ser), len(gt_ser))))

    metrics = [wer, cer]#, ser]
    metrics = np.mean(metrics, axis=1)
    return metrics

def iterate_dataset_iam():
    lst = iam_main_loader('test')
    for item in tqdm(lst):
        yield item

def predict(model, data_info, imgH, imgW):
    # Create folder to stock predictions
    model.eval()

    # Go through data folder to make predictions
    decoded, real = [], []
    for img, txt in iterate_dataset_iam():
        # print(filename)
        # Process predictions
        img = img_resize(img, height=imgH, width=imgW, keep_ratio=True)
        img = torch.Tensor(img).float().unsqueeze(0)
        img = Variable(img.unsqueeze(1))
        # print('img shape', img.shape)
        if params.cuda and torch.cuda.is_available():
            img = img.cuda()
        # print(img.type)
        with torch.no_grad():
            pred = model(img)
        pred_size = Variable(torch.LongTensor([pred.size(0)] * img.size(0)))

        # Convert probability output to string
        tdec = pred.argmax(2).permute(1, 0).cpu().numpy().squeeze()
        # print(tdec)
        # print(tdec.ndim)
        # Convert path to label, batch has size 1 here
        if tdec.ndim == 0:
            dec_transcr = ''.join([params.icdict[tdec.item()]]).replace('_', '')
        else:
            tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
            dec_transcr = ''.join([params.icdict[t] for t in tt]).replace('_', '')
        decoded.append(dec_transcr)
        real.append(txt.replace('|', " "))

    wer, cer = ocr_metrics(decoded, real)
    print(f"Test: cer = {cer}, wer = {wer}")    

if __name__ == "__main__":

    MODEL = RCNN(imheight=params.imgH,
                 nc=params.NC,
                 n_conv_layers=params.N_CONV_LAYERS,
                 n_conv_out=params.N_CONV_OUT,
                 conv=params.CONV,
                 batch_norm=params.BATCH_NORM,
                 max_pool=params.MAX_POOL,
                 n_r_layers=params.N_REC_LAYERS,
                 n_r_input=params.N_REC_INPUT,
                 n_hidden=params.N_HIDDEN,
                 n_out=len(params.alphabet),
                 bidirectional=params.BIDIRECTIONAL,
                 feat_extractor=params.feat_extractor,
                 dropout=params.DROPOUT)
    #
    # MODEL.load_state_dict(torch.load('/media/vn_nguyen/hdd/hux/Results/08-19_12:48:18/IAM_model_imgH64.pth'))
    # print(MODEL)
    # torch.save(MODEL, '/home/loisonv/Text_Recognition/trained_networks/ICFHR2014_model_imgH32.pth')

    # MODEL = torch.load('/home/hux/HTR/trained_networks/IAM_model_imgH64.pth')
    # MODEL.load_state_dict(torch.load(params.model_path))
    PP = os.path.dirname(__file__)
    model_file = os.path.join(PP, 'trained_networks', 'IAM_model_imgH64.pth')
    MODEL.load_state_dict(torch.load(model_file))

    if params.cuda and torch.cuda.is_available():
        MODEL = MODEL.cuda()
    DATA_LOC = params.data_path

    predict(MODEL, DATA_LOC, imgH=params.imgH, imgW=params.imgW)

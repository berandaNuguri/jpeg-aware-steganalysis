from jpeg_aware_steganalysis._paths import data_path as _checkout_data_path, model_path as _checkout_model_path
import os
import numpy as np
import torch
import torch.nn as nn
from Model import SF3Net
from Onehot_cross_attention import OneHotLWENet
from .Custom_Dataloader import return_dataloader
from torch.nn import functional as F
from Utils import AverageMeter, accuracy_2cls
from sklearn.metrics import roc_curve, auc
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.classification.auc import AUC
from torchmetrics.functional.classification.auc import _auc_update


def check_nans(*args):
    for ar in args:
        if np.isnan(ar).any():
            return True
    return False

def wauc(y_true, y_pred):
    """
    https://www.kaggle.com/anokas/weighted-auc-metric-updated
    """
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1, drop_intermediate=False)
    if check_nans(fpr, tpr):
        return np.nan
    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])
    normalization = np.dot(areas, weights)

    competition_metric = 0
    for idx, weight in enumerate(weights):
        y_min = tpr_thresholds[idx]
        y_max = tpr_thresholds[idx + 1]
        mask = (y_min < tpr) & (tpr < y_max)
        if mask.sum()==0:
            continue
        x_padding = np.linspace(fpr[mask][-1], 1, 100)
        x = np.concatenate([fpr[mask], x_padding])
        y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])
        y = y - y_min
        score = auc(x, y)
        submetric = score * weight
        competition_metric += submetric

    return competition_metric/normalization\


class wAUC(AUC):
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.
        """
        preds = 1 - F.softmax(preds.double(), dim=1)[:,0]
        target = torch.clip(target, min=0, max=1)
        x, y = _auc_update(preds, target)
        self.x.append(x)
        self.y.append(y)
    def compute(self):
        """Computes MD5 based on inputs passed in to ``update`` previously."""
        x = dim_zero_cat(self.x)
        y = dim_zero_cat(self.y)
        return wauc(y, x)


def single_epoch_valid(model, criterion, val_set):
    model.eval()
    auc_avg = wAUC()
    acc_avg = AverageMeter()
    loss_avg = AverageMeter()
    for batch_idx, data in enumerate(val_set):
        images, images_dct, qtables, labels = data
        images_cover, images_stego = images.split(1, dim=1)
        cover_dct, stego_dct = images_dct.split(1, dim=1)
        labels_cover, labels_stego = labels.split(1, dim=1)
        qtables_cover, qtables_stego = qtables.split(1, dim=1)
        images = torch.cat((images_cover, images_stego), 0).squeeze(1).cuda()
        images_dct = torch.cat((cover_dct, stego_dct), 0).squeeze(1).cuda()
        labels = torch.cat((labels_cover, labels_stego), 0).squeeze(1).cuda()
        qtables = torch.cat((qtables_cover, qtables_stego), 0).cuda()
        b = len(images)
        with torch.no_grad():
            outputs = model(images, images_dct, qtables)
            loss = criterion(outputs, labels)
            loss_avg.update(loss.item(), b)
            acc_avg.update(accuracy_2cls(outputs, labels).item(), b)
            auc_avg.update(outputs.cpu(), labels.cpu())
    return acc_avg.avg, loss_avg.avg, auc_avg.compute()

if __name__ == '__main__':
    torch.cuda.set_device(1)

    cover_dir = str(_checkout_data_path('JPEG/Cover/ALASKA_v2_TIFF_256_GrayScale_QF85'))
    stego_dir = str(_checkout_data_path('JPEG/Stego/Alaska_JUNIWARD_256/ALASKA_v2_TIFF_256_GrayScale_QF85_JUNIWARD-0.4-20000'))
    test_file = str(_checkout_data_path('JPEG/Config/Alaska_split_14000_1000_5000/test.txt'))

    model = SF3Net()
    model_path = str(_checkout_model_path('sf3net/ALASKA_256_JUNIWARD_0.4_QF85ckpt'))
    state_dict = torch.load(model_path, map_location='cpu')

    model.load_state_dict(state_dict)
    model.cuda()
    criterion = nn.CrossEntropyLoss()

    #%%
    test_set = return_dataloader(cover_dir, stego_dir, test_file, 'test')
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=16, shuffle=False, num_workers=4)
    test_acc, test_loss, auc = single_epoch_valid(model, criterion, test_loader)

    print("test acc is {:.4f}, loss is {:.4f}, auc is {:.4f}".format(test_acc, test_loss, auc))
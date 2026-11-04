import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import os
import shutil
import torch

import numpy as np

from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import torch.nn.functional as F
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


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy_2cls(output, target):
    batch_size = target.size(0)
    assert batch_size == output.size(0)

    _, prediction = torch.max(output.data, 1)
    accuracy = prediction.eq(target.data).sum()*100.0/(batch_size)
    return accuracy


# Adjust BN estimated value
def adjust_bn_stats(model, train_loader):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        images, images_dct, qtables, labels = data
        images_cover, images_stego = images.split(1, dim=1)
        cover_dc, stego_dc = images_dct.split(1, dim=1)
        qtables_cover, qtables_stego = qtables.split(1, dim=1)
        images = torch.cat((images_cover, images_stego), 0).squeeze(1).cuda()
        images_dct = torch.cat((cover_dc, stego_dc), 0).squeeze(1).cuda()
        qtables = torch.cat((qtables_cover, qtables_stego), 0).cuda()
        with torch.no_grad():
            outputs = model(images, images_dct, qtables)


def single_epoch_train(model, optimizer, criterion, train_set):
    model.train()
    acc_avg = AverageMeter()
    loss_avg = AverageMeter()

    # train_bar = tqdm(train_set)
    for batch_idx, data in enumerate(train_set):
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

        optimizer.zero_grad()
        outputs = model(images, images_dct, qtables)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_avg.update(loss.item(), b)
        acc_avg.update(accuracy_2cls(outputs, labels).item(), b)

        acc_avg.update(accuracy_2cls(outputs, labels).item(), b)

        # train_bar.set_postfix(loss="{:.4f}".format(loss.item()))

    return acc_avg.avg, loss_avg.avg


def single_epoch_valid(model, criterion, val_set, auc_avg):
    model.eval()
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


def train(model, optimizer, scheduler, criterion, train_loader, valid_loader, test_loader, model_path, log_dir, logging, args):
    max_acc = 0
    best_epoch = 0
    auc_avg = wAUC()
    train_time = time.time()
    for epoch in range(args.total_epochs):
        if args.start_nopc_epoch <= epoch:
            if hasattr(train_loader.dataset, 'datasets'):
                for dataset in train_loader.dataset.datasets:
                    dataset.nopc()
            else:
                train_loader.dataset.nopc()
        if args.start_nopc_epoch == epoch:
            logging.info("Start no pc !")

        epoch_start = time.time()
        train_acc, train_loss = single_epoch_train(model, optimizer, criterion, train_loader)  # train
        if scheduler is not None:
            scheduler.step()
        adjust_bn_stats(model, train_loader)
        val_acc, val_loss, auc = single_epoch_valid(model, criterion, valid_loader, auc_avg)  # validation
        auc_avg.reset()
        print("Epoch:{}, train_acc:{:.3f}, train_loss:{:.3f}, val_acc:{:.3f}, val_loss:{:.3f}".format(
            epoch, train_acc, train_loss, val_acc, val_loss))

        if max_acc < val_acc:
            max_acc = val_acc
            torch.save(model.state_dict(), model_path + '/' + str(epoch+1) + 'ckpt')
            best_epoch = epoch + 1
            print('save model_epoch{}'.format(epoch + 1))
        with open(log_dir + '/train.csv', 'a+') as f_val:     # 把测试集准确率和损失写入csv文件
            if epoch == 0:
                f_val.write('%s, %s, %s, %s, %s, %s\n' % ('epoch', 'train_acc', 'valid_acc', 'train_loss', 'valid_loss', 'epoch_time'))
            f_val.write('%d,%.4f,%.4f, %.4f, %.4f, %ds\n' % (epoch+1, train_acc, val_acc, train_loss, val_loss, time.time()-epoch_start))

    end_train_time = time.time()
    auc_avg.reset()
    model.load_state_dict(torch.load(model_path + '/' + str(best_epoch) + 'ckpt'))
    test_time = time.time()
    test_acc, test_loss, test_auc = single_epoch_valid(model, criterion, test_loader, auc_avg)  # test
    end_test_time = time.time()
    print('test_acc:{:.4f}, test_loss:{:.4f}, test_auc:{:.4f}'.format(test_acc, test_loss, test_auc))
    with open(log_dir + '/time.txt', 'a+') as f_val:
        f_val.write('train_time: %.2f hours, test_time: %.2f s\n' % ((end_train_time-train_time)/10 *180/3600, end_test_time-test_time))
    with open(log_dir + '/test.txt', 'a+') as f_val:  # 把测试集准确率和损失写入csv文件
        f_val.write('Best_epoch: %d, test_acc: %.4f, test_loss: %.4f, test_auc:%.4f\n' % (best_epoch, test_acc,
                                                                                          test_loss, test_auc))
from jpeg_aware_steganalysis._paths import data_path as _checkout_data_path
import csv
import os
import scipy

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import jpegio as jio
import pandas as pd
import numpy as np
import random
from random import random as rand
from scipy import fftpack
from numpy.lib.stride_tricks import as_strided


def train_transforms(data, data_dct, qtable):
    # data = np.expand_dims(data, axis=1)
    data = data.astype(np.float32)
    dct = data_dct.astype(np.float32)
    qtable = qtable.astype(np.float32)
    return torch.from_numpy(data), torch.from_numpy(dct), torch.from_numpy(qtable)


def segmented_stride(M, fun, blk_size=(8, 8), overlap=(0,0)):
    # This is some complex function of blk_size and M.shape
    B = block_view(M, block=blk_size)
    B[:, :, :, :] = fun(B)
    return M


def block_view(A, block=(8, 8)):
    """Provide a 2D block view to 2D array. No error checking made.
    Therefore meaningful (as implemented) only for blocks strictly
    compatible with the shape of A."""
    # simple shape and strides computations may seem at first strange
    # unless one is able to recognize the 'tuple additions' involved ;-)
    shape = (A.shape[0] // block[0], A.shape[1] // block[1]) + block
    strides = (block[0] * A.strides[0], block[1] * A.strides[1]) + A.strides
    return as_strided(A, shape=shape, strides=strides)


def decompress_structure(S):
    # Decompress DCT coefficients C using quantization table Q
    H = S.coef_arrays[0].shape[0]
    W = S.coef_arrays[0].shape[1]
    n = len(S.coef_arrays)
    assert H % 8 == 0, 'Wrong image size'
    assert W % 8 == 0, 'Wrong image size'
    I = np.zeros((H, W, n), dtype=np.float64)  # Returns Y, Cb and Cr
    for i in range(n):
        Q = S.quant_tables[S.comp_info[i].quant_tbl_no]
        # this multiplication is done on integers
        fun = lambda x: np.multiply(x, Q)
        C = np.float64(segmented_stride(S.coef_arrays[i], fun))
        fun = lambda x: fftpack.idct(fftpack.idct(x, norm='ortho', axis=2), norm='ortho', axis=3) + 128
        I[:, :, i] = segmented_stride(C, fun)
    return I

def decompress_structure1(S, Q):
    # Decompress DCT coefficients C using quantization table Q
    coef_arrays = S
    H = coef_arrays.shape[0]
    W = coef_arrays.shape[1]
    n = 1
    assert H % 8 == 0, 'Wrong image size'
    assert W % 8 == 0, 'Wrong image size'
    I = np.zeros((H, W, n), dtype=np.float64)  # Returns Y, Cb and Cr
    # this multiplication is done on integers
    fun = lambda x: np.multiply(x, Q)
    C = np.float64(segmented_stride(coef_arrays, fun))
    fun = lambda x: fftpack.idct(fftpack.idct(x, norm='ortho', axis=2), norm='ortho', axis=3) + 128
    I[:, :, 0] = segmented_stride(C, fun)
    return I


def rot_and_flip_jpeg(S, rot, flip=False):
    """Lossless rotation and flips in the DCT domain"""
    assert S.coef_arrays[0].shape[0] % 8 == 0, 'Image height is not a multiple of 8!'
    assert S.coef_arrays[0].shape[1] % 8 == 0, 'Image width is not a multiple of 8!'
    I = np.empty([S.coef_arrays[0].shape[0], S.coef_arrays[0].shape[1], S.image_components], dtype=np.int32)
    for ch in range(I.shape[-1]):
        C = np.copy(S.coef_arrays[ch])
        C = np.rot90(C, k=rot)
        fun = lambda x: np.rot90(x, k=4-rot, axes=(2, 3))
        C = segmented_stride(C, fun)
        # transpose
        if rot in [1, 3]:
            fun = lambda x: np.copy(np.transpose(x, axes=[0, 1, 3, 2]))
            C = segmented_stride(C, fun)
            for q in range(len(S.quant_tables)):
                S.quant_tables[q] = S.quant_tables[q].T
        # multiply even rows by -1
        if rot in [1, 2]:
            C[1::2, :] *= -1
        # multiply even columns by -1
        if rot in [2, 3]:
            C[:, 1::2] *= -1
        # flip
        if flip == True:
            C = np.fliplr(C)
            fun = lambda x: np.flip(x, axis=3)
            C = segmented_stride(C, fun)
            C[:, 1::2] *= -1
        S.coef_arrays[ch] = C[:, :]
    return S


def rot_and_flip_jpeg1(coef_arrays, quant_tables, rot, flip=False):
    """Lossless rotation and flips in the DCT domain"""

    assert coef_arrays.shape[0] % 8 == 0, 'Image height is not a multiple of 8!'
    assert coef_arrays.shape[1] % 8 == 0, 'Image width is not a multiple of 8!'
    I = np.empty([coef_arrays.shape[0], coef_arrays.shape[1], 1], dtype=np.int32)

    C = np.copy(coef_arrays).astype(np.int16)
    C = np.rot90(C, k=rot)
    fun = lambda x: np.rot90(x, k=4-rot, axes=(2, 3))
    C = segmented_stride(C, fun)
    # transpose
    if rot in [1, 3]:
        fun = lambda x: np.copy(np.transpose(x, axes=[0, 1, 3, 2]))
        C = segmented_stride(C, fun)
        for q in range(len(quant_tables)):
            quant_tables[q] = quant_tables[q].T
    # multiply even rows by -1
    if rot in [1, 2]:
        C[1::2, :] *= -1
    # multiply even columns by -1
    if rot in [2, 3]:
        C[:, 1::2] *= -1
    # flip
    if flip == True:
        C = np.fliplr(C)
        fun = lambda x: np.flip(x, axis=3)
        C = segmented_stride(C, fun)
        C[:, 1::2] *= -1
    coef_arrays[:, :] = C[:, :]
    return coef_arrays


def abs_bounded_onehot(I, T=5):
    """Threshold + abs + onehot encoding"""
    I = np.abs(I)
    I = np.clip(I, a_min=0, a_max=T)
    I_oh = np.zeros((I.shape) + (T+1,))
    for ch in range(I.shape[-1]):
        I_oh[:, :, ch, :] = (I[:, :, ch, np.newaxis] == range(T+1))
    return I_oh.reshape((I.shape[0], I.shape[1], -1))


def clip_bounded_onehot(I, T=5):
    """Threshold + abs + onehot encoding"""
    I = np.clip(I, a_min=-T, a_max=T)
    I_oh = np.zeros((I.shape) + (2*T+1,))
    for ch in range(I.shape[-1]):
        I_oh[:, :, ch, :] = (I[:, :, ch, np.newaxis] == np.arange(-T, T+1))
    return I_oh.reshape((I.shape[0], I.shape[1], -1))


def abs_bounded_onehot_signed(I, T=5):
    """Encode values with the same absolute value in the same channel, distinguishing signs."""
    I = np.clip(I, a_min=-T, a_max=T)
    I_oh = np.zeros(I.shape + (T + 1,))

    for ch in range(I.shape[-1]):
        for t in range(-T, T + 1):
            if t < 0:
                I_oh[:, :, ch, abs(t)] += (I[:, :, ch] == t) * -1
            else:
                I_oh[:, :, ch, abs(t)] += (I[:, :, ch] == t) * 1

    return I_oh.reshape((I.shape[0], I.shape[1], -1))


class MyDataSet(Dataset):
    def __init__(self, sample_file='', transform=True, rot=False, T=10):
        self.sample_file = sample_file
        self.transform = transform
        self.rot = rot
        self.T = T
        self.cover_list, self.stego_list = self.read_file(sample_file)
        self.data_num = len(self.cover_list)

    def __preprocess_strcture(self, tmp):
        image_dct = np.dstack(tmp.coef_arrays).astype(np.float32)
        image = decompress_structure(tmp).astype(np.float32)
        image = image.transpose(2, 0, 1)
        image_dct = np.clip(np.abs(image_dct), 0, 10).astype(np.float32)
        image_dct = image_dct.transpose(2, 0, 1)

        return image, image_dct

    def __len__(self):
        return self.data_num

    def __getitem__(self, item):
        file_index = int(item)
        cover_path, _ = self.cover_list[file_index]
        stego_path, _ = self.stego_list[file_index]
        
        assert os.path.exists(cover_path), "%s is not exist" % (cover_path)
        assert os.path.exists(stego_path), "%s is not exist" % (stego_path)
        if self.rot:
            rot = random.randint(0, 3)
            flip = random.random() < 0.5
        else:
            rot = 0
            flip = False

        tmp_cover = jio.read(cover_path)
        tmp_stego = jio.read(stego_path)
        tmp_cover = rot_and_flip_jpeg(tmp_cover, rot, flip)
        tmp_stego = rot_and_flip_jpeg(tmp_stego, rot, flip)

        cover, cover_dct = self.__preprocess_strcture(tmp_cover)
        stego, stego_dct = self.__preprocess_strcture(tmp_stego)
        qtable_cover = tmp_cover.quant_tables[tmp_cover.comp_info[0].quant_tbl_no]
        qtable_stego = tmp_stego.quant_tables[tmp_stego.comp_info[0].quant_tbl_no]

        labels = np.array([0, 1], dtype=np.int64)
        if self.transform:
            cover, cover_dct, qtable_cover = train_transforms(cover, cover_dct, qtable_cover)
            stego, stego_dct, qtable_stego = train_transforms(stego, stego_dct, qtable_stego)

        labels = torch.from_numpy(labels).long()
        return (torch.stack([cover, stego]), torch.stack([cover_dct, stego_dct]),
                torch.stack([qtable_cover, qtable_stego]), labels)
        
    def read_file(self, listfile):
        df = pd.read_csv(listfile)

        names = df['image_name'].dropna().astype(str).apply(str.strip).tolist() if 'image_name' in df.columns else []
        benchs = df['bench'].dropna().astype(str).apply(str.strip).tolist() if 'bench' in df.columns else []
        steg_algo = self.sample_file.split('/')[-1].split('_')[0] + '_' + self.sample_file.split('/')[-1].split('_')[1]
        
        stego_list = [os.path.join(str(_checkout_data_path('')), bench, 'Stego', steg_algo, image_name) for bench, image_name in zip(benchs, names)]
        cover_list = [os.path.join(str(_checkout_data_path('')), bench, 'Cover', image_name) for bench, image_name in zip(benchs, names)]

        cover_path_list = []
        stego_path_list = []
        for cover_file, stego_file in zip(cover_list, stego_list):
            cover_path_list.append([cover_file, 0])
            stego_path_list.append([stego_file, 1])
        return cover_path_list, stego_path_list

    def nopc(self):
        # 在保证每批类别balance的情况下，去掉成对约束
        random.shuffle(self.cover_list)
        random.shuffle(self.stego_list)


def return_dataloader(sample_file, type, rot=False):
    print('Loading_' + type + '_data')
    pick_dataset = MyDataSet(sample_file, rot=rot, T=10)
    pick_len = 2 * pick_dataset.__len__()
    print("The number of "+type+" is {}".format(pick_len))
    print('pick data end')
    return pick_dataset
import os
import sys
import jpeglib
import cv2
import numpy as np
import glob
from typing import Dict, Tuple


def dct_step_size(channel, samp_factor, orient="vh"):
    """
    채널별 step size (픽셀 단위).
    orient = 'vh' (기본, jpeglib 스타일) / 'hv'
    """
    ch2idx = {'Y':0, 'Cb':1, 'Cr':2}
    c = ch2idx[channel]

    if orient == "vh":
        vY, hY = samp_factor[0]
        vC, hC = samp_factor[c]
    else:  
        hY, vY = samp_factor[0]
        hC, vC = samp_factor[c]

    assert hY % hC == 0 and vY % vC == 0, \
        f"Invalid sampling ratio {channel}: Y=({vY},{hY}), C=({vC},{hC})"

    step_h = 8 * (hY // hC)
    step_v = 8 * (vY // vC)

    return step_h, step_v

def align_boxes(x1,y1,x2,y2,channel,samp_factor, orient="vh"):
    sh, sv = dct_step_size(channel, samp_factor, orient)
    bh0 = x1 // sh
    bv0 = y1 // sv
    bh1 = (x2 + sh - 1) // sh
    bv1 = (y2 + sv - 1) // sv
    
    return (bv0,bv1),(bh0,bh1)

def upsampling_ratio(samp_factor, channel, orient="vh"):
    if orient=="vh":
        vY,hY = samp_factor[0]
        vC,hC = samp_factor[1 if channel=="Cb" else 2]
    else: # hv
        hY,vY = samp_factor[0]
        hC,vC = samp_factor[1 if channel=="Cb" else 2]
    rx = hY//max(1,hC)
    ry = vY//max(1,vC)
    return rx,ry

def qt_to_nparray(im, channel):
    qt = im.qt
    if len(qt) >= 3:
        idx = {'Y':0,'Cb':1,'Cr':2}[channel]
        return np.array(qt[idx],dtype=np.float32)
    elif len(qt)==2:
        return np.array(qt[0 if channel=='Y' else 1],dtype=np.float32)
    elif len(qt)==1:
        return np.array(qt[0],dtype=np.float32)
    else:
        raise ValueError(f"Unexpected qt len {len(qt)}")

def blocks_to_plane(blocks_hw88):
    Hb,Wb = blocks_hw88.shape[:2]
    return blocks_hw88.transpose(0,2,1,3).reshape(Hb*8,Wb*8)

def roi_qdct_mapping(raw, roi_boxes, orient):
    qdct_list, qt_list = [], []

    for (x1,y1,x2,y2) in roi_boxes:
        samp = raw.samp_factor
        # --- Y ---
        vsl,hsl = align_boxes(x1,y1,x2,y2,'Y',samp,orient)
        if orient == 'vh':
            y_qdct = raw.Y[vsl[0]:vsl[1],hsl[0]:hsl[1], :, :].astype(np.float32)
        elif orient == 'hv':
            y_qdct = raw.Y[hsl[0]:hsl[1],vsl[0]:vsl[1], :, :].astype(np.float32)

        Hy,Wy = y_qdct.shape[:2]
        y_qdct = blocks_to_plane(y_qdct)

        # --- Cb, Cr ---
        cb_qdct,cr_qdct=None,None
        for ch in ('Cb','Cr'):
            vsl_c,hsl_c=align_boxes(x1,y1,x2,y2,ch,samp,orient)
            if orient=='vh':
                c_qdct = getattr(raw,ch)[vsl_c[0]:vsl_c[1],hsl_c[0]:hsl_c[1], :, :].astype(np.float32)
            elif orient=='hv':
                c_qdct = getattr(raw,ch)[hsl_c[0]:hsl_c[1],vsl_c[0]:vsl_c[1], :, :].astype(np.float32)
            rx,ry = upsampling_ratio(samp,ch,orient)
            if ry>1: c_qdct=np.repeat(c_qdct,ry,axis=0)
            if rx>1: c_qdct=np.repeat(c_qdct,rx,axis=1)
            c_qdct = c_qdct[:Hy,:Wy]
            c_qdct = blocks_to_plane(c_qdct)
            if ch=='Cb': cb_qdct=c_qdct
            else: cr_qdct=c_qdct

        h_min,w_min = min(cb_qdct.shape[0],cr_qdct.shape[0]),min(cb_qdct.shape[1],cr_qdct.shape[1])
        cb_qdct=cb_qdct[:h_min,:w_min]; cr_qdct=cr_qdct[:h_min,:w_min]; y_qdct=y_qdct[:h_min,:w_min]
        qdct=np.stack([y_qdct,cb_qdct,cr_qdct],axis=-1)

        # QTable
        qtY=qt_to_nparray(raw,'Y').reshape(8,8)
        if len(raw.qt)>=3:
            qtCb=qt_to_nparray(raw,'Cb').reshape(8,8)
            qtCr=qt_to_nparray(raw,'Cr').reshape(8,8)
            qt=np.stack([qtY,qtCb,qtCr],axis=0)
        elif len(raw.qt)==2:
            qtC=qt_to_nparray(raw,'Cb').reshape(8,8)
            qt=np.stack([qtY,qtC,qtC],axis=0)
        else:
            qt=np.stack([qtY,qtY,qtY],axis=0)

        qdct_list.append(qdct.astype(np.float32))
        qt_list.append(qt.astype(np.float32))

    return qdct_list, qt_list

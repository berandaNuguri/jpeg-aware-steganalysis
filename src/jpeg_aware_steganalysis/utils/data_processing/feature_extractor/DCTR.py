import os
import math
import time
import jpeglib
import numpy as np
from scipy.signal import convolve2d
from scipy.fftpack import idct

def idct2(block: np.ndarray) -> np.ndarray:
    """2D IDCT (orthonormal)."""
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def DCTR(image_path: str, coords: tuple=None, QF: float=75) -> np.ndarray:
    """
    JPEG 이미지의 DCTR 피처를 계산.
      - image_path: JPEG 파일 경로
      - QF       : 양자화 품질 인자
      - coords   : (x0, y0, x1, y1) 픽셀 단위 좌표. None이면 전체 이미지.
    """
    T = 4

    # 1) q 계산
    if QF < 50:
        q = min(8 * (50.0 / QF), 100.0)
    else:
        q = max(8 * (2.0 - (QF / 50.0)), 0.2)

    # 2) DCT basis A 생성
    k = np.arange(8); l = np.arange(8)
    k, l = np.meshgrid(k, l, indexing='ij')
    A = 0.5 * np.cos(((2*k + 1) * l * np.pi) / 16)
    A[0, :] /= np.sqrt(2)
    A = A.T  # shape (8,8)

    # 3) 대칭 위치 그룹 생성
    merged_coords = []
    for i in range(1, 6):
        for j in range(1, 6):
            group = np.array([
                [i,     j],
                [i, 10-j],
                [10-i,  j],
                [10-i, 10-j]
            ])
            group = group[(group < 9).all(axis=1)]
            group = np.unique(group, axis=0) - 1
            merged_coords.append(group)

    # 4) JPEG DCT 계수 & 양자화 테이블 읽기
    ij = jpeglib.read_dct(image_path)
    ji = jpeglib.to_jpegio(ij)
    coef_full = ji.coef_arrays[0]   # shape=(H_blocks*8, W_blocks*8)
    quant     = ji.quant_tables[0]  # shape=(8,8)

    # 5) ROI(coords)가 주어지면 픽셀→블록 경계로 맞춰 슬라이싱
    if coords is not None:
        x0, y0, x1, y1 = coords
        px0 = (x0 // 8) * 8
        py0 = (y0 // 8) * 8
        px1 = math.ceil(x1 / 8) * 8
        py1 = math.ceil(y1 / 8) * 8
        coef = coef_full[py0:py1, px0:px1]
    else:
        coef = coef_full

    H, W = coef.shape

    # 6) 블록별 역양자화 + IDCT → 공간 도메인 영상 복원
    I_spatial = np.zeros((H, W), dtype=np.float32)
    for i in range(0, H, 8):
        for j in range(0, W, 8):
            block = coef[i:i+8, j:j+8] * quant
            I_spatial[i:i+8, j:j+8] = idct2(block)
    I_spatial += 128.0

    # 7) 피처 벡터 할당
    modeFeaDim = len(merged_coords) * (T + 1)
    F = np.zeros(64 * modeFeaDim, dtype=np.float32)
    idx = 0

    # 8) 64개 DCT 모드 순회
    for mode_r in range(8):
        for mode_c in range(8):
            DCTbase = np.outer(A[:, mode_r], A[:, mode_c])
            R = convolve2d(I_spatial - 128.0, DCTbase, mode='valid')
            R = np.abs(np.round(R / q))
            R[R > T] = T

            # 9) 그룹별 히스토그램
            for grp in merged_coords:
                hist_vec = np.zeros(T+1, dtype=np.float32)
                for (r_shift, c_shift) in grp:
                    sub = R[r_shift::8, c_shift::8]
                    h, _ = np.histogram(sub, bins=np.arange(T+2))
                    hist_vec += h
                total = hist_vec.sum()
                if total > 0:
                    hist_vec /= total
                F[idx:idx+T+1] = hist_vec
                idx += T+1

    return F

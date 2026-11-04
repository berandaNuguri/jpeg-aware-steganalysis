import sys
import os
import cv2
import numpy as np
import jpeglib
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F

from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset

from jpeg_aware_steganalysis.utils.data_processing.roi import spatial_roi, nzac_roi, juniward_roi, random_roi, calibration_roi, inconsist_roi
from jpeg_aware_steganalysis.utils.data_processing.image_processing.jpeg_to_qdct import roi_qdct_mapping

class DefaultDataset(Dataset):
    def __init__(self, df, root_dir='', args=None, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.args = args

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.df.iloc[idx]['image_path'])
        algorithm = self.df.iloc[idx]['algorithm']
        
        # 0: Cover, 1: Stego
        if algorithm == 'cover' or algorithm == 'Cover':
            label = 0       
        else:
            label = 1   

        img = cv2.imread(image_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            img = self.transform(img)

        return img, label

class PairedPatchDataset(Dataset):
    def __init__(self, df, root_dir='', transform=None, args=None):
        self.df           = df
        self.root_dir     = root_dir
        self.transform    = transform
        self.args         = args
        self._box_cache   = {}

    def __len__(self):
        return len(self.df)

    def _build_metadata(self, img_path, row):
        lp = img_path.lower()
        if lp.endswith('.jpg') or lp.endswith('.jpeg'):
            metadata = [row['qf'], row['subsampling'], row['width'], row['height']]
        elif lp.endswith('.png') or lp.endswith('.bmp') or lp.endswith('.tiff'):
            metadata = [row['width'], row['height']]

        qtable = Image.open(img_path).quantization
        metadata.append(np.array(qtable[0]))    
    
        return metadata

    def _extract_boxes(self, key, img, img_path, residual=None):
        if key in self._box_cache and 'npatch' not in self.args.roi_method:
            return self._box_cache[key]

        roitype = self.args.roi_method

        if 'nzac' in roitype:
            boxes = nzac_roi.extract_patch_coords(self.args, img, img_path)
        elif 'juniward' in roitype:
            boxes = juniward_roi.extract_patch_coords(self.args, Image.open(img_path))
        elif 'wavelet' in roitype:
            boxes = spatial_roi.extract_patch_coords(self.args, img)
        elif 'rand' in roitype:
            boxes = random_roi.extract_patch_coords(self.args, img, residual)
        elif 'calibration' in roitype:
            boxes = calibration_roi.extract_patch_coords(self.args, img_path)
        elif 'inconsist' in roitype:
            boxes = inconsist_roi.extract_patch_coords(self.args, img, img_path)
        else:
            raise ValueError(f"Unknown ROI type: {roitype}")

        if 'rand' not in roitype:
            self._box_cache[key] = boxes
        return boxes

    def _image_to_patches(self, img, boxes):
        patches = []
        for (x1, y1, x2, y2) in boxes:
            patch = img[y1:y2, x1:x2, :]
            if self.transform:
                patch = self.transform(patch)
            patches.append(patch)

        return torch.stack(patches, dim=0)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = row['image_name']

        base_dir = os.path.join(self.root_dir, self.args.data)
            
        cover_path = os.path.join(base_dir, 'Cover', image_name)    
        stego_path = os.path.join(base_dir, 'Stego', self.args.stego_method, image_name)

        if not os.path.exists(cover_path):
            raise FileNotFoundError(f"Cover not found: {cover_path}")
        if not os.path.exists(stego_path):
            raise FileNotFoundError(f"Stego not found: {stego_path}")

        try:
            cover_img = cv2.imread(cover_path, cv2.IMREAD_COLOR)
            cover_img = cv2.cvtColor(cover_img, cv2.COLOR_BGR2RGB)

        except Exception as e:
            raise RuntimeError(f"Error reading cover image {cover_path}: {e}")

        try:
            stego_img = cv2.imread(stego_path, cv2.IMREAD_COLOR)
            stego_img = cv2.cvtColor(stego_img, cv2.COLOR_BGR2RGB)

        except Exception as e:
            raise RuntimeError(f"Error reading stego image {stego_path}: {e}")

        
        residual = np.abs(cover_img - stego_img)
        
        cover_boxes = self._extract_boxes(f"{image_name}|cover", cover_img, cover_path, 'random')
        cover_patches = self._image_to_patches(cover_img, cover_boxes)
        
        stego_boxes = self._extract_boxes(f"{image_name}|stego", stego_img, stego_path, residual)
        stego_patches = self._image_to_patches(stego_img, stego_boxes)

        # --- Metadata ---
        metadata = self._build_metadata(cover_path, row)
        
        data = np.stack([cover_patches, stego_patches])
        label = torch.cat([torch.zeros(1, dtype=torch.long), torch.ones(1, dtype=torch.long)], dim=0)
        
        return data, label, metadata
    
class PairedYCbCrPatchDataset(Dataset):
    def __init__(self, df, root_dir='', transform=None, args=None):
        self.df           = df
        self.root_dir     = root_dir
        self.transform    = transform
        self.args         = args
        self._box_cache   = {}

    def __len__(self):
        return len(self.df)

    def _build_metadata(self, img_path, row):
        lp = img_path.lower()
        if lp.endswith('.jpg') or lp.endswith('.jpeg'):
            metadata = [row['qf'], row['subsampling'], row['width'], row['height']]
        elif lp.endswith('.png') or lp.endswith('.bmp') or lp.endswith('.tiff'):
            metadata = [row['width'], row['height']]

        return metadata

    def _extract_boxes(self, key, img, img_path, residual=None):
        if key in self._box_cache and 'npatch' not in self.args.roi_method:
            return self._box_cache[key]

        roitype = self.args.roi_method

        if 'nzac' in roitype:
            boxes = nzac_roi.extract_patch_coords(self.args, img, img_path)
        elif 'juniward' in roitype:
            boxes = juniward_roi.extract_patch_coords(self.args, Image.open(img_path))
        elif 'wavelet' in roitype:
            boxes = spatial_roi.extract_patch_coords(self.args, img)
        elif 'rand' in roitype:
            boxes = random_roi.extract_patch_coords(self.args, img, residual)
        elif 'calibration' in roitype:
            boxes = calibration_roi.extract_patch_coords(self.args, img_path)
        else:
            raise ValueError(f"Unknown ROI type: {roitype}")

        if 'rand' not in roitype:
            self._box_cache[key] = boxes
        return boxes

    def _image_to_patches(self, img, boxes):
        patches = []
        for (x1, y1, x2, y2) in boxes:
            patch = img[y1:y2, x1:x2, :]
            if self.transform:
                patch = self.transform(patch)
            patches.append(patch)

        return torch.stack(patches, dim=0)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = row['image_name']

        base_dir = os.path.join(self.root_dir, self.args.data)
            
        cover_path = os.path.join(base_dir, 'Cover', image_name)    
        stego_path = os.path.join(base_dir, 'Stego', self.args.stego_method, image_name)

        if not os.path.exists(cover_path):
            raise FileNotFoundError(f"Cover not found: {cover_path}")
        if not os.path.exists(stego_path):
            raise FileNotFoundError(f"Stego not found: {stego_path}")

        try:
            cover_img = np.array(Image.open(cover_path).convert("YCbCr"))
            cover_qt_raw = jpeglib.read_dct(cover_path).qt
            if len(cover_qt_raw) == 1:
                cover_qt = np.stack([cover_qt_raw[0], cover_qt_raw[0]])
            else:
                cover_qt = np.stack([cover_qt_raw[0], cover_qt_raw[1]])
            cover_qt = np.array(cover_qt, dtype=np.float32)
            
        except Exception as e:
            raise RuntimeError(f"Error reading cover image {cover_path}: {e}")

        try:
            stego_img = np.array(Image.open(stego_path).convert("YCbCr"))
            stego_qt_raw = jpeglib.read_dct(stego_path).qt
            if len(stego_qt_raw) == 1:
                stego_qt = np.stack([stego_qt_raw[0], stego_qt_raw[0]])
            else:
                stego_qt = np.stack([stego_qt_raw[0], stego_qt_raw[1]])
            stego_qt = np.array(stego_qt, dtype=np.float32)

        except Exception as e:
            raise RuntimeError(f"Error reading stego image {stego_path}: {e}")
        
        residual = np.abs(cover_img - stego_img)
        
        cover_boxes = self._extract_boxes(f"{image_name}|cover", cover_img, cover_path, 'random')
        cover_patches = self._image_to_patches(cover_img, cover_boxes)
        
        stego_boxes = self._extract_boxes(f"{image_name}|stego", stego_img, stego_path, residual)
        stego_patches = self._image_to_patches(stego_img, stego_boxes)

        # --- Metadata ---
        data = torch.stack([cover_patches, stego_patches], dim=0)
        qt = torch.from_numpy(np.stack([cover_qt, stego_qt])).float()
        label = torch.tensor([0, 1], dtype=torch.long)
        metadata = tuple(self._build_metadata(cover_path, row))

        return data, qt, label, metadata

class PairedDCTPatchDataset(Dataset):
    def __init__(self, df, root_dir='', transform=None, args=None):
        self.df        = df
        self.root_dir  = root_dir
        self.transform = transform
        self.args      = args

        self._box_cache  = {}

    def __len__(self):
        return len(self.df)

    def _build_metadata(self, img_path, row):
        lp = img_path.lower()
        if lp.endswith('.jpg') or lp.endswith('.jpeg'):
            metadata = [row['qf'], row['subsampling'], row['width'], row['height']]
        elif lp.endswith(('.png', '.bmp', '.tiff')):
            metadata = [row['width'], row['height']]
        return metadata
    
    # ----------------------------
    # Dataset Methods
    # ----------------------------
    def _extract_boxes(self, key, img, img_path, residual=None):
        if key in self._box_cache and 'npatch' not in self.args.roi_method:
            return self._box_cache[key]

        roitype = self.args.roi_method
        if 'nzac' in roitype:
            boxes = nzac_roi.extract_patch_coords(self.args, img, img_path)
        elif 'juniward' in roitype:
            boxes = juniward_roi.extract_patch_coords(self.args, Image.open(img_path))
        elif 'wavelet' in roitype:
            boxes = spatial_roi.extract_patch_coords(self.args, img)
        elif 'rand' in roitype:
            boxes = random_roi.extract_patch_coords(self.args, img, residual)
        elif 'calibration' in roitype:
            boxes = calibration_roi.extract_patch_coords(self.args, img_path)
        elif 'inconsist' in roitype:
            boxes = inconsist_roi.extract_patch_coords(self.args, img, img_path)
        else:
            raise ValueError(f"Unknown ROI type: {roitype}")

        if 'rand' not in roitype:
            self._box_cache[key] = boxes
        return boxes

    def _detect_orient(self, spatial, jpeg):
        H, W = spatial.shape[:2]
        Hb, Wb = jpeg.Y.shape[:2]
        H_vh, W_vh = Hb * 8, Wb * 8
        diff_vh = abs(H - H_vh) + abs(W - W_vh)
        H_hv, W_hv = Wb * 8, Hb * 8
        diff_hv = abs(H - H_hv) + abs(W - W_hv)
        return "vh" if diff_vh <= diff_hv else "hv"

    def _dequantize_qdct(self, qdct: torch.Tensor, qt: torch.Tensor) -> torch.Tensor:
        assert qdct.dim() == 4 and qdct.size(1) == 3
        assert qt.dim() == 4 and qt.size(1) == 3 and qt.size(2) == 8 and qt.size(3) == 8
        N, C, H, W = qdct.shape
        assert H % 8 == 0 and W % 8 == 0

        qdct_blocks = qdct.reshape(N, C, H//8, 8, W//8, 8)
        qt_per_ch = qt.index_select(dim=1, index=torch.tensor([0,1,2], device=qdct.device))
        qt_bc = qt_per_ch.unsqueeze(2).unsqueeze(4)
        dqdct_blocks = qdct_blocks.to(torch.float32) * qt_bc.to(torch.float32)
        return dqdct_blocks.reshape(N, C, H, W)

    def _transforms(self, patches):
        if self.transform is None:
            return patches

        transformed = []
        for patch in patches:
            patch = self.transform(patch)
            transformed.append(patch)

        return torch.stack(transformed, dim=0)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = row['image_name']
        base_dir = os.path.join(self.root_dir, self.args.data)
        cover_path = os.path.join(base_dir, 'Cover', image_name)
        stego_path = os.path.join(base_dir, 'Stego', self.args.stego_method, image_name)

        if not os.path.exists(cover_path): raise FileNotFoundError(f"Cover not found: {cover_path}")
        if not os.path.exists(stego_path): raise FileNotFoundError(f"Stego not found: {stego_path}")

        cover_img = cv2.cvtColor(cv2.imread(cover_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        cover_raw = jpeglib.read_dct(cover_path)
        stego_img = cv2.cvtColor(cv2.imread(stego_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        stego_raw = jpeglib.read_dct(stego_path)

        residual = np.abs(cover_img - stego_img)
        cover_boxes = self._extract_boxes(f"{image_name}|cover", cover_img, cover_path, 'random')
        orient = self._detect_orient(cover_img, cover_raw)
        cover_qdct, cover_qt = roi_qdct_mapping(cover_raw, cover_boxes, orient)
        stego_boxes = self._extract_boxes(f"{image_name}|stego", stego_img, stego_path, residual)
        orient = self._detect_orient(stego_img, stego_raw)
        stego_qdct, stego_qt = roi_qdct_mapping(stego_raw, stego_boxes, orient)

        cover_qdct = torch.stack([torch.from_numpy(c).permute(2,0,1) for c in cover_qdct], dim=0)
        cover_qt   = torch.stack([torch.from_numpy(c) for c in cover_qt], dim=0)
        stego_qdct = torch.stack([torch.from_numpy(s).permute(2,0,1) for s in stego_qdct], dim=0)
        stego_qt   = torch.stack([torch.from_numpy(s) for s in stego_qt], dim=0)

        cover_dqdct = self._dequantize_qdct(cover_qdct, cover_qt)
        stego_dqdct = self._dequantize_qdct(stego_qdct, stego_qt)

        cover_dqdct = self._transforms(cover_dqdct)
        stego_dqdct = self._transforms(stego_dqdct)

        data = torch.stack([cover_dqdct, stego_dqdct], dim=0)
        label = torch.tensor([0, 1], dtype=torch.long)
        metadata = tuple(self._build_metadata(cover_path, row))
        return data, label, metadata

class TwoStreamDataset(Dataset):
    def __init__(self, df, root_dir='', transform=None, args=None):
        self.df        = df
        self.root_dir  = root_dir

        if isinstance(transform, dict):
            self.spatial_transform = transform.get('spatial', None)
            self.dct_transform = transform.get('dct', None)
        else:
            self.spatial_transform = None
            self.dct_transform = None

        self.args      = args
        self._box_cache  = {}

    def __len__(self):
        return len(self.df)

    def _build_metadata(self, img_path, row):
        lp = img_path.lower()
        if lp.endswith('.jpg') or lp.endswith('.jpeg'):
            metadata = [row['qf'], row['subsampling'], row['width'], row['height']]
        elif lp.endswith(('.png', '.bmp', '.tiff')):
            metadata = [row['width'], row['height']]
        return metadata

    def _spatial_crop(self, img, boxes):
        H, W = img.shape[:2]

        patches = []

        for (x1, y1, x2, y2) in boxes:
            patch = img[y1:y2, x1:x2, :]
            patches.append(patch)

        return patches

    # ----------------------------
    # Dataset Methods
    # ----------------------------
    def _extract_boxes(self, key, img, img_path, residual=None):
        if key in self._box_cache and 'npatch' not in self.args.roi_method:
            return self._box_cache[key]

        roitype = self.args.roi_method
        if 'nzac' in roitype:
            boxes = nzac_roi.extract_patch_coords(self.args, img, img_path)
        elif 'juniward' in roitype:
            boxes = juniward_roi.extract_patch_coords(self.args, Image.open(img_path))
        elif 'wavelet' in roitype:
            boxes = spatial_roi.extract_patch_coords(self.args, img)
        elif 'rand' in roitype:
            boxes = random_roi.extract_patch_coords(self.args, img, residual)
        elif 'calibration' in roitype:
            boxes = calibration_roi.extract_patch_coords(self.args, img_path)
        else:
            raise ValueError(f"Unknown ROI type: {roitype}")

        if 'rand' not in roitype:
            self._box_cache[key] = boxes
        return boxes

    def _detect_orient(self, spatial, jpeg):
        H, W = spatial.shape[:2]
        Hb, Wb = jpeg.Y.shape[:2]
        H_vh, W_vh = Hb * 8, Wb * 8
        diff_vh = abs(H - H_vh) + abs(W - W_vh)
        H_hv, W_hv = Wb * 8, Hb * 8
        diff_hv = abs(H - H_hv) + abs(W - W_hv)
        return "vh" if diff_vh <= diff_hv else "hv"

    def _dequantize_qdct(self, qdct: torch.Tensor, qt: torch.Tensor) -> torch.Tensor:
        assert qdct.dim() == 4 and qdct.size(1) == 3
        assert qt.dim() == 4 and qt.size(1) == 3 and qt.size(2) == 8 and qt.size(3) == 8
        N, C, H, W = qdct.shape
        assert H % 8 == 0 and W % 8 == 0

        qdct_blocks = qdct.reshape(N, C, H//8, 8, W//8, 8)
        qt_per_ch = qt.index_select(dim=1, index=torch.tensor([0,1,2], device=qdct.device))
        qt_bc = qt_per_ch.unsqueeze(2).unsqueeze(4)
        dqdct_blocks = qdct_blocks.to(torch.float32) * qt_bc.to(torch.float32)
        return dqdct_blocks.reshape(N, C, H, W)

    def _spatial_transforms(self, patches):
        if self.spatial_transform is None:
            return patches

        transformed = []
        for patch in patches:
            patch = self.spatial_transform(patch)
            transformed.append(patch)

        return torch.stack(transformed, dim=0)

    def _dct_transforms(self, patches):
        if self.dct_transform is None:
            return patches

        transformed = []
        for patch in patches:
            patch = self.dct_transform(patch)
            transformed.append(patch)

        return torch.stack(transformed, dim=0)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = row['image_name']
        base_dir = os.path.join(self.root_dir, self.args.data)
        cover_path = os.path.join(base_dir, 'Cover', image_name)
        stego_path = os.path.join(base_dir, 'Stego', self.args.stego_method, image_name)

        if not os.path.exists(cover_path): raise FileNotFoundError(f"Cover not found: {cover_path}")
        if not os.path.exists(stego_path): raise FileNotFoundError(f"Stego not found: {stego_path}")

        cover_img = cv2.cvtColor(cv2.imread(cover_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        cover_raw = jpeglib.read_dct(cover_path)
        stego_img = cv2.cvtColor(cv2.imread(stego_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        stego_raw = jpeglib.read_dct(stego_path)

        residual = np.abs(cover_img - stego_img)
        cover_boxes = self._extract_boxes(f"{image_name}|cover", cover_img, cover_path, 'random')
        orient = self._detect_orient(cover_img, cover_raw)
        cover_img = self._spatial_crop(cover_img, cover_boxes)
        cover_qdct, cover_qt = roi_qdct_mapping(cover_raw, cover_boxes, orient)
        stego_boxes = self._extract_boxes(f"{image_name}|stego", stego_img, stego_path, residual)
        orient = self._detect_orient(stego_img, stego_raw)
        stego_img = self._spatial_crop(stego_img, stego_boxes)
        stego_qdct, stego_qt = roi_qdct_mapping(stego_raw, stego_boxes, orient)

        cover_qdct = torch.stack([torch.from_numpy(c).permute(2,0,1) for c in cover_qdct], dim=0)
        cover_qt   = torch.stack([torch.from_numpy(c) for c in cover_qt], dim=0)
        stego_qdct = torch.stack([torch.from_numpy(s).permute(2,0,1) for s in stego_qdct], dim=0)
        stego_qt   = torch.stack([torch.from_numpy(s) for s in stego_qt], dim=0)

        cover_img = self._spatial_transforms(cover_img)
        stego_img = self._spatial_transforms(stego_img)

        cover_dqdct = self._dequantize_qdct(cover_qdct, cover_qt)
        stego_dqdct = self._dequantize_qdct(stego_qdct, stego_qt)
        
        cover_dqdct = self._dct_transforms(cover_dqdct)
        stego_dqdct = self._dct_transforms(stego_dqdct)

        data = (torch.stack([cover_img, stego_img], dim=0), torch.stack([cover_dqdct, stego_dqdct], dim=0))
        label = torch.tensor([0, 1], dtype=torch.long)
        metadata = tuple(self._build_metadata(cover_path, row))

        return data, label, metadata

class SF3NetDataset(Dataset):
    def __init__(self, df, root_dir='', transform=None, args=None):
        self.df        = df
        self.root_dir  = root_dir
        self.transform = transform
        self.args      = args
        self._box_cache  = {}

    def __len__(self):
        return len(self.df)

    def _build_metadata(self, img_path, row):
        lp = img_path.lower()
        if lp.endswith('.jpg') or lp.endswith('.jpeg'):
            metadata = [row['qf'], row['subsampling'], row['width'], row['height']]
        elif lp.endswith(('.png', '.bmp', '.tiff')):
            metadata = [row['width'], row['height']]
        return metadata

    # ---------- helpers ----------
    @staticmethod
    def _rgb_to_y(rgb_uint8: np.ndarray) -> np.ndarray:
        y = np.array(Image.fromarray(rgb_uint8, mode="RGB").convert("YCbCr"))[..., 0]
        return y.astype(np.float32) / 255.0

    @staticmethod
    def _qdct_to_gray_tensor(qdct: torch.Tensor) -> torch.Tensor:
        if isinstance(qdct, np.ndarray):
            qdct = torch.from_numpy(qdct)

        if qdct.dim() == 4 and qdct.size(1) == 3:
            # [R,3,H,W]
            return qdct.float().mean(dim=1, keepdim=True)
        if qdct.dim() == 4 and qdct.size(-1) == 3:
            # [R,H,W,3]
            return qdct.permute(0, 3, 1, 2).float().mean(dim=1, keepdim=True)
        raise ValueError(f"Unexpected qdct shape: {tuple(qdct.shape)}")

    # ----------------------------
    # Crops + transforms
    # ----------------------------
    def _spatial_crop(self, img_rgb_uint8, boxes):
        patches = []
        for (x1, y1, x2, y2) in boxes:
            patch_rgb = img_rgb_uint8[y1:y2, x1:x2, :]
            patch_y = self._rgb_to_y(patch_rgb)

            if self.transform is not None:
                pil = Image.fromarray((patch_y * 255.0).round().astype(np.uint8), mode="L")
                t = self.transform(pil)
                if t.dim() == 3 and t.size(0) != 1:
                    t = t[:1]
                patches.append(t)
            else:
                patches.append(torch.from_numpy(patch_y).unsqueeze(0))

        return torch.stack(patches, dim=0)

    def _transforms_qdct(self, qdct_list):

        patches = []
        for p in qdct_list:
            p = np.asarray(p)

            if p.ndim == 3:
                y = p[..., 0]
            elif p.ndim == 2:
                y = p
            else:
                raise ValueError(f"Unexpected qdct patch shape: {p.shape}")

            t = torch.from_numpy(y).unsqueeze(0).float()
            patches.append(t)

        return torch.stack(patches, dim=0)

    # ----------------------------
    # ROI boxes
    # ----------------------------
    def _extract_boxes(self, key, img, img_path, residual=None):
        if key in self._box_cache and 'npatch' not in self.args.roi_method:
            return self._box_cache[key]

        roitype = self.args.roi_method
        if 'nzac' in roitype:
            boxes = nzac_roi.extract_patch_coords(self.args, img, img_path)
        elif 'juniward' in roitype:
            boxes = juniward_roi.extract_patch_coords(self.args, Image.open(img_path))
        elif 'wavelet' in roitype:
            boxes = spatial_roi.extract_patch_coords(self.args, img)
        elif 'rand' in roitype:
            boxes = random_roi.extract_patch_coords(self.args, img, residual)
        elif 'calibration' in roitype:
            boxes = calibration_roi.extract_patch_coords(self.args, img_path)
        else:
            raise ValueError(f"Unknown ROI type: {roitype}")

        if 'rand' not in roitype:
            self._box_cache[key] = boxes
        return boxes

    def _detect_orient(self, spatial, jpeg):
        H, W = spatial.shape[:2]
        Hb, Wb = jpeg.Y.shape[:2]
        H_vh, W_vh = Hb * 8, Wb * 8
        diff_vh = abs(H - H_vh) + abs(W - W_vh)
        H_hv, W_hv = Wb * 8, Hb * 8
        diff_hv = abs(H - H_hv) + abs(W - W_hv)
        return "vh" if diff_vh <= diff_hv else "hv"

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = row['image_name']
        base_dir = os.path.join(self.root_dir, self.args.data)
        cover_path = os.path.join(base_dir, 'Cover', image_name)
        stego_path = os.path.join(base_dir, 'Stego', self.args.stego_method, image_name)

        if not os.path.exists(cover_path): raise FileNotFoundError(f"Cover not found: {cover_path}")
        if not os.path.exists(stego_path): raise FileNotFoundError(f"Stego not found: {stego_path}")

        cover_img = cv2.cvtColor(cv2.imread(cover_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        cover_raw = jpeglib.read_dct(cover_path)
        stego_img = cv2.cvtColor(cv2.imread(stego_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        stego_raw = jpeglib.read_dct(stego_path)

        residual = np.abs(cover_img.astype(np.int16) - stego_img.astype(np.int16)).astype(np.uint8)

        # ---- cover ----
        cover_boxes = self._extract_boxes(f"{image_name}|cover", cover_img, cover_path, 'random')
        orient = self._detect_orient(cover_img, cover_raw)

        cover_spatial = self._spatial_crop(cover_img, cover_boxes)  # [R,1,H,W]
        cover_qdct, cover_qt = roi_qdct_mapping(cover_raw, cover_boxes, orient)
        cover_qdct = self._transforms_qdct(cover_qdct)                  # [R,1,H,W]

        qt_list = []
        for qt in cover_qt:
            qt = np.array(qt)
            if qt.ndim == 3 and qt.shape[0] >= 1:
                qt_y = qt[0]  # [8,8]
            elif qt.ndim == 2:
                qt_y = qt
            else:
                raise ValueError(f"Unexpected qt shape: {qt.shape}")
            qt_list.append(torch.from_numpy(qt_y).unsqueeze(0))  # [1,8,8]
        cover_qt = torch.stack(qt_list, dim=0).float()  # [R,1,8,8]

        # ---- stego ----
        stego_boxes = self._extract_boxes(f"{image_name}|stego", stego_img, stego_path, residual)
        orient = self._detect_orient(stego_img, stego_raw)

        stego_spatial = self._spatial_crop(stego_img, stego_boxes)  # [R,1,H,W]
        stego_qdct, stego_qt = roi_qdct_mapping(stego_raw, stego_boxes, orient)
        stego_qdct = self._transforms_qdct(stego_qdct)                  # [R,1,H,W]

        qt_list = []
        for qt in stego_qt:
            qt = np.array(qt)
            if qt.ndim == 3 and qt.shape[0] >= 1:
                qt_y = qt[0]
            elif qt.ndim == 2:
                qt_y = qt
            else:
                raise ValueError(f"Unexpected qt shape: {qt.shape}")
            qt_list.append(torch.from_numpy(qt_y).unsqueeze(0))  # [1,8,8]
        stego_qt = torch.stack(qt_list, dim=0).float()  # [R,1,8,8]

        # ---- pack ----
        data = (
            torch.stack([cover_spatial, stego_spatial], dim=0),  # [2,R,1,H,W]
            torch.stack([cover_qdct,   stego_qdct],   dim=0),     # [2,R,1,H,W]
            torch.stack([cover_qt,     stego_qt],     dim=0),     # [2,R,1,8,8]
        )

        label = torch.tensor([0, 1], dtype=torch.long)
        metadata = tuple(self._build_metadata(cover_path, row))
        return data, label, metadata


# class SF3NetDataset(Dataset):
#     def __init__(self, df, root_dir='', transform=None, args=None):
#         self.df        = df
#         self.root_dir  = root_dir
#         self.transform = transform
#         self.args      = args

#         self._box_cache  = {}

#     def __len__(self):
#         return len(self.df)

#     def _build_metadata(self, img_path, row):
#         lp = img_path.lower()
#         if lp.endswith('.jpg') or lp.endswith('.jpeg'):
#             metadata = [row['qf'], row['subsampling'], row['width'], row['height']]
#         elif lp.endswith(('.png', '.bmp', '.tiff')):
#             metadata = [row['width'], row['height']]
#         return metadata

#     def _spatial_crop(self, img, boxes):
#         H, W = img.shape[:2]

#         patches = []

#         for (x1, y1, x2, y2) in boxes:
#             patch = img[y1:y2, x1:x2, :]
#             patches.append(patch)

#         transformed = []
#         for patch in patches:
#             if self.transform is None:
#                 transformed.append(patch)

#             patch = self.transform(Image.fromarray(patch))
#             transformed.append(patch)

#         return torch.stack(transformed, dim=0)

#     # ----------------------------
#     # Dataset Methods
#     # ----------------------------
#     def _extract_boxes(self, key, img, img_path, residual=None):
#         if key in self._box_cache and 'npatch' not in self.args.roi_method:
#             return self._box_cache[key]

#         roitype = self.args.roi_method
#         if 'nzac' in roitype:
#             boxes = nzac_roi.extract_patch_coords(self.args, img, img_path)
#         elif 'juniward' in roitype:
#             boxes = juniward_roi.extract_patch_coords(self.args, Image.open(img_path))
#         elif 'wavelet' in roitype:
#             boxes = spatial_roi.extract_patch_coords(self.args, img)
#         elif 'rand' in roitype:
#             boxes = random_roi.extract_patch_coords(self.args, img, residual)
#         elif 'calibration' in roitype:
#             boxes = calibration_roi.extract_patch_coords(self.args, img_path)
#         else:
#             raise ValueError(f"Unknown ROI type: {roitype}")

#         if 'rand' not in roitype:
#             self._box_cache[key] = boxes
#         return boxes

#     def _detect_orient(self, spatial, jpeg):
#         H, W = spatial.shape[:2]
#         Hb, Wb = jpeg.Y.shape[:2]
#         H_vh, W_vh = Hb * 8, Wb * 8
#         diff_vh = abs(H - H_vh) + abs(W - W_vh)
#         H_hv, W_hv = Wb * 8, Hb * 8
#         diff_hv = abs(H - H_hv) + abs(W - W_hv)
#         return "vh" if diff_vh <= diff_hv else "hv"

#     def _transforms(self, patches):
#         if self.transform is None:
#             return patches

#         transformed = []
#         for patch in patches:
#             patch = self.transform(patch)
#             transformed.append(patch)

#         return torch.stack(transformed, dim=0)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         image_name = row['image_name']
#         base_dir = os.path.join(self.root_dir, self.args.data)
#         cover_path = os.path.join(base_dir, 'Cover', image_name)
#         stego_path = os.path.join(base_dir, 'Stego', self.args.stego_method, image_name)

#         if not os.path.exists(cover_path): raise FileNotFoundError(f"Cover not found: {cover_path}")
#         if not os.path.exists(stego_path): raise FileNotFoundError(f"Stego not found: {stego_path}")

#         cover_img = cv2.cvtColor(cv2.imread(cover_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
#         cover_raw = jpeglib.read_dct(cover_path)
#         stego_img = cv2.cvtColor(cv2.imread(stego_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
#         stego_raw = jpeglib.read_dct(stego_path)

#         residual = np.abs(cover_img - stego_img)
#         cover_boxes = self._extract_boxes(f"{image_name}|cover", cover_img, cover_path, 'random')
#         orient = self._detect_orient(cover_img, cover_raw)
#         cover_img = self._spatial_crop(cover_img, cover_boxes)
#         cover_qdct, cover_qt = roi_qdct_mapping(cover_raw, cover_boxes, orient)
#         stego_boxes = self._extract_boxes(f"{image_name}|stego", stego_img, stego_path, residual)
#         orient = self._detect_orient(stego_img, stego_raw)
#         stego_img = self._spatial_crop(stego_img, stego_boxes)
#         stego_qdct, stego_qt = roi_qdct_mapping(stego_raw, stego_boxes, orient)

#         # cover_qdct = torch.stack([torch.from_numpy(c).permute(2,0,1) for c in cover_qdct], dim=0)
#         cover_qt   = torch.stack([torch.from_numpy(c) for c in cover_qt], dim=0)
#         # stego_qdct = torch.stack([torch.from_numpy(s).permute(2,0,1) for s in stego_qdct], dim=0)
#         stego_qt   = torch.stack([torch.from_numpy(s) for s in stego_qt], dim=0)

#         cover_qdct = self._transforms(cover_qdct)
#         stego_qdct = self._transforms(stego_qdct)

#         data = (torch.stack([cover_img, stego_img], dim=0), torch.stack([cover_qdct, stego_qdct], dim=0), torch.stack([cover_qt, stego_qt], dim=0))
#         label = torch.tensor([0, 1], dtype=torch.long)
#         metadata = tuple(self._build_metadata(cover_path, row))

#         return data, label, metadata


class VisualizeDCTDataset(Dataset):
    def __init__(self, df, root_dir='', transform=None, args=None):
        self.df        = df
        self.root_dir  = root_dir
        if isinstance(transform, dict):
            self.spatial_transform = transform.get('spatial', None)
            self.dct_transform = transform.get('dct', None)
        else:
            self.spatial_transform = None
            self.dct_transform = None

        self.args      = args

        self._box_cache  = {}

    def __len__(self):
        return len(self.df)

    def _build_metadata(self, img_path, row):
        lp = img_path.lower()
        if lp.endswith('.jpg') or lp.endswith('.jpeg'):
            metadata = [row['qf'], row['subsampling'], row['width'], row['height']]
        elif lp.endswith(('.png', '.bmp', '.tiff')):
            metadata = [row['width'], row['height']]
        return metadata

    def _spatial_crop(self, img, boxes):
        H, W = img.shape[:2]

        patches = []

        for (x1, y1, x2, y2) in boxes:
            patch = img[y1:y2, x1:x2, :]
            patches.append(patch)

        return patches

    # ----------------------------
    # Dataset Methods
    # ----------------------------
    def _extract_boxes(self, key, img, img_path, residual=None):
        if key in self._box_cache and 'npatch' not in self.args.roi_method:
            return self._box_cache[key]

        roitype = self.args.roi_method
        if 'nzac' in roitype:
            boxes = nzac_roi.extract_patch_coords(self.args, img, img_path)
        elif 'juniward' in roitype:
            boxes = juniward_roi.extract_patch_coords(self.args, Image.open(img_path))
        elif 'wavelet' in roitype:
            boxes = spatial_roi.extract_patch_coords(self.args, img)
        elif 'rand' in roitype:
            boxes = random_roi.extract_patch_coords(self.args, img, residual)
        elif 'calibration' in roitype:
            boxes = calibration_roi.extract_patch_coords(self.args, img_path)
        elif 'inconsist' in roitype:
            boxes = inconsist_roi.extract_patch_coords(self.args, img, img_path)
        else:
            raise ValueError(f"Unknown ROI type: {roitype}")

        if 'rand' not in roitype:
            self._box_cache[key] = boxes
        return boxes


    def _detect_orient(self, spatial, jpeg):
        H, W = spatial.shape[:2]
        Hb, Wb = jpeg.Y.shape[:2]
        H_vh, W_vh = Hb * 8, Wb * 8
        diff_vh = abs(H - H_vh) + abs(W - W_vh)
        H_hv, W_hv = Wb * 8, Hb * 8
        diff_hv = abs(H - H_hv) + abs(W - W_hv)
        return "vh" if diff_vh <= diff_hv else "hv"

    def _dequantize_qdct(self, qdct: torch.Tensor, qt: torch.Tensor) -> torch.Tensor:
        assert qdct.dim() == 4 and qdct.size(1) == 3
        assert qt.dim() == 4 and qt.size(1) == 3 and qt.size(2) == 8 and qt.size(3) == 8
        N, C, H, W = qdct.shape
        assert H % 8 == 0 and W % 8 == 0

        qdct_blocks = qdct.reshape(N, C, H//8, 8, W//8, 8)
        qt_per_ch = qt.index_select(dim=1, index=torch.tensor([0,1,2], device=qdct.device))
        qt_bc = qt_per_ch.unsqueeze(2).unsqueeze(4)
        dqdct_blocks = qdct_blocks.to(torch.float32) * qt_bc.to(torch.float32)
        return dqdct_blocks.reshape(N, C, H, W)
    def _spatial_transforms(self, patches):
        if self.spatial_transform is None:
            return patches

        transformed = []
        for patch in patches:
            patch = self.spatial_transform(patch)
            transformed.append(patch)

        return torch.stack(transformed, dim=0)

    def _dct_transforms(self, patches):
        if self.dct_transform is None:
            return patches

        transformed = []
        for patch in patches:
            patch = self.dct_transform(patch)
            transformed.append(patch)

        return torch.stack(transformed, dim=0)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = row['image_name']
        base_dir = os.path.join(self.root_dir, self.args.data)
        cover_path = os.path.join(base_dir, 'Cover', image_name)
        stego_path = os.path.join(base_dir, 'Stego', self.args.stego_method, image_name)

        if not os.path.exists(cover_path): raise FileNotFoundError(f"Cover not found: {cover_path}")
        if not os.path.exists(stego_path): raise FileNotFoundError(f"Stego not found: {stego_path}")

        cover_img = cv2.cvtColor(cv2.imread(cover_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        cover_raw = jpeglib.read_dct(cover_path)
        stego_img = cv2.cvtColor(cv2.imread(stego_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        stego_raw = jpeglib.read_dct(stego_path)

        residual = np.abs(cover_img - stego_img)
        cover_boxes = self._extract_boxes(f"{image_name}|cover", cover_img, cover_path, 'random')
        orient = self._detect_orient(cover_img, cover_raw)
        cover_img = self._spatial_crop(cover_img, cover_boxes)
        cover_qdct, cover_qt = roi_qdct_mapping(cover_raw, cover_boxes, orient)
        stego_boxes = self._extract_boxes(f"{image_name}|stego", stego_img, stego_path, residual)
        orient = self._detect_orient(stego_img, stego_raw)
        stego_img = self._spatial_crop(stego_img, stego_boxes)
        stego_qdct, stego_qt = roi_qdct_mapping(stego_raw, stego_boxes, orient)
        
        cover_qdct = torch.stack([torch.from_numpy(c).permute(2,0,1) for c in cover_qdct], dim=0)
        cover_qt   = torch.stack([torch.from_numpy(c) for c in cover_qt], dim=0)
        stego_qdct = torch.stack([torch.from_numpy(s).permute(2,0,1) for s in stego_qdct], dim=0)
        stego_qt   = torch.stack([torch.from_numpy(s) for s in stego_qt], dim=0)

        cover_dqdct = self._dequantize_qdct(cover_qdct, cover_qt)
        stego_dqdct = self._dequantize_qdct(stego_qdct, stego_qt)

        cover_img = self._spatial_transforms(cover_img)
        stego_img = self._spatial_transforms(stego_img)
        cover_dqdct = self._dct_transforms(cover_dqdct)
        stego_dqdct = self._dct_transforms(stego_dqdct)

        data = (torch.stack([cover_img, stego_img], dim=0), torch.stack([cover_dqdct, stego_dqdct], dim=0), torch.stack([cover_qdct, stego_qdct], dim=0), torch.stack([cover_qt, stego_qt], dim=0))
        paths = [cover_path, stego_path]

        return data, paths

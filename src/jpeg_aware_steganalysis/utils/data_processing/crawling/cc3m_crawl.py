from jpeg_aware_steganalysis._paths import data_path as _checkout_data_path
import os
import io
import requests
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def _is_valid_image_pillow(content: bytes, min_size: int = 256) -> bool:
    """PIL로 디코딩 후 유효성/해상도/컬러 체크"""
    try:
        bio = io.BytesIO(content)
        img = Image.open(bio)
        img.verify()

        img = Image.open(io.BytesIO(content))
        img.load()
    except Exception:
        return False

    arr = np.array(img)
    h, w = arr.shape[:2]
    if h < min_size or w < min_size:
        return False

    # 흑백 이미지 제외
    if img.mode in ("L", "LA"):
        return False
    if img.mode == "RGB":
        if np.all(arr[:, :, 0] == arr[:, :, 1]) and np.all(arr[:, :, 1] == arr[:, :, 2]):
            return False

    return True


def _is_valid_image_opencv(content: bytes, min_size: int = 256) -> bool:
    """OpenCV로 디코딩 후 유효성/해상도/컬러 체크"""
    try:
        buf = np.frombuffer(content, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if img is None:
            return False

        h, w = img.shape[:2]
        if h < min_size or w < min_size:
            return False

        # 단일 채널 제외
        if img.ndim == 2:
            return False

        if img.ndim == 3:
            c = img.shape[2]
            if c == 2:
                return False
            if c >= 3:
                b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
                if np.array_equal(b, g) and np.array_equal(g, r):
                    return False

        return True
    except Exception:
        return False


def try_download(url, mode="jpeg"):
    """
    이미지 다운로드 시도 → (ext, content) 반환, 실패 시 None
    JPEG/Spatial 상관없이 Pillow + OpenCV 모두 통과해야 저장
    """
    ext = os.path.splitext(url.split("?")[0])[1].lower()

    if mode == "jpeg":
        valid_exts = [".jpg", ".jpeg"]
    elif mode == "spatial":
        valid_exts = [".png", ".tif", ".tiff"]
    else:
        raise ValueError("mode은 'jpeg' 또는 'spatial' 중 하나여야 합니다.")

    if ext not in valid_exts:
        return None

    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None

        content = r.content

        # 유효성 검사는 모든 포맷에서 적용
        if not (_is_valid_image_pillow(content) and _is_valid_image_opencv(content)):
            return None

        return (ext, content)
    except Exception:
        return None


def download_main(tsv_path, save_dir, max_images=None, num_workers=8, mode="jpeg"):
    urls = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if not parts:
                continue
            urls.append(parts[-1])

    saved_count = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(try_download, url, mode): url for url in urls}

        for future in tqdm(as_completed(futures), total=len(urls), desc="다운로드 중"):
            result = future.result()
            if result:
                ext, content = result
                fname = os.path.join(save_dir, f"image_{saved_count:06d}{ext}")
                with open(fname, "wb") as f:
                    f.write(content)
                saved_count += 1

                if max_images and saved_count >= max_images:
                    break

    print(f"\n총 {saved_count}개 이미지 다운로드 완료.")


if __name__ == "__main__":
    save_dir = str(_checkout_data_path('CC3M/PNG_10000/Cover/'))
    os.makedirs(save_dir, exist_ok=True)
    download_main("./data/CC3M/Train_GCC-training.tsv", save_dir, max_images=10000, num_workers=32, mode="spatial")

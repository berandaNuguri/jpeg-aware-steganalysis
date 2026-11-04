from jpeg_aware_steganalysis._paths import data_path as _checkout_data_path
import os
import io
import sys
import requests
import numpy as np
import random
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess

def _check_jpeg(content: bytes) -> bool:
    return content.startswith(b"\xFF\xD8") and content.endswith(b"\xFF\xD9")

def _check_rgb(content: bytes, min_size: int = 256) -> bool:
    try:
        img = Image.open(io.BytesIO(content))
        img.load()
        img.verify()
    except Exception:
        return False

    if img.width < min_size or img.height < min_size:
        return False

    if img.mode != "RGB":
        return False

    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return False

    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    if np.array_equal(r, g) and np.array_equal(g, b):
        return False

    return True


def _check_jpeglib(path: str) -> bool:
    """jpeglib.read_dct()를 별도 프로세스에서 실행해 안전하게 검사"""
    code = f"""
            import jpeglib
            try:
                jpeg = jpeglib.read_dct(r'''{path}''')
                _ = jpeg.qt[0]
                exit(0)
            except Exception:
                exit(1)
            """
    result = subprocess.run([sys.executable, "-c", code], capture_output=True)
    return result.returncode == 0


def try_download(url, mode="jpeg", tmp_dir="./tmp_check"):
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

        if mode == "jpeg" and not _check_jpeg(content):
            return None

        if not _check_rgb(content):
            return None

        if mode == "jpeg":
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_path = os.path.join(tmp_dir, "tmp_check.jpg")
            with open(tmp_path, "wb") as f:
                f.write(content)
            if not _check_jpeglib(tmp_path):
                return None

        return (ext, content)
    except Exception:
        return None


def download_main(tsv_path, save_dir, max_images=None, num_workers=8, mode="jpeg", seed=42):
    urls = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts:
                urls.append(parts[0])

    random.seed(seed)
    random.shuffle(urls)

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
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

    print(f"\n총 {saved_count}개 이미지 다운로드 완료.")

if __name__ == "__main__":
    save_dir = str(_checkout_data_path('CC12M/PNG_15000/Cover/'))
    os.makedirs(save_dir, exist_ok=True)
    download_main(
        "./data/CC12M/cc12m.tsv",
        save_dir,
        max_images=15000,
        num_workers=32,
        mode="spatial",
        seed=2025
    )
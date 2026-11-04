from jpeg_aware_steganalysis._paths import data_path as _checkout_data_path
import sys
import os
import io
import requests
import numpy as np
import warnings
import subprocess
import tempfile
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from PIL import Image, ImageFile
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.TiffImagePlugin")
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.JpegImagePlugin")

def estimate_qf(quant_table):
    std_table = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68,109,103, 77], [24, 35, 55, 64, 81,104,113, 92],
        [49, 64, 78, 87,103,121,120,101], [72, 92, 95, 98,112,100,103, 99]
    ], dtype=np.float32)
    best_quality, best_error = None, np.inf
    for quality in range(1, 101):
        scale = 5000 / quality if quality < 50 else 200 - 2 * quality
        comp = np.floor((std_table * scale + 50) / 100)
        comp = np.clip(comp, 1, 255)
        err = np.mean(np.abs(comp - quant_table))
        if err < best_error:
            best_error, best_quality = err, quality
    return best_quality

def download_image_bytes(url: str, timeout: int = 30):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0'
        }
        resp = requests.get(url, stream=True, timeout=timeout, headers=headers)
        resp.raise_for_status()
        ct = resp.headers.get('Content-Type', '').lower()
        if 'jpeg' not in ct:
            return None
        data = resp.content
        if not data.startswith(b'\xff\xd8'):
            return None
        return data
    except Exception:
        return None

def passes_filters(example: dict) -> bool:
    try:
        return (
            example["width"] >= 256 and example["height"] >= 256 and
            (example["url"].endswith('.jpg') or example["url"].endswith('.jpeg')) and
            example["nsfw_score_opennsfw2"] < 0.1 and example["nsfw_score_gantman"] < 0.1
        )
    except (KeyError, TypeError):
        return False

def is_jpeglib_safe(content: bytes) -> bool:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp.write(content)
            tmp.flush()
            tmp_path = tmp.name
        cmd = [
            sys.executable, "-c",
            ("import jpeglib; "
             f"p=r'{tmp_path}'; "
             "jpeglib.read_spatial(p); "
             "jpeglib.read_dct(p)")
        ]
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return proc.returncode == 0
    except Exception:
        return False
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

def download_and_process_image(url: str, qf_threshold: list):
    content = download_image_bytes(url)
    if not content:
        return None

    try:
        with Image.open(io.BytesIO(content)) as img:
            img.load()

            # ---- 최소 사이즈 필터 ----
            w, h = img.size
            if w < 256 or h < 256:
                return None
            # ------------------------

            if img.mode in ('L', 'LA'):
                return None

            if img.mode == 'RGB':
                arr = np.array(img)
                if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1):
                    return None

            qtables = getattr(img, "quantization", None)
            if not qtables or 0 not in qtables:
                return None

            qt0 = np.array(qtables[0])
            if qt0.size != 64:
                return None
            quant_table = qt0.reshape(8, 8)

            qf = estimate_qf(quant_table)
            if qf is not None and qf_threshold[1] >= qf >= qf_threshold[0]:
                if is_jpeglib_safe(content):
                    return content

    except Exception:
        return None
    return None

def main(
    output_dir: str = "Cover",
    max_images: int = 50,
    max_workers: int = 16,
    qf_threshold: list = [75, 100]
):
    os.makedirs(output_dir, exist_ok=True)
    ds = load_dataset("kakaobrain/coyo-700m", split="train", streaming=True)
    it = iter(ds)
    futures = {}
    pbar = tqdm(total=max_images, desc="이미지 수집 중")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while pbar.n < max_images:
            while len(futures) < max_workers:
                try:
                    ex = next(it)
                    if passes_filters(ex):
                        fut = executor.submit(download_and_process_image, ex["url"], qf_threshold)
                        futures[fut] = ex["url"]
                except StopIteration:
                    break
            if not futures:
                break
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for fut in done:
                url = futures.pop(fut)
                content = fut.result()
                if content:
                    try:
                        ext = os.path.splitext(url)[1].lower()
                        if ext not in ('.jpg', '.jpeg'):
                            ext = '.jpg'
                        final_fn = f"image_{pbar.n:06d}{ext}"
                        final_path = os.path.join(output_dir, final_fn)
                        with open(final_path, 'wb') as f:
                            f.write(content)
                        pbar.update(1)
                    except Exception as e:
                        print(f"[오류] 파일 저장 중 예외 발생 (URL: {url}): {e}")

    for fut in futures:
        fut.cancel()
    pbar.close()
    print(f"\n✅ 완료: 저장된 이미지 수 = {pbar.n}")

if __name__ == "__main__":
    main(
        output_dir=str(_checkout_data_path('COYO700M/JPG_30000_QF75_100/Cover/')),
        max_images=30000,
        max_workers=48,
        qf_threshold=[75, 100]
    )

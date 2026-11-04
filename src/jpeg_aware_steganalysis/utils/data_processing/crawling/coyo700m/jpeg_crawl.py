from jpeg_aware_steganalysis._paths import data_path as _checkout_data_path
import sys
import os
import io
import subprocess
import tempfile
import requests
import numpy as np
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from PIL import Image

def download_image_bytes(url: str, timeout: int = 10):
    """
    스트리밍으로 이미지 바이트를 가져오고, JPEG 포맷만 반환.
    """
    try:
        resp = requests.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()
        ct = resp.headers.get('Content-Type', '').lower()
        if 'jpeg' not in ct:
            return None
        ext = os.path.splitext(url)[1].lower()
        if ext not in ('.jpg', '.jpeg'):
            return None

        data = resp.content
        if not data.startswith(b'\xff\xd8'):
            return None

        return data, ext
    except Exception:
        return None

def is_image_valid(content: bytes) -> bool:
    if not content.startswith(b'\xff\xd8'):
        return False

    try:
        bio = io.BytesIO(content)
        img = Image.open(bio)
        img.verify()
        img = Image.open(io.BytesIO(content))
        img.load()
    except Exception:
        return False

    if img.mode in ('L', 'LA'):
        return False

    if img.mode == 'RGB':
        arr = np.array(img)
        if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1):
                    return False
        # if arr.ndim == 3 and arr.shape[2] == 3:
        #     if np.all(arr[:,:,0] == arr[:,:,1]) and np.all(arr[:,:,1] == arr[:,:,2]):
        #         return False
        
    return True

def is_jpeglib_safe(content: bytes) -> bool:
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        tmp_path = tmp.name

    cmd = [
        sys.executable, "-c",
        ("import jpeglib; "
         f"jpeglib.read_spatial(r'{tmp_path}'); "
         f"jpeglib.read_dct(r'{tmp_path}')")
    ]
    proc = subprocess.run(cmd,
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL)
    os.remove(tmp_path)
    return proc.returncode == 0

def passes_filters(example: dict) -> bool:
    try:
        w, h = example["width"], example["height"]
        url = example["url"]
        nsfw1 = example["nsfw_score_opennsfw2"]
        nsfw2 = example["nsfw_score_gantman"]
    except KeyError:
        return False

    return (
        w >= 256 and h >= 256 and
        (url.endswith('.jpg') or url.endswith('.jpeg')) and
        nsfw1 < 0.1 and nsfw2 < 0.1
    )

def main(
    output_dir: str = "Cover",
    max_images: int = 50,
    max_workers: int = 16,
):
    os.makedirs(output_dir, exist_ok=True)

    ds = load_dataset("kakaobrain/coyo-700m", split="train", streaming=True)
    it = iter(ds)
    success = 0
    futures = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while len(futures) < max_workers and success + len(futures) < max_images:
            try:
                ex = next(it)
            except StopIteration:
                break
            if passes_filters(ex):
                fut = executor.submit(download_image_bytes, ex["url"])
                futures[fut] = ex["url"]

        while futures and success < max_images:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for fut in done:
                url = futures.pop(fut)
                res = fut.result()
                if res:
                    content, ext = res
                    if is_image_valid(content):
                        if is_jpeglib_safe(content):
                            fn = f"image_{success:06d}{ext}"
                            with open(os.path.join(output_dir, fn), 'wb') as f:
                                f.write(content)
                            success += 1
                            print(f"[{success}/{max_images}] 저장: {fn}")
                        else:
                            print(f"[SKIP] jpeglib 파싱 실패: {url}")

                if success + len(futures) < max_images:
                    for ex in it:
                        if passes_filters(ex):
                            fut2 = executor.submit(download_image_bytes, ex["url"])
                            futures[fut2] = ex["url"]
                            break

        for fut in futures:
            fut.cancel()

    print(f"\n✅ 완료: 저장된 이미지 수 = {success}")

if __name__ == "__main__":
    main(output_dir=str(_checkout_data_path('COYO700M/Cover/')), max_images=100000, max_workers=32)

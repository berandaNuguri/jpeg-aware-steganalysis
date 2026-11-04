from jpeg_aware_steganalysis._paths import data_path as _checkout_data_path
import os
import io
import csv
import random
import requests
import numpy as np
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from PIL import Image
from typing import Dict, Optional, Tuple

def load_content(url: str, timeout: int = 10) -> Optional[Tuple[bytes, str]]:
    try:
        resp = requests.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()

        ctype = resp.headers.get("Content-Type", "").lower()
        if "image" not in ctype:
            return None

        ext = os.path.splitext(url.split("?")[0])[1].lower()
        if ext not in [".png", ".tif", ".tiff"]:
            return None

        content = resp.content
        return content, ext

    except Exception:
        return None


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


def _metadata_filtering(ex: Dict, min_size: int = 256) -> bool:
    try:
        w, h = ex["width"], ex["height"]
        url = ex["url"].lower()

        if not (url.endswith(".png") or url.endswith(".tif") or url.endswith(".tiff")):
            return False

        if ex["nsfw_score_opennsfw2"] >= 0.1:
            return False
        if ex["nsfw_score_gantman"] >= 0.1:
            return False

    except KeyError:
        return False

    return w >= min_size and h >= min_size


def _ensure_csv_with_header(csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_name", "image_url"])


def main(
    output_dir: str,
    max_images: int = 15000,
    max_workers: int = 32,
    min_size: int = 256,
    seed: int = 42,
    images_per_shard: int = 1000000,
    csv_path: Optional[str] = None,
):
    os.makedirs(output_dir, exist_ok=True)

    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(os.path.normpath(output_dir)), "saved_links.csv")

    _ensure_csv_with_header(csv_path)

    print("Loading COYO700M metadata shards (streaming=False)...")
    dataset = load_dataset("kakaobrain/coyo-700m", split="train", streaming=False)

    num_saved = 0

    dataset_len = len(dataset)
    shard_count = max(1, (dataset_len + images_per_shard - 1) // images_per_shard)
    print(f"Dataset size: {dataset_len}, shard_count: {shard_count}")

    shards = list(range(shard_count))

    random.seed(seed)
    random.shuffle(shards)
    print(f"Shuffled {shard_count} shards.")

    with open(csv_path, "a", newline="", encoding="utf-8") as f_csv:
        csv_writer = csv.writer(f_csv)

        for shard_id in shards:
            print(f"\n📦 Loading shard {shard_id+1}/{shard_count} ...")

            shard = dataset.shard(num_shards=shard_count, index=shard_id)
            shard_list = list(shard)

            random.shuffle(shard_list)
            futures = {}
            idx = 0
            total_rows = len(shard_list)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                while len(futures) < max_workers and idx < total_rows:
                    ex = shard_list[idx]
                    idx += 1

                    if not _metadata_filtering(ex, min_size=min_size):
                        continue

                    fut = executor.submit(load_content, ex["url"])
                    futures[fut] = ex["url"]

                while futures and num_saved < max_images:
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)

                    for fut in done:
                        url = futures.pop(fut)
                        result = fut.result()

                        if result:
                            content, ext = result
                            if _check_rgb(content, min_size=min_size):
                                image_name = f"image_{num_saved:06d}{ext}"
                                out_path = os.path.join(output_dir, image_name)

                                # 실제 파일 저장 성공한 경우에만 CSV 기록
                                with open(out_path, "wb") as f_img:
                                    f_img.write(content)

                                csv_writer.writerow([image_name, url])
                                f_csv.flush()

                                num_saved += 1
                                print(f"[{num_saved}/{max_images}] Saved: {image_name}")

                        while idx < total_rows and len(futures) < max_workers:
                            ex = shard_list[idx]
                            idx += 1

                            if not _metadata_filtering(ex, min_size=min_size):
                                continue

                            fut2 = executor.submit(load_content, ex["url"])
                            futures[fut2] = ex["url"]

                for fut in futures:
                    fut.cancel()

            if num_saved >= max_images:
                break

    print(f"\nCompleted: {num_saved} images saved !!")
    print(f"CSV saved: {csv_path}")


if __name__ == "__main__":
    main(
        output_dir=str(_checkout_data_path('Test/Spatial_30000/Cover/')),
        max_images=30000,
        images_per_shard=50000,
        max_workers=32,
        min_size=256,
        seed=2025,
        csv_path=str(_checkout_data_path('Test/Spatial_30000/saved_links.csv')),
    )
 
from jpeg_aware_steganalysis._paths import data_path as _checkout_data_path
import os
import io
import random
import glob
import cv2
import jpeglib
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

Image.MAX_IMAGE_PIXELS = None

def process_image(task):
    path, output_dir, target_qf, quant_libraries, resize = task
    basename = os.path.splitext(os.path.basename(path))[0]
    ext = os.path.splitext(path)[1].lower()
    lib = random.choice(quant_libraries)

    try:
        img = Image.open(path).convert("L")

        # ====== 리사이즈 ======
        if resize is not None:
            try:
                img = img.resize((resize, resize), Image.Resampling.LANCZOS)
            except AttributeError:
                img = img.resize((resize, resize), Image.LANCZOS)

        # -------------------------------
        # 1) Pillow branch
        # -------------------------------
        if lib == 'pillow':
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=int(target_qf))

        # -------------------------------
        # 2) OpenCV branch
        # -------------------------------
        elif lib == 'opencv':
            # PIL → numpy(BGR)
            arr = np.array(img)[:, :, ::-1]
            success, encimg = cv2.imencode(
                '.jpg', arr, [cv2.IMWRITE_JPEG_QUALITY, int(target_qf)]
            )
            if not success:
                return False
            buf = io.BytesIO(encimg.tobytes())

        # -------------------------------
        # 3) jpeglib branch
        # -------------------------------
        elif lib.startswith('jpeglib'):
            out_path = os.path.join(output_dir, basename+'.jpg')
            jpeglib.version.set(lib.split('_')[-1])

            if (resize is not None) or (ext in ('.png', '.PNG', '.tiff', '.TIFF', '.pgm', '.PGM')):
                im = jpeglib.from_spatial(np.array(img, dtype=np.uint8))
                im.write_spatial(out_path, quality=int(target_qf))
                return True
            else:
                # JPEG→DCT 복사 후 품질 재설정
                dj = jpeglib.read_dct(path)
                dj.write_dct(out_path, quality=int(target_qf))
                return True

        else:
            raise ValueError(f"Unsupported library: {lib}")
        
        # -------------------------------
        # 공통 파일 저장
        # -------------------------------
        out_name = basename + '.jpg'
        out_path = os.path.join(output_dir, out_name)
        with open(out_path, 'wb') as f:
            f.write(buf.getvalue())
        return True

    except Exception as e:
        print(f"[ERROR][{lib}] {os.path.basename(path)}: {e}")
        return False


def main():
    IMAGE_ROOT    = str(_checkout_data_path('COYO700M/PNG_30000_v2/Cover/'))
    OUTPUT_ROOT   = str(_checkout_data_path('COYO700M/PNG_30000_v2/'))
    RESIZE        = 256 
    SEED          = 42
    NUM_WORKERS   = 16
    TARGET_QF     = 75

    QUANT_LIBRARIES = ['pillow']
    random.seed(SEED)
    np.random.seed(SEED)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    all_paths = (
        glob.glob(os.path.join(IMAGE_ROOT, '*.jpg')) +
        glob.glob(os.path.join(IMAGE_ROOT, '*.jpeg')) +
        glob.glob(os.path.join(IMAGE_ROOT, '*.png')) +
        glob.glob(os.path.join(IMAGE_ROOT, '*.pgm')) +
        glob.glob(os.path.join(IMAGE_ROOT, '*.tif')) +
        glob.glob(os.path.join(IMAGE_ROOT, '*.tiff'))
    )

    out_dir = os.path.join(OUTPUT_ROOT, f'QF{TARGET_QF}', 'Cover')
    if RESIZE is not None:
        out_dir = os.path.join(OUTPUT_ROOT, f'R{RESIZE}_QF{TARGET_QF}', 'Cover')
    os.makedirs(out_dir, exist_ok=True)

    tasks = [
        (p, out_dir, TARGET_QF, QUANT_LIBRARIES, RESIZE)
        for p in all_paths
    ]
    info_resize = f", Resize={RESIZE}x{RESIZE}" if RESIZE is not None else ""
    print(f"[INFO] {len(tasks)} images to process with QF={TARGET_QF} using {QUANT_LIBRARIES}{info_resize}")

    if NUM_WORKERS > 1:
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
            list(tqdm(ex.map(process_image, tasks),
                      total=len(tasks), desc="Compressing"))
    else:
        for t in tqdm(tasks, desc="Compressing"):
            process_image(t)

    print("[INFO] Done.")


if __name__ == '__main__':
    main()

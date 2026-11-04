from jpeg_aware_steganalysis._paths import data_path as _checkout_data_path
import os
import argparse
import random
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

def main(args):
    # CSV 로드
    csv_path = os.path.join(args.csv_root, args.data, f'{args.stego_method}_test.csv')
    test_df = pd.read_csv(csv_path)

    if args.target_qt is not None:
        # QTable 정보 불러오기
        with open(args.qtable_info_path, 'rb') as f:
            stats = pickle.load(f)

        qt_key = f'QT{args.target_qt}'
        if qt_key not in stats:
            raise KeyError(f"{args.target_qt} not in {args.qtable_info_path}")

        target_table = np.array(stats[qt_key]['qtable'][:64])

        def match_qtable(row):
            img_path = os.path.join(args.dataset_root, args.data, "Cover", row['image_name'])
            try:
                with Image.open(img_path) as img:
                    qtables = img.quantization
                    if not qtables or 0 not in qtables:
                        return False
                    ytable = np.array(qtables[0])
                return np.array_equal(ytable, target_table)
            except Exception:
                return False

        # target_qt 매칭 필터링
        test_df = test_df[test_df.apply(match_qtable, axis=1)]
        print(f"→ Filtered dataset to QT={args.target_qt}, remaining {len(test_df)} cover samples")

        # 1024x1024 이상 비율 계산
        if len(test_df) > 0:
            large_df = test_df[(test_df['width'] >= 1024) & (test_df['height'] >= 1024)]
            ratio = len(large_df) / len(test_df) * 100
            print(f"→ Among them, images ≥1024x1024: {len(large_df)*2} / {len(test_df)*2} ({ratio:.2f}%)")
        else:
            print("→ No samples matched, skipping size ratio check.")
    else:
        print("⚠ No target_qt specified. Skipping QT filtering.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check ratio of >=1024x1024 images for target_qt')

    parser.add_argument('--dataset_root', type=str, default=str(_checkout_data_path('')), help='Root path to the dataset.')
    parser.add_argument('--csv_root', type=str, default='./csv/', help='Root path to the CSV files.')
    parser.add_argument('--data', type=str, default='NSR_Crawling_2025/JPG_Test/', help='Dataset name')
    parser.add_argument('--stego_method', type=str, default='nsF5_0.4', help='Steganography method name.')
    parser.add_argument('--qtable_info_path', type=str, default=str(_checkout_data_path('NSR_Crawling_2025/JPG_Test/qtable_info.pkl')),
                        help='Path to qtable_info.pkl mapping QT→qtable.')
    parser.add_argument('--target_qt', type=str, default='95_1',
                        help='Target QT key (e.g. 90_1, 80_2).')

    args = parser.parse_args()
    main(args)

from jpeg_aware_steganalysis._paths import data_path as _checkout_data_path
import os
import re
import argparse

def main(start_num: int, target_dir: str):
    # 폴더 존재 여부 체크
    if not os.path.isdir(target_dir):
        print(f"지정한 폴더가 없습니다: {target_dir}")
        return

    # image_000000.jpg 같은 형식 매칭
    pattern = re.compile(r'^(image_)(\d+)(\.[^.]+)$')

    files = []
    for fname in os.listdir(target_dir):
        m = pattern.match(fname)
        if m:
            prefix, num_str, ext = m.groups()
            idx = int(num_str)
            files.append((idx, fname, prefix, num_str, ext))

    if not files:
        print("매칭되는 파일이 없습니다. (image_xxxxxx.ext 형식)")
        return

    # 번호 기준으로 정렬
    files.sort(key=lambda x: x[0])

    # 자릿수(패딩) 유지: 000000 → 6자리 등
    width = len(files[0][3])

    print(f"{target_dir} 안의 {len(files)}개 파일을 번호 {start_num:0{width}d}부터 재번호 부여합니다.")

    # 충돌 방지: 1단계 – 임시 이름으로 변경
    temp_names = {}
    for i, (_, old_name, prefix, _, ext) in enumerate(files):
        old_path = os.path.join(target_dir, old_name)
        tmp_name = f"__tmp_rename__{i}__{ext}"
        tmp_path = os.path.join(target_dir, tmp_name)
        os.rename(old_path, tmp_path)
        temp_names[tmp_name] = (i, prefix, ext)

    # 2단계 – 최종 이름으로 변경
    for tmp_name, (i, prefix, ext) in temp_names.items():
        tmp_path = os.path.join(target_dir, tmp_name)
        new_idx = start_num + i
        new_num_str = f"{new_idx:0{width}d}"
        new_name = f"{prefix}{new_num_str}{ext}"
        new_path = os.path.join(target_dir, new_name)

        # 혹시 이미 있으면 경고만 찍고 스킵
        if os.path.exists(new_path):
            print(f"경고: {new_name} 이미 존재해서 건너뜁니다.")
        else:
            os.rename(tmp_path, new_path)

    print("완료!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="지정한 폴더의 image_xxxxxx.ext 파일 번호를 재지정합니다."
    )
    parser.add_argument(
        "--start",
        type=int,
        default=15000,
        help="새로 시작할 번호 (예: 15000 → image_015000.xxx부터 시작)"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=str(_checkout_data_path('COYO700M/Spatial_15000/Cover/')),
        help="이미지가 있는 폴더 경로 (기본값: 현재 폴더)"
    )
    args = parser.parse_args()
    main(args.start, args.dir)

import pandas as pd
import os

# --- Configs ---
input_csv = './csv/COYO700M/QF90/nsF5_0.4_train.csv'
output_csv = './csv/COYO700M/QF90/nsF5_0.4_train.csv'

insert_folder = 'Base100000'
# --- ---

def modify_image_paths(input_path, output_path):
    """
    CSV 파일의 image_path를 수정하여 새 파일로 저장합니다.
    """
    # 1. 파일 존재 여부 확인
    if not os.path.exists(input_path):
        print(f"❌ 오류: 입력 파일 '{input_path}'을(를) 찾을 수 없습니다. 파일 경로를 확인해주세요.")
        return

    # 2. CSV 파일을 pandas DataFrame으로 읽기
    print(f"📖 '{input_path}' 파일을 읽는 중...")
    df = pd.read_csv(input_path)

    # 3. 'image_path' 컬럼 수정
    print("🛠️  경로를 수정하는 중...")
    df['image_path'] = df['image_path'].apply(
        lambda path: path.replace('COYO700M/Base100000_QTSwap50/', f'COYO700M/{insert_folder}/', 1)
    )

    # 4. 수정된 DataFrame을 새 CSV 파일로 저장 (인덱스는 제외)
    print(f"💾 '{output_path}' 파일로 저장하는 중...")
    df.to_csv(output_path, index=False)

    print(f"\n✅ 작업 완료! 수정된 파일이 '{output_path}'에 저장되었습니다.")
    print("\n--- 수정 결과 미리보기 (상위 5개) ---")
    print(df.head())


# 스크립트 실행
if __name__ == '__main__':
    modify_image_paths(input_csv, output_csv)
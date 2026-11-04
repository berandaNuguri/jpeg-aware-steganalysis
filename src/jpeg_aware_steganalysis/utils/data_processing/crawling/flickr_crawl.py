from jpeg_aware_steganalysis._paths import data_path as _checkout_data_path
import os
import time
import requests
from flickrapi import FlickrAPI
from tqdm import tqdm

# 1) 사용자 설정
API_KEY    = 'YOUR_FLICKR_API_KEY'
API_SECRET = 'YOUR_FLICKR_API_SECRET'
SAVE_DIR   = str(_checkout_data_path('FlickrSteg/'))
os.makedirs(SAVE_DIR, exist_ok=True)

# 2) 필터 조건 (public domain 제외)
#  1: CC BY-NC-SA  2: CC BY-NC  3: CC BY-NC-ND
#  4: CC BY       5: CC BY-SA  6: CC BY-ND
#  8: US Govt Work
LICENSES = '0,9,10'

MIN_DIM = 256
MAX_DIM = 1024

# 3) Flickr API 초기화
flickr = FlickrAPI(API_KEY, API_SECRET, format='parsed-json')

def crawl_page(page):
    resp = flickr.photos.search(
        license=LICENSES,
        extras='original_format,url_o,width_o,height_o',
        media='photos',
        per_page=500,
        page=page,
        safe_search=1,
        content_type=1,
        sort='date-posted-desc'
    )
    photos = resp['photos']['photo']
    results = []
    for p in photos:
        if p.get('originalformat') != 'png' or 'url_o' not in p:
            continue
        w = int(p.get('width_o', 0))
        h = int(p.get('height_o', 0))
        if not (MIN_DIM < w < MAX_DIM and MIN_DIM < h < MAX_DIM):
            continue
        results.append(p['url_o'])
    return results

def download_images(urls):
    for url in tqdm(urls, desc='다운로드 중'):
        fname = os.path.join(SAVE_DIR, os.path.basename(url))
        if os.path.exists(fname):
            continue
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                with open(fname, 'wb') as f:
                    f.write(r.content)
            time.sleep(0.1)
        except Exception as e:
            print(f"[오류] {url} -> {e}")

def main(max_pages=10):
    all_urls = []
    for page in range(1, max_pages+1):
        urls = crawl_page(page)
        if not urls:
            break
        all_urls.extend(urls)
    print(f"총 수집된 이미지 URL: {len(all_urls)}")
    download_images(all_urls)

if __name__ == '__main__':
    main(max_pages=20)

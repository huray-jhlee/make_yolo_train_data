import os
import cv2
from tqdm import tqdm
# 이미지 경로 설정
image_dir = "/home/ai04/jh/codes/240923_additional_data/test_data/images"

# 이미지 파일들 리스트 가져오기
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

# 이미지 파일 개수와 로드 상태 체크
invalid_images = []
for image_file in tqdm(image_files):
    image_path = os.path.join(image_dir, image_file)
    image = cv2.imread(image_path)
    
    # 이미지가 제대로 열리지 않은 경우
    if image is None:
        invalid_images.append(image_file)

# 결과 출력
if invalid_images:
    print(f"다음 파일들은 열리지 않았습니다 ({len(invalid_images)}개):")
    for invalid_image in invalid_images:
        print(invalid_image)
else:
    print("모든 이미지가 정상적으로 열렸습니다.")
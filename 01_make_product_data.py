import os
import cv2
import shutil
from glob import glob
from tqdm import tqdm
from collections import defaultdict
import threading

lock = threading.Lock()

# tqdm을 사용하여 각 스레드에서 작업 진행 상황을 모니터링하는 함수
def save_data(task_img_paths, tmp_period, used_cat, args, progress_bar):
    for task_img_path in task_img_paths:
        # used_case Check
        cat = task_img_path.split("/")[-3]
        if cat not in used_cat:
            with lock:
                progress_bar.update(1)
            continue

        # get_key_idx
        key_idx = os.path.basename(task_img_path).split(".")[0].replace("_", "-")
        filename = f"{tmp_period}_{key_idx}"

        # img 저장
        img = cv2.imread(task_img_path)
        img_save_path = os.path.join(args.save_dir, "images", f"{filename}.png")

        # label 저장
        task_label_path = task_img_path.replace("원천데이터", "label_txt").replace(".jpg", ".txt")
        label_save_path = img_save_path.replace("/images/", "/labels/").replace(".png", ".txt")
        
        if not os.path.exists(task_label_path):
            print("label not exists")
            with lock:
                progress_bar.update(1)
            continue
        
        if os.path.exists(img_save_path) and os.path.exists(label_save_path):
            with lock:
                progress_bar.update(1)
            continue
        
        cv2.imwrite(img_save_path, img)
        shutil.copy(task_label_path, label_save_path)

        # 작업이 끝날 때마다 진행 상태 업데이트
        with lock:
            progress_bar.update(1)

def main(args):
    path_dict = defaultdict(dict)
    if args.task is None:
        tasks = os.listdir(args.data_dir)
    elif args.task == "train":
        tasks = ['1.Training']
    else :
        tasks = ['2.Validation']

    for task in tasks:
        task_dir = os.path.join(args.data_dir, task)
        task_img_paths = glob(os.path.join(task_dir, "**", "원천데이터", "**", "*.jpg"), recursive=True)
        task_label_paths = glob(os.path.join(task_dir, "**", "label_txt", "**", "*.txt"), recursive=True)
        path_dict[task]['images'] = task_img_paths
        path_dict[task]['labels'] = task_label_paths

    period = "product"

    os.makedirs(os.path.join(args.save_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "labels"), exist_ok=True)

    # train and val, union list
    used_cat = ["주류", "주류1", "주류2", "소스", "소스1", "소스2", "과자", "과자1", "과자2", "과자3", "과자4", \
        "면류", "음료", "음료1", "음료2", "음료3", "커피차", "커피차1", "커피차2", "디저트", \
        "유제품", "통조림_안주", "상온HMR1", "상온HMR2", "상온HMR3"]
    
    for task in tasks:
        print(task)
        tmp_period = f"{period}-{'val' if task == '2.Validation' else 'train'}"
        task_img_paths = path_dict[task]['images']

        # tqdm progress bar 생성
        progress_bar = tqdm(total=len(task_img_paths), desc=f"Processing {task}")

        # 스레드 리스트
        threads = []
        num_threads = args.workers  # argparse에서 받은 스레드 개수
        chunk_size = len(task_img_paths) // num_threads

        # 스레드 생성 및 시작
        for i in range(num_threads):
            start_idx = i * chunk_size
            if i == num_threads - 1:  # 마지막 스레드는 나머지 할당
                task_img_paths_chunk = task_img_paths[start_idx:]
            else:
                task_img_paths_chunk = task_img_paths[start_idx:start_idx + chunk_size]

            thread = threading.Thread(target=save_data, args=(task_img_paths_chunk, tmp_period, used_cat, args, progress_bar))
            threads.append(thread)
            thread.start()

        # 모든 스레드가 완료될 때까지 대기
        for thread in threads:
            thread.join()

        # 모든 작업이 끝난 후 진행 바 종료
        progress_bar.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/data/aihub/product_dataset/119.상품_이미지_데이터/01.데이터/")
    parser.add_argument("--save_dir", type=str, default="/home/ai04/jh/codes/240923_additional_data/test_data_product")
    parser.add_argument("--workers", type=int, default=8, help="Number of threads to use")
    parser.add_argument("--task", type=str, default=None, help="default value is None, select train or val")
    args = parser.parse_args()
    main(args)
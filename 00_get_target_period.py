import os
import cv2
import pickle
import threading
import numpy as np
import pandas as pd

from glob import glob
from tqdm import tqdm
from collections import defaultdict
from sqlalchemy import create_engine

from utils import convert_to_yolo_format, defaultdict_to_dict
from config import DevServer, MainServer

def save_datas(data_list):
    for img_path, info_dict in tqdm(data_list):
        file_key = info_dict['file_key']
        labels = info_dict['labels']
        
        min_period, min_idx = min(file_key, key=lambda x: x[1])
        filename = f"{min_period}_{str(min_idx).zfill(7)}"

        new_img_path = os.path.join(args.save_dir, "images", f"{filename}.png")
        new_label_path = os.path.join(args.save_dir, "labels", f"{filename}.txt")
        
        # img load and save
        img = cv2.imread(img_path)
        cv2.imwrite(new_img_path, img)
        
        # save_label
        np.savetxt(new_label_path, labels, fmt=["%d", "%.6f", "%.6f", "%.6f", "%.6f"])

def main(args):
    # 특정 period로 dataframe 쿼리
    # crop_info로 한번 거르기
    # 이미 관련해서 한번 정리는 완료
    # 2, 3, 4, 6, 13, 102 정도만 먼저 준비
    
    # 음식이 아닌 클래스
    # 음식아님 : 1838
    # 근데 여기서 체크할 점.
    # 음식아님이 다 가공식품은 아니다.
    # 기존에 상현님이 가공식품은 ai-hub에서만 가져다가 사용했다고 했었음.
    # ai-hub, 가공식품은 DB에 존재하지 않을텐데?
    # https://huraypositive.atlassian.net/wiki/spaces/~712020afccd78ac9874708b0f7c01e25e74962/pages/3201302529/Detection+_240920#%EC%B6%94%EA%B0%80-%EB%82%B4%EC%9A%A9
    
    
    query = f"""
            SELECT h.idx as idx, c.org_path as origin_path, c.crop_info as crop_info, h.check_class_id as class_id, h.class_id as prev_class_id, h.period
            FROM huray_image_data as h
            JOIN crop_table as c
            ON h.idx = c.idx
            WHERE h.period IN ({','.join(map(str, args.periods))})
            """
            
    df = pd.read_sql(query, ENGINE)
    
    if df['crop_info'].isna().any():
        # raise Exception("crop_info null")
        print(f"before dropna in crop_info : {len(df)}")
        df = df.dropna(subset=['crop_info'])
        print(f"after dropna in crop_info : {len(df)}")
    
    # class_id, prev_class_id가 둘다 null인 경우가 존재
    # 이것도 데이터에서 제외
    
    if df['class_id'].isna().any():
        df['class_id'] = df['class_id'].fillna(df['prev_class_id'])
        df = df.dropna(subset=['class_id'])
        df['class_id'] = df['class_id'].astype(int)
    
    data_dict = defaultdict(lambda: defaultdict(list))
    
    print(f"처리할 데이터  : {len(df)}")
    
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        idx = row['idx']
        period = row['period']
        
        # img_path
        img_path = row['origin_path'].replace("/data3/", "/data/")
        if not os.path.exists(img_path):
            continue
        
        img = cv2.imread(img_path)
        height, width, _ = img.shape
        
        # label check
        if row['class_id'] != 1838:
            label = 0
        else :
            # 음식 아님인 경우인데, 음식아님 = 가공식품이 아니니까 그냥 넘기기
            continue
        
        # bbox 
        splited_bbox_info = row['crop_info'].replace("[", "").replace("]", "").split(",")
        splited_bbox_info = [sb.strip() for sb in splited_bbox_info]
        
        if args.thr is not None and len(splited_bbox_info) == 5:
            bbox_info, conf_score = splited_bbox_info[:4], splited_bbox_info[-1]
            if float(conf_score) < args.thr:
                print("thresholding")
                continue
        else:
            bbox_info = splited_bbox_info[:4]  # 맨 마지막에 score가 들어있는 경우...
        
        10
        yolo_bbox = convert_to_yolo_format(bbox_info, width, height)
        
        data_dict[img_path]["file_key"].append((period, idx))
        data_dict[img_path]["labels"].append([label]+yolo_bbox)
        
        # TODO: remove
        # cnt += 1
        # if cnt == limit_cnt:
        #     break
    
    print("data_dict 생성완료")
    serializable_dict = defaultdict_to_dict(data_dict)
    with open(f"{'-'.join(args.periods)}.pkl", "wb") as f:
        pickle.dump(serializable_dict, f, pickle.HIGHEST_PROTOCOL)
    
    total_data_list = list(data_dict.items())
    total_len = len(total_data_list)
    chunk_size = (total_len + args.workers - 1) // args.workers
    
    thread_list = []
    for i in range(args.workers):
        start_idx = i * chunk_size
        end_idx = min((i+1)*chunk_size, total_len)
        data_slice = total_data_list[start_idx:end_idx]
        
        thread = threading.Thread(target=save_datas, args=(data_slice,))
        thread_list.append(thread)
        thread.start()
    
    for thread in thread_list:
        thread.join()
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--periods", "-p", type=str, default="5")
    # parser.add_argument("--save_dir", "-s", type=str, default="/data/jh/detection_dataset/240924_dataset")
    parser.add_argument("--save_dir", "-s", type=str, default="/home/ai04/jh/codes/240923_additional_data/test_data")
    parser.add_argument("--server", "-v", type=str, default="main", help="dev or main")
    parser.add_argument("--workers", "-w", type=int, default=8)
    parser.add_argument("--thr", type=float, default=None)
    # 2, 6, 13, 102
    args = parser.parse_args()
    
    args.periods = args.periods.split(",")

    if args.server == "dev":
        server_info = DevServer
    elif args.server == "main":
        server_info = MainServer
    else :
        raise Exception("Invalid server arguments")
    
    os.makedirs(os.path.join(args.save_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "labels"), exist_ok=True)
    
    ENGINE = create_engine(f"mysql+pymysql://{server_info.DB_USERNAME}:{server_info.DB_PASSWORD}@{server_info.DB_HOST}/{server_info.DB_NAME}")
    
    main(args)
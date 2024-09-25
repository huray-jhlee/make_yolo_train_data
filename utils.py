import os
import cv2
from collections import defaultdict

# visualization yolo label
def draw_bbox(image, bbox, color=(0, 255, 0), thickness=2):
    bbox = [int(x) for x in bbox]
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
    return image

def draw_with_label(image_path, bbox_info):
    img_obj = cv2.imread(image_path)
    img_obj = cv2.cvtColor(img_obj, cv2.COLOR_BGR2RGB)
    img_width, img_height, _ = img_obj.shape
    
    if isinstance(bbox_info, list):
        bbox_info_list = bbox_info
    
    elif isinstance(bbox_info, str) and os.path.isfile(bbox_info):
        with open(bbox_info, 'r') as f:
            label_datas = f.readlines()
        bbox_info_list = [list(map(float, x.strip().split(" ")))[1:] for x in label_datas]
    else :
        raise Exception("invalid bbox_info")
    
    for x_center, y_center, bbox_width, bbox_height in bbox_info_list:
        x_center *= img_width
        y_center *= img_height
        bbox_width *= img_width
        bbox_height *= img_height
        
        x_min = x_center - (bbox_width / 2)
        x_max = x_center + (bbox_width / 2)
        y_min = y_center - (bbox_height / 2)
        y_max = y_center + (bbox_height / 2)
        
        draw_obj = draw_bbox(img_obj, [x_min, y_min, x_max, y_max])
    
    return draw_obj

def convert_to_yolo_format(bbox, width, height):
    bbox = list(map(int, bbox))
    
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    
    box_width = x_max - x_min
    box_height = y_max - y_min
    
    norm_x_center = x_center / width
    norm_y_center = y_center / height
    norm_box_width = box_width / width
    norm_box_height = box_height / height
    
    return [norm_x_center, norm_y_center, norm_box_width, norm_box_height]

def defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d
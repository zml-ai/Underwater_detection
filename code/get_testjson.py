import json
import os
import cv2
import mmcv
from tqdm import tqdm

# Refernce = r'F:/study/0_Project/sea_detection/mmdet_luo/data/train_new.json'
# with open(Refernce,'r') as load_f:
#     f = json.load(load_f)
#
# a=f["images"]
# b=f["annotations"]
# # license=f["license"]
# # info=f["info"]
# categories=f["categories"]
# categories= sorted(categories, key=lambda d:d['id'], reverse = False)
# del categories[0]
# print(categories)
# 根路径，里面包含images(图片文件夹)，annos.txt(bbox标注)，classes.txt(类别标签),以及annotations文件夹(如果没有则会自动创建，用于保存最后的json)
root_path = r'F:\\study\\0_Project\\sea_detection\\2020dataset\\test-B-image'
# 用于创建训练集或验证集
phase = 'test'


# 读取images文件夹的图片名称
indexes = [f for f in os.listdir(os.path.join(root_path, 'image'))]


img=[]
img1=[]
for k, index in enumerate(tqdm(indexes)):
    # 用opencv读取图片，得到图像的宽和高
    im = cv2.imread(os.path.join(root_path, 'image/') + index)

    height,width,dim = im.shape

    # 添加图像的信息到dataset中
    img.append({'file_name': index,
                'id': k+1,
                'width': width,
                'height': height})
    # if width < 500:
    #     img.append({'file_name': index,
    #                               'id': k,
    #                               'width': width,
    #                               'height': height})
    # else:
    #     img1.append({'file_name': index,
    #                 'id': k,
    #                 'width': width,
    #                 'height': height})
# print(len(img))
print(len(img))

categories1=[{"supercategory": "holothurian", "id": 1, "name": "holothurian"},
             {"supercategory": "echinus", "id": 2, "name": "echinus"},
             {"supercategory": "scallop", "id": 3, "name": "scallop"},
             {"supercategory": "starfish", "id": 4, "name": "starfish"},
             {"supercategory": "waterweeds", "id": 5, "name": "waterweeds"},
             ]
# categories1= sorted(categories1, key=lambda d:d['id'], reverse = False)
print(categories1)
# categories2=[{"supercategory": "\u6807\u8d34\u6c14\u6ce1", "id": 3, "name": "\u6807\u8d34\u6c14\u6ce1"},
#  {"supercategory": "\u6807\u8d34\u6b6a\u659c", "id": 1, "name": "\u6807\u8d34\u6b6a\u659c"},
#  {"supercategory": "\u6807\u8d34\u8d77\u76b1", "id": 2, "name": "\u6807\u8d34\u8d77\u76b1"}]
# categories2= sorted(categories2, key=lambda d:d['id'], reverse = False)
# print(categories2)
seatestdataset={
"images": img,
"annotations": [],
"categories": categories1
}
# ptestdataset={
#     "info": info,
#     "licenses": license,
#     "images":img1,
#     "annotations": [],
#     "categories": categories2
# }

with open(r"F:/study/0_Project/sea_detection/mmdet_luo/data/testB.json", 'w') as f:
  json.dump(seatestdataset, f)
# with open(r"D:\chongqing1_round1_testA_20191223\ptest2017.json", 'w') as f:
#   json.dump(ptestdataset, f)




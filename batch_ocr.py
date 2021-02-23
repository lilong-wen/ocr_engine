import cv2
import codecs
import glob
import ocr_engine
import json

scanned_folder = './data/scanned/'
camera_folder = './data/camera/'
# scanned_folder = './data/scanned_small/'
# camera_folder = './data/camera_small/'


reader = ocr_engine.Reader()
scanned_imgs = glob.glob(scanned_folder + "*")
camera_imgs = glob.glob(camera_folder + "*")

scanned_result_json = './data/scanned_result.json'
camera_result_json = './data/camera_result.json'
# scanned_result_json = './data/scanned_result_small.json'
# camera_result_json = './data/camera_result_small.json'


scanned_result_dict = {}
for scanned_item in scanned_imgs:
    # item_result = reader.readtext(scanned_item, detail=1, paragraph=True)
    item_result = reader.readtext(scanned_item, detail=1, paragraph=False)
    img = cv2.imread(scanned_item, 0)
    h, w = img.shape
    new_item_result = []
    print(scanned_item)
    for item in item_result:
        tmp = []
        for coord_item in item[0]:
            new_x = round(coord_item[0] / w, 2)
            new_y = round(coord_item[1] / h, 2)
            tmp.append([new_x, new_y])

        new_item_result.append([tmp, item[1]])

    scanned_result_dict[str(scanned_item.split("/")[-1][:-4])] = str(new_item_result)

with codecs.open(scanned_result_json, 'w', encoding='utf-8') as scanned_f:
    json.dump(scanned_result_dict, scanned_f, ensure_ascii=False)


camera_result_dict = {}
for camera_item in camera_imgs:
    # item_result = reader.readtext(camera_item, detail=1, paragraph=True)
    item_result = reader.readtext(camera_item, detail=1, paragraph=False)
    img = cv2.imread(camera_item, 0)
    h, w = img.shape
    print(camera_item)
    new_item_result = []
    for item in item_result:
        tmp = []
        for coord_item in item[0]:
            new_x = round(coord_item[0] / w, 2)
            new_y = round(coord_item[1] / h, 2)
            tmp.append([new_x, new_y])
        new_item_result.append([tmp, item[1]])
    camera_result_dict[str(camera_item.split("/")[-1][:-4])] = str(new_item_result)

with codecs.open(camera_result_json, 'w', encoding='utf-8') as camera_f:
    json.dump(camera_result_dict, camera_f, ensure_ascii=False)

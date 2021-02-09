import codecs
import glob
import ocr_engine
import json

scanned_folder = './data/scanned/'
camera_folder = './data/camera/'

reader = ocr_engine.Reader()
scanned_imgs = glob.glob(scanned_folder + "*")
camera_imgs = glob.glob(camera_folder + "*")

scanned_result_json = './data/scanned_result.json'
camera_result_json = './data/camera_result.json'

scanned_result_dict = {}
for scanned_item in scanned_imgs:
    item_result = reader.readtext(scanned_item, detail=1, paragraph=True)
    print(scanned_item)
    scanned_result_dict[str(scanned_item.split("/")[-1][:-4])] = item_result

with codecs.open(scanned_result_json, 'w', encoding='utf-8') as scanned_f:
    json.dump(scanned_result_dict, scanned_f, ensure_ascii=False)


camera_result_dict = {}
for camera_item in camera_imgs:
    item_result = reader.readtext(camera_item, detail=1, paragraph=True)
    print(camera_item)
    camera_result_dict[str(camera_item.split("/")[-1][:-4])] = item_result

with codecs.open(camera_result_json, 'w', encoding='utf-8') as camera_f:
    json.dump(camera_result_dict, camera_f, ensure_ascii=False)

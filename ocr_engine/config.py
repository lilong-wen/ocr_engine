import os

os.environ["LRU_CACHE_CAPACITY"] = "1"

BASE_PATH = os.path.dirname(__file__)

MODULE_PATH = BASE_PATH + "/weights/"

# detector parameters
DETECTOR_FILENAME = 'craft_mlt_25k.pth'

# recognizer parameters
latin_lang_list = ['en']

other_lang_list = ['ch_sim']

all_lang_list = latin_lang_list + other_lang_list
imgH = 64

number = '0123456789'
symbol  = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '
special_c0 = 'ุู'
special_c1 = 'ิีืึ'+ 'ั'
special_c2 = '่้๊๋'
special_c3 = '็์'
special_c = special_c0+special_c1+special_c2+special_c3 + 'ำ'

# All language characters
characters = {
    'all_char': 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'+\
            'ÀÁÂÃÄÅÆÇÈÉÊËÍÎÑÒÓÔÕÖØÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿąęĮįıŁłŒœŠšųŽž',
    'en_char' : 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ',
    'th_number' : '0123456789',
}

# first element is url path, second is file size
model_url = {
    'detector': ('https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip', '2f8227d2def4037cdb3b34389dcf9ec1'),
    'chinese_sim.pth': ('https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/chinese_sim.zip', '0e19a9d5902572e5237b04ee29bdb636'),
}

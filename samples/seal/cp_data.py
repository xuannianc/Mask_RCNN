import shutil
import os.path as osp
import json
import logging
import sys
import os

SOURCE_DIR = '/home/adam/Pictures/201801/vat'
TARGET_DIR = '/home/adam/workspace/github/Mask_RCNN/datasets/seal'
logger = logging.getLogger('cp_data')
logger.setLevel(logging.DEBUG)  # default log level
format = logging.Formatter("%(asctime)s %(name)-8s %(levelname)-8s %(lineno)-4d %(message)s")  # output format
sh = logging.StreamHandler(stream=sys.stdout)  # output to standard output
sh.setFormatter(format)
logger.addHandler(sh)


def copy_data_from_vat_dir(subset='train'):
    region_data_file_path = osp.join(TARGET_DIR, 'via_region_data.json')
    region_data = json.load(open(region_data_file_path))
    region_data_values = list(region_data.values())
    for idx, region_data_value in enumerate(region_data_values[80:]):
        filename = region_data_value['filename']
        src_file_path = osp.join(SOURCE_DIR, filename)
        tgt_file_path = osp.join(TARGET_DIR, subset, filename)
        if osp.exists(tgt_file_path):
            logger.debug('{} already exists'.format(filename))
        else:
            logger.debug('copy {}th file {}'.format(idx, filename))
        shutil.copy(src_file_path, tgt_file_path)


# copy_data_from_vat_dir()
copy_data_from_vat_dir(subset='val')

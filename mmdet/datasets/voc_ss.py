import os.path as osp
import xml.etree.ElementTree as ET

from .builder import DATASETS
from .voc import VOCDataset
import mmcv


@DATASETS.register_module()
class VOCSSDataset(VOCDataset):
    # VOC Dataset with SuperPixels

    def __init__(self, **kwargs):
        super(VOCSSDataset, self).__init__(**kwargs)

    def load_annotations(self, ann_file):
        data_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        for img_id in img_ids:
            filename = f'JPEGImages/{img_id}.jpg'
            ssname = f'SuperPixels/{img_id}.jpg'
            xml_path = osp.join(self.img_prefix, 'Annotations',
                                f'{img_id}.xml')
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            width = 0
            height = 0
            if size is not None:
                width = int(size.find('width').text)
                height = int(size.find('height').text)
            else:
                img_path = osp.join(self.img_prefix, 'JPEGImages',
                                    '{}.jpg'.format(img_id))
                img = Image.open(img_path)
                width, height = img.size
            data_infos.append(
                dict(id=img_id, filename=filename, width=width, height=height, ssname=ssname))

        return data_infos


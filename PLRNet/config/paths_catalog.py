import os
import os.path as osp

class DatasetCatalog(object):

    DATA_DIR = osp.abspath(osp.join(osp.dirname(__file__),
                '..','..','data'))
    
    DATASETS = {
        'data_train_small': {
            'img_dir': 'data/train/images',
            'ann_file': 'data/train/annotation-small.json'
        },
        'data_test_small': {
            'img_dir': 'data/val/images',
            'ann_file': 'data/val/annotation-small.json'
        },
        'data_train': {
            'img_dir': 'data/train/images',
            'ann_file': 'data/train/annotation.json'
        },
        'data_test': {
            'img_dir': 'data/val/images',
            'ann_file': 'data/val/annotation.json'
        },
        'ai4sf_train': {
            'img_dir': 'ai4sf_256px_area50',
            'ann_file': 'ai4sf_256px_area50/train_coco.json'
        },
        'ai4sf_val': {
            'img_dir': 'ai4sf_256px_area50',
            'ann_file': 'ai4sf_256px_area50/val_coco.json'
        },
        'ai4sf_test': {
            'img_dir': 'ai4sf_256px_area50',
            'ann_file': 'ai4sf_256px_area50/test_coco.json'
        },
    }

    @staticmethod
    def get(name):
        assert name in DatasetCatalog.DATASETS
        data_dir = DatasetCatalog.DATA_DIR
        attrs = DatasetCatalog.DATASETS[name] 
       
        args = dict(
            root = osp.join(data_dir,attrs['img_dir']),
            ann_file = osp.join(data_dir,attrs['ann_file'])
        )

        if 'train' in name:
            return dict(factory="TrainDataset", args=args)
        if ('test' in name or 'val' in name) and 'ann_file' in attrs:
            return dict(factory="TestDatasetWithAnnotations", args=args)
        raise NotImplementedError()

import cv2
import numpy as np
import os.path as osp
import rasterio
import torchvision.datasets as dset

from PIL import Image
from pycocotools.coco import COCO
from shapely.geometry import Polygon
from torch.utils.data.dataloader import default_collate


class TestDatasetWithAnnotations(dset.coco.CocoDetection):
    """
    Val/test dataset that produces the same annotation fields as TrainDataset
    (junctions, juncs_tag, juncs_index, edges_positive, bbox, mask) so that
    forward_train can be called during validation to get comparable losses.
    No data augmentation is applied.
    """

    def __init__(self, root, ann_file, transform=None):
        super(TestDatasetWithAnnotations, self).__init__(root, ann_file)
        self.root = root
        self.ids  = sorted(self.ids)
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms   = transform

    def __getitem__(self, idx):
        img_id    = self.ids[idx]
        img_info  = self.coco.loadImgs(ids=[img_id])[0]
        file_name = img_info['file_name']
        width     = img_info['width']
        height    = img_info['height']

        # --- load image ---
        img_path = osp.join(self.root, file_name)
        if img_path.lower().endswith(('.tif', '.tiff')):
            with rasterio.open(img_path) as src:
                image = src.read([1, 2, 3]).transpose(1, 2, 0).astype(np.float32) / 10000.0
        else:
            pil_img = Image.open(img_path).convert('RGB')
            image   = np.array(pil_img).astype(np.float32)
            if image.max() > 1.0:
                image /= 255.0

        # --- build annotations (same logic as TrainDataset, no augmentation) ---
        ann_ids  = self.coco.getAnnIds(imgIds=[img_id])
        ann_coco = self.coco.loadAnns(ids=ann_ids)

        ann = {
            'filename'      : file_name,
            'img_id'        : img_id,
            'junctions'     : [],
            'juncs_index'   : [],
            'juncs_tag'     : [],
            'edges_positive': [],
            'bbox'          : [],
            'width'         : width,
            'height'        : height,
        }

        pid         = 0
        instance_id = 0
        seg_mask    = np.zeros([height, width], dtype=np.float64)

        for ann_per_ins in ann_coco:
            juncs, tags = [], []
            segmentations = ann_per_ins['segmentation']

            for i, segm in enumerate(segmentations):
                segm = np.array(segm).reshape(-1, 2)
                segm[:, 0] = np.clip(segm[:, 0], 0, width  - 1e-4)
                segm[:, 1] = np.clip(segm[:, 1], 0, height - 1e-4)
                points    = segm[:-1]
                junc_tags = np.ones(points.shape[0])

                if i == 0:
                    poly = Polygon(points)
                    if poly.area > 0:
                        convex_point = np.array(poly.convex_hull.exterior.coords)
                        convex_index = [(p == convex_point).all(1).any() for p in points]
                        juncs.extend(points.tolist())
                        junc_tags[convex_index] = 2
                        tags.extend(junc_tags.tolist())
                        ann['bbox'].append(list(poly.bounds))
                        seg_mask += self.coco.annToMask(ann_per_ins)
                else:
                    juncs.extend(points.tolist())
                    tags.extend(junc_tags.tolist())
                    interior_contour = segm.reshape(-1, 1, 2)
                    cv2.drawContours(seg_mask, [np.int0(interior_contour)],
                                     -1, color=0, thickness=-1)

            idxs  = np.arange(len(juncs))
            edges = np.stack((idxs, np.roll(idxs, 1))).transpose(1, 0) + pid

            ann['juncs_index'].extend([instance_id] * len(juncs))
            ann['junctions'].extend(juncs)
            ann['juncs_tag'].extend(tags)
            ann['edges_positive'].extend(edges.tolist())

            if len(juncs) > 0:
                instance_id += 1
                pid += len(juncs)

        seg_mask = np.clip(seg_mask, 0, 1)

        # handle empty annotations (same fallback as TrainDataset)
        if len(ann['junctions']) == 0:
            ann['mask']           = np.zeros((height, width), dtype=np.float64)
            ann['junctions']      = np.asarray([[0, 0]],    dtype=np.float32)
            ann['bbox']           = np.asarray([[0,0,0,0]], dtype=np.float32)
            ann['juncs_tag']      = np.asarray([0],         dtype=np.int64)
            ann['juncs_index']    = np.asarray([0],         dtype=np.int64)
            ann['edges_positive'] = np.zeros((0, 2),        dtype=np.int64)
        else:
            ann['mask'] = seg_mask
            for key, _type in (['junctions',      np.float32],
                               ['juncs_tag',      np.int64],
                               ['juncs_index',    np.int64],
                               ['edges_positive', np.int64],
                               ['bbox',           np.float32]):
                ann[key] = np.array(ann[key], dtype=_type)

        # no augmentation — reminder=0 signals identity transform
        ann['reminder'] = 0

        if self._transforms is not None:
            return self._transforms(image, ann)
        return image, ann

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        return self.coco.imgs[img_id]

    @staticmethod
    def collate_fn(batch):
        return (default_collate([b[0] for b in batch]),
                [b[1] for b in batch])

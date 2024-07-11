#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import abc
import cv2
import itertools
import numpy as np
from .component import Component
from .datasets import Database


class Segmentation(Component):
    """
    Represent segmentation instances in the composition.
    """
    def __init__(self):
        super().__init__(2)
        self.database = None

    def parse_options(self, params):
        import argparse
        import itertools
        choices = list(itertools.chain.from_iterable([db().get_names() for db in Database.__subclasses__()]))
        choices.append('all')
        parser = argparse.ArgumentParser(prog='Segmentation', add_help=False)
        parser.add_argument('--database', required=True, choices=choices,
                            help='Select database model.')
        args, unknown = parser.parse_known_args(params)
        print(parser.format_usage())
        self.database = args.database
        return unknown

    @abc.abstractmethod
    def train(self, anns_train, anns_valid):
        pass

    @abc.abstractmethod
    def load(self, mode):
        pass

    @abc.abstractmethod
    def process(self, ann, pred):
        pass

    def show(self, viewer, ann, pred):
        datasets = [subclass().get_names() for subclass in Database.__subclasses__()]
        categories = Database.__subclasses__()[next((idx for idx, subset in enumerate(datasets) if self.database in subset), None)]().get_categories()
        names = list([cat.name for cat in categories.values()])
        colors = Database.__subclasses__()[next((idx for idx, subset in enumerate(datasets) if self.database in subset), None)]().get_colors()
        mapping = dict(zip(names, range(len(names))))
        drawing = dict(zip(range(len(names)), colors))
        ann_order = [(img_ann.filename, img_ann.tile) for img_ann in ann.images]  # keep order among 'ann' and 'pred'
        for img_pred in pred.images:
            image_idx = [np.array_equal(img_pred.filename, f) & np.array_equal(img_pred.tile, t) for f, t in ann_order].index(True)
            # mapping['Background'] = 255
            # drawing[255] = (255, 255, 255)
            # import matplotlib.pyplot as plt
            # import matplotlib.colors
            # from .utils import contours2mask
            # fig, axes = plt.subplots(1, 2)
            # ax0, ax1 = axes
            # ax0.get_xaxis().set_ticks([])
            # ax0.get_yaxis().set_ticks([])
            # ax1.get_xaxis().set_ticks([])
            # ax1.get_yaxis().set_ticks([])
            # img = plt.imread(ann.images[0].filename)
            # ann_contours, ann_labels = [], []
            # for obj in ann.images[0].objects:
            #     for contour in obj.multipolygon:
            #         ann_contours.append(contour)
            #     for cat in obj.categories:
            #         ann_labels.append(str(cat.label.name))
            # ann_mask = contours2mask(img.shape[0], img.shape[1], ann_contours, ann_labels, mapping)
            # pred_contours, pred_labels = [], []
            # for obj in pred.images[0].objects:
            #     for contour in obj.multipolygon:
            #         pred_contours.append(contour)
            #     for cat in obj.categories:
            #         pred_labels.append(str(cat.label.name))
            # pred_mask = contours2mask(img.shape[0], img.shape[1], pred_contours, pred_labels, mapping)
            # cmap = matplotlib.colors.ListedColormap(np.array(list(drawing.values()))/255.0)
            # norm = matplotlib.colors.BoundaryNorm(list(range(len(mapping)+1)), cmap.N)
            # ax0.set_title('ann')
            # ax0.imshow(ann_mask, cmap=cmap, norm=norm)
            # ax1.set_title('pred')
            # mappable = ax1.imshow(pred_mask, cmap=cmap, norm=norm)
            # cbar = plt.colorbar(mappable, ax=axes, shrink=0.7, )
            # cbar.ax.get_yaxis().set_ticks([])
            # for idx, val in enumerate(mapping):
            #     cbar.ax.text(len(mapping)+1, (idx+0.45), val, ha='left', va='center', )
            # plt.show()
            for objs_idx, objs_val in enumerate([ann.images[image_idx].objects, img_pred.objects]):
                contours, labels = [], []
                for obj in objs_val:
                    for contour in obj.multipolygon:
                        contours.append(contour)
                        labels.append(str(obj.categories[0].label.name))
                # Draw contours to mask (draw subgroup of contours to handle connected components with holes correctly)
                image = viewer.get_image(img_pred)
                np_contours = np.empty((len(contours),), dtype=object)
                for idx in range(len(np_contours)):
                    np_contours[idx] = contours[idx]
                np_labels = np.array(labels)
                for key, val in mapping.items():
                    if objs_idx == 0:
                        cv2.drawContours(image, np_contours[np_labels == key], -1, drawing[val], thickness=1)
                    else:
                        mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
                        cv2.drawContours(mask, np_contours[np_labels == key], -1, 255, thickness=cv2.FILLED)
                        image[mask == 255] = np.array(0.2*image[mask == 255] + 0.8*np.array(drawing[val]), np.uint8)

    def evaluate(self, fs, ann, pred):
        from .utils import numpy2geometry
        # id_component;filename;num_ann;num_pred[;id_polygon;num_contours[;contour];num_labels[;label]]
        ann_order = [(img_ann.filename, img_ann.tile) for img_ann in ann.images]  # keep order among 'ann' and 'pred'
        for img_pred in pred.images:
            image_idx = [np.array_equal(img_pred.filename, f) & np.array_equal(img_pred.tile, t) for f, t in ann_order].index(True)
            fs.write(str(self.get_component_class()) + ';' + ann.images[image_idx].filename + ';' + str(ann.images[image_idx].tile.tolist()) + ';' + str(len(ann.images[image_idx].objects)) + ';' + str(len(img_pred.objects)))
            for objs_idx, objs_val in enumerate([ann.images[image_idx].objects, img_pred.objects]):
                for obj in objs_val:
                    fs.write(';' + str(obj.id) + ';' + str(len(obj.multipolygon)))
                    for contour in obj.multipolygon:
                        fs.write(';' + str(numpy2geometry(contour)))
                    fs.write(';' + str(len(obj.categories)))
                    for cat in obj.categories:
                        fs.write(';' + str(cat.label.name))
            fs.write('\n')

    def save(self, dirname, pred):
        import os
        import json
        datasets = [subclass().get_names() for subclass in Database.__subclasses__()]
        categories = Database.__subclasses__()[next((idx for idx, subset in enumerate(datasets) if self.database in subset), None)]().get_categories()
        names = list([cat.name for cat in categories.values()])
        for img_pred in pred.images:
            # Create a blank json that matched the labeler provided jsons with default values
            output_json = dict({'images': [], 'annotations': [], 'categories': []})
            output_json['images'].append(dict({'id': 0, 'file_name': os.path.basename(img_pred.filename), 'width': int(img_pred.tile[2]-img_pred.tile[0]), 'height': int(img_pred.tile[3]-img_pred.tile[1]), 'date_captured': img_pred.timestamp}))
            for idx, obj in enumerate(img_pred.objects):
                multipolygon, areas, tls, brs = [], [], [], []
                for polygon in obj.multipolygon:
                    multipolygon.append(list(map(int, list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(polygon)))))))
                    areas.append(cv2.contourArea(polygon))
                    tls.append(polygon.min(0))
                    brs.append(polygon.max(0))
                # Polygon must have 3 or more pair of points
                if len(multipolygon[0]) < 6:
                    continue
                tl = [np.min([pt[:, dim] for pt in tls]) for dim in [0, 1]]
                br = [np.max([pt[:, dim] for pt in brs]) for dim in [0, 1]]
                output_json['annotations'].append(dict({'id': idx, 'image_id': 0, 'category_id': int(names.index(obj.categories[-1].label.name)+1), 'segmentation': multipolygon, 'area': float(np.sum(areas)), 'bbox': list(map(int, [tl[0], tl[1], br[0]-tl[0], br[1]-tl[1]])), 'iscrowd': int(len(img_pred.objects) > 1)}))
            for idx, label in enumerate(names):
                output_json['categories'].append(dict({'id': idx, 'name': label, 'supercategory': ''}))
            # Save COCO annotation file
            root, extension = os.path.splitext(img_pred.filename)
            with open(dirname + os.path.basename(root) + '.json', 'w') as ofs:
                json.dump(output_json, ofs)

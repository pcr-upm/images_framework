#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import abc
import cv2
import itertools
import numpy as np
from .component import Component
from .detection import Detection
from .datasets import Database


class Recognition(Component):
    """
    Represent recognition instances in the composition.
    """
    def __init__(self):
        super().__init__(4)
        self.database = None

    def parse_options(self, params):
        import argparse
        import itertools
        parser = argparse.ArgumentParser(prog='Recognition', add_help=False)
        parser.add_argument('--database', required=True, choices=list(itertools.chain.from_iterable([db().get_names() for db in Database.__subclasses__()])),
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
        datasets = [db.__name__ for db in Database.__subclasses__()]
        ann_order = [img_ann.filename for img_ann in ann.images]  # same order among 'ann' and 'pred' images
        for img_pred in pred.images:
            Detection().show(viewer, ann, pred)
            categories = Database.__subclasses__()[datasets.index(self.database)]().get_categories().values() if self.database else []
            colors = Database.__subclasses__()[datasets.index(self.database)]().get_colors()
            drawing = dict(zip([cat.name for cat in categories], colors))
            image_idx = [np.array_equal(img_pred.filename, elem) for elem in ann_order].index(True)
            for objs_idx, objs_val in enumerate([ann.images[image_idx].objects, img_pred.objects]):
                for obj in objs_val:
                    values = [drawing[cat.label.name] if cat.label in categories else (0, 255, 0) for cat in obj.categories]
                    color = np.mean(values, axis=0)
                    # Draw text
                    (xmin, ymin, xmax, ymax) = obj.bb
                    num_categories = len(obj.categories)
                    for label_idx, label_val in enumerate(obj.categories):
                        text = cv2.getTextSize(label_val.label.name, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
                        pt = (int(xmin+((xmax-xmin)/2.0)-(text[0]/2.0)), int(ymin-(10*num_categories) if objs_idx == 0 else ymax+text[1]+10)+(10*label_idx))
                        viewer.rectangle(img_pred, (pt[0], pt[1]-text[1]), (pt[0]+text[0]-1, pt[1]+1), color)
                        viewer.text(img_pred, label_val.label.name, pt, 0.3, (255, 255, 255) if color[0]*0.299+color[1]*0.587+color[2]*0.114 < 186 else (0, 0, 0))

    def evaluate(self, fs, ann, pred):
        # id_component;filename;num_ann;num_pred[;ann_id[;ann_label]][;pred_id[;pred_label;pred_score]]
        ann_order = [img_ann.filename for img_ann in ann.images]  # same order among 'ann' and 'pred' images
        for img_pred in pred.images:
            image_idx = [np.array_equal(img_pred.filename, elem) for elem in ann_order].index(True)
            ann_categories = list(itertools.chain.from_iterable([obj.categories for obj in ann.images[image_idx].objects]))
            pred_categories = list(itertools.chain.from_iterable([obj.categories for obj in img_pred.objects]))
            fs.write(str(self.get_component_class()) + ';' + ann.images[image_idx].filename + ';' + str(len(ann_categories)) + ';' + str(len(pred_categories)))
            for obj_idx, obj_val in enumerate([ann.images[image_idx].objects, img_pred.objects]):
                for obj in obj_val:
                    fs.write(';' + str(obj.id))
                    for cat in obj.categories:
                        fs.write(';' + cat.label.name) if obj_idx == 0 else fs.write(';' + cat.label.name + ';' + str(cat.score))
            fs.write('\n')

    def save(self, dirname, pred):
        import os
        import json
        datasets = [db.__name__ for db in Database.__subclasses__()]
        categories = Database.__subclasses__()[datasets.index(self.database)]().get_categories()
        for img_pred in pred.images:
            # Create a blank json that matched the labeler provided jsons with default values
            output_json = dict({'images': [], 'annotations': [], 'categories': []})
            output_json['images'].append(dict({'id': 0, 'file_name': os.path.basename(img_pred.filename), 'width': int(img_pred.tile[2]-img_pred.tile[0]), 'height': int(img_pred.tile[3]-img_pred.tile[1]), 'date_captured': img_pred.timestamp}))
            for idx, obj in enumerate(img_pred.objects):
                output_json['annotations'].append(dict({'id': idx+1, 'image_id': 0, 'category_id': int(categories.index(obj.categories[-1])+1), 'bbox': list(map(int, [obj.bb[0], obj.bb[1], obj.bb[2]-obj.bb[0], obj.bb[3]-obj.bb[1]])), 'iscrowd': int(len(img_pred.objects) > 1)}))
            for idx, label in enumerate(categories):
                output_json['categories'].append(dict({'id': idx+1, 'name': label.name, 'supercategory': ''}))
            # Save COCO annotation file
            root, extension = os.path.splitext(img_pred.filename)
            with open(dirname + os.path.basename(root) + '.json', 'w') as ofs:
                json.dump(output_json, ofs)

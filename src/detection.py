#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@geoaitech.com'

import abc
import cv2
import math
import numpy as np
from .component import SatelliteComponent
from .datasets import Database


class SatelliteDetection(SatelliteComponent):
    """
    Represent detection instances in the composition.
    """
    def __init__(self):
        super().__init__(1)
        self.database = None

    def parse_options(self, params):
        import argparse
        parser = argparse.ArgumentParser(prog='SatelliteDetection', add_help=False)
        parser.add_argument('--database', required=True, choices=[db.__name__ for db in Database.__subclasses__()],
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
        ann_order = [satellite_img.filename for satellite_img in ann.images]  # same order among 'ann' and 'pred' images
        for satellite_img in pred.images:
            categories = Database.__subclasses__()[datasets.index(self.database)]().get_categories()
            colors = Database.__subclasses__()[datasets.index(self.database)]().get_colors()
            drawing = dict(zip([cat.name for cat in categories], colors))
            image_idx = [np.array_equal(satellite_img.filename, elem) for elem in ann_order].index(True)
            for obj_idx, obj_val in enumerate([ann.images[image_idx].objects, satellite_img.objects]):
                for satellite_obj in obj_val:
                    values = [drawing[cat.name] if cat in categories else (0, 255, 0) for cat in satellite_obj.categories.get_labels()]
                    color = np.mean(values, axis=0)
                    # Draw rectangle (bb or obb)
                    if satellite_obj.obb != (-1, -1, -1, -1, -1, -1, -1, -1):
                        (x1, y1, x2, y2, x3, y3, x4, y4) = satellite_obj.obb
                        pts = [np.array([[pt] for pt in [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]], dtype=np.int32)]
                        contour = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int32)
                        thickness = int(round(math.log(max(math.e, np.sqrt(cv2.contourArea(contour))), 2)))
                        viewer.polygon(satellite_img, pts, color, -1 if obj_idx == 0 else thickness)
                        viewer.circle(satellite_img, (int(round(x1)), int(round(y1))), thickness+1, color)
                    else:
                        (xmin, ymin, xmax, ymax) = satellite_obj.bb
                        contour = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=np.int32)
                        thickness = int(round(math.log(max(math.e, np.sqrt(cv2.contourArea(contour))), 2)))
                        viewer.rectangle(satellite_img, (int(round(xmin)), int(round(ymin))), (int(round(xmax)), int(round(ymax))), color, -1 if obj_idx == 0 else thickness)

    def evaluate(self, fs, ann, pred):
        # id_component;filename;num_ann;num_pred[;ann_id;ann_bb;ann_obb[;ann_label]][;pred_id;pred_bb;pred_obb[;pred_label;pred_score]]
        ann_order = [satellite_img.filename for satellite_img in ann.images]  # same order among 'ann' and 'pred' images
        for satellite_img in pred.images:
            image_idx = [np.array_equal(satellite_img.filename, elem) for elem in ann_order].index(True)
            fs.write(str(self.get_component_class()) + ';' + ann.images[image_idx].filename + ';' + str(len(ann.images[image_idx].objects)) + ';' + str(len(satellite_img.objects)))
            for obj_idx, obj_val in enumerate([ann.images[image_idx].objects, satellite_img.objects]):
                for satellite_obj in obj_val:
                    fs.write(';' + str(satellite_obj.id) + ';' + str(satellite_obj.bb) + ';' + str(satellite_obj.obb))
                    scores = satellite_obj.categories.get_scores()
                    for label_idx, label_val in enumerate(satellite_obj.categories.get_labels()):
                        fs.write(';' + label_val.name) if obj_idx == 0 else fs.write(';' + label_val.name + ';' + str(scores[label_idx]))
            fs.write('\n')

    def save(self, dirname, pred):
        import os
        from pascal_voc_writer import Writer
        for satellite_img in pred.images:
            img = cv2.imread(satellite_img.filename)
            writer = Writer(satellite_img.filename, img.shape[0], img.shape[1], img.shape[2])
            for satellite_obj in satellite_img.objects:
                (xmin, ymin, xmax, ymax) = satellite_obj.bb
                writer.addObject(satellite_obj.categories.get_labels()[0].name, xmin, ymin, xmax, ymax)
            # Save Pascal VOC annotation file
            root, extension = os.path.splitext(satellite_img.filename)
            writer.save(dirname + os.path.basename(root) + '.xml')

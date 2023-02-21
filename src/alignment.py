#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import abc
import cv2
import math
import numpy as np
from .component import Component
from .datasets import Database


class Alignment(Component):
    """
    Represent alignment instances in the composition.
    """
    def __init__(self):
        super().__init__(3)
        self.database = None

    def parse_options(self, params):
        import argparse
        parser = argparse.ArgumentParser(prog='Alignment', add_help=False)
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

    def euler_to_rotation_matrix(self, headpose):
        # http://euclideanspace.com/maths/geometry/rotations/conversions/eulerToMatrix/index.htm
        # Change coordinates system
        euler = np.array([-(headpose[0]-90), -headpose[1], -(headpose[2]+90)])
        # Convert to radians
        rad = np.deg2rad(euler)
        cy = np.cos(rad[0])
        sy = np.sin(rad[0])
        cp = np.cos(rad[1])
        sp = np.sin(rad[1])
        cr = np.cos(rad[2])
        sr = np.sin(rad[2])
        Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])  # yaw
        Rp = np.array([[cp, -sp, 0.0], [sp, cp, 0.0], [0.0, 0.0, 1.0]])  # pitch
        Rr = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])  # roll
        return np.matmul(np.matmul(Ry, Rp), Rr)

    def show(self, viewer, ann, pred):
        axis = np.eye(3, dtype=float)
        datasets = [db.__name__ for db in Database.__subclasses__()]
        ann_order = [img_ann.filename for img_ann in ann.images]  # same order among 'ann' and 'pred' images
        for img_pred in pred.images:
            categories = Database.__subclasses__()[datasets.index(self.database)]().get_categories()
            colors = Database.__subclasses__()[datasets.index(self.database)]().get_colors()
            drawing = dict(zip([cat.name for cat in categories], colors))
            image_idx = [np.array_equal(img_pred.filename, elem) for elem in ann_order].index(True)
            for objs_idx, objs_val in enumerate([ann.images[image_idx].objects, img_pred.objects]):
                for obj in objs_val:
                    values = [drawing[cat.label.name] if cat.label in categories else (0, 255, 0) for cat in obj.categories]
                    color = np.mean(values, axis=0)
                    # Draw rectangle
                    (xmin, ymin, xmax, ymax) = obj.bb
                    contour = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=np.int32)
                    thickness = int(round(math.log(max(math.e, np.sqrt(cv2.contourArea(contour))), 2)))
                    viewer.rectangle(img_pred, (int(round(xmin)), int(round(ymin))), (int(round(xmax)), int(round(ymax))), color, thickness)
                    # Draw axis
                    mu = cv2.moments(contour)
                    mid = (int(round(mu['m10']/mu['m00'])), int(round(mu['m01']/mu['m00'])))
                    ann_axis = np.matmul(self.euler_to_rotation_matrix(obj.headpose), axis)
                    face_axis = ann_axis * 5.0
                    viewer.line(img_pred, mid, mid+face_axis[:, 0], color, thickness)
                    viewer.line(img_pred, mid, mid+face_axis[:, 1], color, thickness)
                    viewer.line(img_pred, mid, mid+face_axis[:, 2], color, thickness)

    def evaluate(self, fs, ann, pred):
        # id_component;filename;num_ann;num_pred[;ann_id[;ann_label]][;pred_id[;pred_label;pred_score]]
        ann_order = [img_ann.filename for img_ann in ann.images]  # same order among 'ann' and 'pred' images
        for img_pred in pred.images:
            image_idx = [np.array_equal(img_pred.filename, elem) for elem in ann_order].index(True)
            fs.write(str(self.get_component_class()) + ';' + ann.images[image_idx].filename + ';' + ann.headpose + ';' + pred.headpose)
            fs.write('\n')

    def save(self, dirname, pred):
        datasets = [db.__name__ for db in Database.__subclasses__()]
        categories = Database.__subclasses__()[datasets.index(self.database)]().get_categories()
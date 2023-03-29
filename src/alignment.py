#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import abc
import cv2
import math
import numpy as np
from .component import Component
from .detection import Detection
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
        # Convert to radians
        rad = np.deg2rad(headpose)
        cy = np.cos(rad[0])
        sy = np.sin(rad[0])
        cp = np.cos(rad[1])
        sp = np.sin(rad[1])
        cr = np.cos(rad[2])
        sr = np.sin(rad[2])
        Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])  # yaw
        Rp = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])  # pitch
        Rr = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])  # roll
        return np.dot(np.dot(Ry, Rp), Rr)

    def show(self, viewer, ann, pred):
        axis = np.eye(3, dtype=float)
        ann_order = [img_ann.filename for img_ann in ann.images]  # same order among 'ann' and 'pred' images
        for img_pred in pred.images:
            # Detection().show(viewer, ann, pred)
            image_idx = [np.array_equal(img_pred.filename, elem) for elem in ann_order].index(True)
            for objs_idx, objs_val in enumerate([ann.images[image_idx].objects, img_pred.objects]):
                for obj in objs_val:
                    # Draw axis
                    (xmin, ymin, xmax, ymax) = obj.bb
                    contour = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=np.int32)
                    thickness = int(round(math.log(max(math.e, np.sqrt(cv2.contourArea(contour))), 2)))
                    # obj.headpose = (90, 0, 0)  # yaw 90
                    obj_axis = np.dot(axis, self.euler_to_rotation_matrix(-np.array(obj.headpose)))
                    # rvec, _ = cv2.Rodrigues(self.euler_to_rotation_matrix(-np.array(obj.headpose)))
                    # tvec, K, dist = (0, 0, 0), np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float), (0, 0, 0, 0)
                    # obj_axis, _ = cv2.projectPoints(axis, rvec=rvec, tvec=tvec, cameraMatrix=K, distCoeffs=dist)
                    obj_axis *= np.sqrt(cv2.contourArea(contour))
                    mu = cv2.moments(contour)
                    mid = np.array([int(round(mu['m10']/mu['m00'])), int(round(mu['m01']/mu['m00']))])
                    viewer.line(img_pred, mid, mid+obj_axis[0, :2].ravel().astype(int), (0,255,0) if objs_idx == 0 else (0,122,0), thickness)
                    viewer.line(img_pred, mid, mid-obj_axis[1, :2].ravel().astype(int), (0,0,255) if objs_idx == 0 else (0,0,122), thickness)
                    viewer.line(img_pred, mid, mid+obj_axis[2, :2].ravel().astype(int), (255,0,0) if objs_idx == 0 else (122,0,0), thickness)

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
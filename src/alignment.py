#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import abc
import cv2
import math
import numpy as np
from scipy.spatial.transform import Rotation
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
        import itertools
        parser = argparse.ArgumentParser(prog='Alignment', add_help=False)
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
        def pairwise(iterable):
            import itertools
            a, b = itertools.tee(iterable)
            next(b, None)
            return zip(a, b)

        axis = np.eye(3)
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
                    # From right-hand rule to left-hand rule
                    euler = Rotation.from_matrix(obj.headpose).as_euler('YXZ', degrees=True)
                    obj_axis = axis @ Rotation.from_euler('YXZ', [-euler[0], -euler[1], -euler[2]], degrees=True).as_matrix()
                    obj_axis *= np.sqrt(cv2.contourArea(contour))
                    mu = cv2.moments(contour)
                    mid = tuple([int(round(mu['m10']/mu['m00'])), int(round(mu['m01']/mu['m00']))])
                    viewer.line(img_pred, mid, tuple(mid+obj_axis[0, :2].ravel().astype(int)), (0,255,0) if objs_idx == 0 else (0,122,0), thickness)  # green: roll (x-axis)
                    viewer.line(img_pred, mid, tuple(mid+obj_axis[1, :2].ravel().astype(int)), (0,0,255) if objs_idx == 0 else (0,0,122), thickness)  # blue: pitch (y-axis)
                    viewer.line(img_pred, mid, tuple(mid+obj_axis[2, :2].ravel().astype(int)), (255,0,0) if objs_idx == 0 else (122,0,0), thickness)  # red: yaw (z-axis)
                    # Draw landmarks with a black border
                    for lp in obj.landmarks.values():
                        for org, dst in pairwise(lp):
                            color = ((0,122,255) if (org.visible and dst.visible) else (0,0,255)) if objs_idx == 0 else ((0,255,0) if (org.visible and dst.visible) else (255,0,0))
                            viewer.line(img_pred, (int(round(org.pos[0])), int(round(org.pos[1]))), (int(round(dst.pos[0])), int(round(dst.pos[1]))), color, int(round(thickness*0.5)))
                        for lnd in lp:
                            color = ((0,122,255) if lnd.visible else (0,0,255)) if objs_idx == 0 else ((0,255,0) if lnd.visible else (255,0,0))
                            viewer.circle(img_pred, (int(round(lnd.pos[0])), int(round(lnd.pos[1]))), radius=0, color=color, thickness=thickness-1)
                            viewer.circle(img_pred, (int(round(lnd.pos[0])), int(round(lnd.pos[1]))), radius=int(round(thickness*0.5)), color=(0,0,0), thickness=1)

    def evaluate(self, fs, ann, pred):
        # id_component;filename;num_ann;num_pred[;ann_id;ann_bb;ann_pose;num_ann_landmarks[;ann_label;ann_pos;ann_visible;ann_confidence]][;pred_id;pred_bb;pred_pose;num_pred_landmarks[;pred_label;pred_pos;pred_visible;pred_confidence]]
        ann_order = [img_ann.filename for img_ann in ann.images]  # same order among 'ann' and 'pred' images
        for img_pred in pred.images:
            image_idx = [np.array_equal(img_pred.filename, elem) for elem in ann_order].index(True)
            fs.write(str(self.get_component_class()) + ';' + ann.images[image_idx].filename + ';' + str(len(ann.images[image_idx].objects)) + ';' + str(len(img_pred.objects)))
            for objs_idx, objs_val in enumerate([ann.images[image_idx].objects, img_pred.objects]):
                for obj in objs_val:
                    landmarks = map(str, [';'.join(map(str, [lnd.label, ';'.join(map(str, lnd.pos)), lnd.visible, lnd.confidence])) for lnds in obj.landmarks.values() for lnd in lnds])
                    fs.write(';' + str(obj.id) + ';' + str(obj.bb) + ';' + str(obj.headpose.tolist()) + ';' + str(len([lnd for lnds in obj.landmarks.values() for lnd in lnds])) + ';' + ';'.join(landmarks))
            fs.write('\n')

    def save(self, dirname, pred):
        import os
        import json
        datasets = [subclass().get_names() for subclass in Database.__subclasses__()]
        idx = [datasets.index(subset) for subset in datasets if self.database in subset]
        for img_idx, img_pred in enumerate(pred.images):
            # Create a blank json that matched the labeler provided jsons with default values
            output_json = dict({'images': [], 'annotations': [], 'mapping': []})
            output_json['images'].append(dict({'id': img_idx, 'file_name': os.path.basename(img_pred.filename), 'width': int(img_pred.tile[2]-img_pred.tile[0]), 'height': int(img_pred.tile[3]-img_pred.tile[1]), 'date_captured': img_pred.timestamp}))
            for obj in img_pred.objects:
                landmarks = list(map(dict, [dict({'label': int(lnd.label), 'pos': list(map(float, lnd.pos)), 'visible': bool(lnd.visible), 'confidence': float(lnd.confidence)}) for lnds in obj.landmarks.values() for lnd in lnds]))
                output_json['annotations'].append(dict({'id': str(obj.id), 'image_id': img_idx, 'bbox': list(map(float, [obj.bb[0], obj.bb[1], obj.bb[2]-obj.bb[0], obj.bb[3]-obj.bb[1]])), 'pose': list(map(float, Rotation.from_matrix(obj.headpose).as_euler('ZYX', degrees=True))), 'landmarks': landmarks, 'iscrowd': int(len(img_pred.objects) > 1)}))
            mapping = Database.__subclasses__()[idx[0]]().get_mapping()
            output_json['mapping'].append(dict({str(lp.value): list(map(int, mapping[lp])) for lp in mapping}))
            # Save COCO annotation file
            root, extension = os.path.splitext(img_pred.filename)
            with open(dirname + os.path.basename(root) + '.json', 'w') as ofs:
                json.dump(output_json, ofs)

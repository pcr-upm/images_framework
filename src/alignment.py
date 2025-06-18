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
        choices = list(itertools.chain.from_iterable([db().get_names() for db in Database.__subclasses__()]))
        choices.append('all')
        choices.append('all_hpgen')
        parser = argparse.ArgumentParser(prog='Alignment', add_help=False)
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
        from images_framework.alignment.landmarks import lps, PersonLandmarkPart as Pl, FaceLandmarkPart as Pf, HandLandmarkPart as Ph, BodyLandmarkPart as Pb

        def pairwise(iterable):
            import itertools
            a, b = itertools.tee(iterable)
            next(b, None)
            return zip(a, b)

        axis = np.eye(3)
        ann_order = [(img_ann.filename, img_ann.tile) for img_ann in ann.images]  # keep order among 'ann' and 'pred'
        for img_pred in pred.images:
            # Detection().show(viewer, ann, pred)
            image_idx = [np.array_equal(img_pred.filename, f) & np.array_equal(img_pred.tile, t) for f, t in ann_order].index(True)
            for objs_idx, objs_val in enumerate([ann.images[image_idx].objects, img_pred.objects]):
                for obj in objs_val:
                    # Draw axis
                    (xmin, ymin, xmax, ymax) = obj.bb
                    contour = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=np.int32)
                    thickness = int(round(math.log(max(math.e, np.sqrt(cv2.contourArea(contour))), 2)))
                    if (obj.headpose != -1).any():
                        # From right-hand rule to left-hand rule
                        euler = Rotation.from_matrix(obj.headpose).as_euler('YXZ', degrees=True)
                        obj_axis = axis @ Rotation.from_euler('YXZ', [-euler[0], -euler[1], -euler[2]], degrees=True).as_matrix()
                        obj_axis *= np.sqrt(cv2.contourArea(contour))
                        mu = cv2.moments(contour)
                        mid = tuple([int(round(mu['m10']/mu['m00'])), int(round(mu['m01']/mu['m00']))])
                        viewer.line(img_pred, mid, tuple(mid+obj_axis[0, :2].ravel().astype(int)), (0,255,0) if objs_idx == 0 else (0,122,0), thickness)  # green: pitch (x-axis)
                        viewer.line(img_pred, mid, tuple(mid+obj_axis[1, :2].ravel().astype(int)), (0,0,255) if objs_idx == 0 else (0,0,122), thickness)  # blue: yaw (y-axis)
                        viewer.line(img_pred, mid, tuple(mid+obj_axis[2, :2].ravel().astype(int)), (255,0,0) if objs_idx == 0 else (122,0,0), thickness)  # red: roll (z-axis)
                    # Draw skeleton for person objects
                    if any(obj.landmarks[Pl.BODY.value].values()):
                        skeleton = [[Pb.LANKLE, Pb.LKNEE, Pb.LHIP], [Pb.RANKLE, Pb.RKNEE, Pb.RHIP]]  # legs
                        skeleton.extend([[Ph.LWRIST, Pb.LELBOW, Pb.LSHOULDER], [Ph.RWRIST, Pb.RELBOW, Pb.RSHOULDER]])  # arms
                        # skeleton.extend([[Pb.LHIP, Pb.LSHOULDER, Pf.NOSE, Pb.RSHOULDER, Pb.RHIP]])  # kinematic
                        # skeleton.extend([[Pf.LEYE, Pf.REYE], [Pb.LSHOULDER, Pb.RSHOULDER], [Pb.LHIP, Pb.RHIP], [Pb.LHIP, Pb.LSHOULDER, Pf.LEAR], [Pb.RHIP, Pb.RSHOULDER, Pf.REAR]])  # coco
                        for part in skeleton:
                            for part_org, part_dst in pairwise(part):
                                if part_org.value not in obj.landmarks[lps[type(part_org)].value].keys() or part_dst.value not in obj.landmarks[lps[type(part_dst)].value].keys() or obj.landmarks[lps[type(part_org)].value][part_org.value] == [] or obj.landmarks[lps[type(part_dst)].value][part_dst.value] == []:
                                    continue
                                # Connection between parts using the last index of each part
                                org, dst = obj.landmarks[lps[type(part_org)].value][part_org.value][-1], obj.landmarks[lps[type(part_dst)].value][part_dst.value][-1]
                                color = ((0,122,255) if (org.visible and dst.visible) else (0,0,255)) if objs_idx == 0 else ((0,255,0) if (org.visible and dst.visible) else (255,0,0))
                                viewer.line(img_pred, (int(round(org.pos[0])), int(round(org.pos[1]))), (int(round(dst.pos[0])), int(round(dst.pos[1]))), color, int(round(thickness*0.5)))
                    # Draw landmarks with a black border
                    for lnds in [landmarks for lps in obj.landmarks.values() for landmarks in lps.values()]:
                        for org, dst in pairwise(lnds):
                            color = ((0,122,255) if (org.visible and dst.visible) else (0,0,255)) if objs_idx == 0 else ((0,255,0) if (org.visible and dst.visible) else (255,0,0))
                            viewer.line(img_pred, (int(round(org.pos[0])), int(round(org.pos[1]))), (int(round(dst.pos[0])), int(round(dst.pos[1]))), color, int(round(thickness*0.5)))
                        for lnd in lnds:
                            color = ((0,122,255) if lnd.visible else (0,0,255)) if objs_idx == 0 else ((0,255,0) if lnd.visible else (255,0,0))
                            viewer.circle(img_pred, (int(round(lnd.pos[0])), int(round(lnd.pos[1]))), radius=0, color=color, thickness=thickness-1)
                            viewer.circle(img_pred, (int(round(lnd.pos[0])), int(round(lnd.pos[1]))), radius=int(round(thickness*0.5)), color=(0,0,0), thickness=1)

    def evaluate(self, fs, ann, pred):
        # id_component;filename;num_ann;num_pred[;ann_id;ann_bb;ann_pose;num_ann_landmarks[;ann_label;ann_pos;ann_visible;ann_confidence]][;pred_id;pred_bb;pred_pose;num_pred_landmarks[;pred_label;pred_pos;pred_visible;pred_confidence]]
        ann_order = [(img_ann.filename, img_ann.tile) for img_ann in ann.images]  # keep order among 'ann' and 'pred'
        for img_pred in pred.images:
            image_idx = [np.array_equal(img_pred.filename, f) & np.array_equal(img_pred.tile, t) for f, t in ann_order].index(True)
            fs.write(str(self.get_component_class()) + ';' + ann.images[image_idx].filename + ';' + str(ann.images[image_idx].tile.tolist()) + ';' + str(len(ann.images[image_idx].objects)) + ';' + str(len(img_pred.objects)))
            for objs_idx, objs_val in enumerate([ann.images[image_idx].objects, img_pred.objects]):
                for obj in objs_val:
                    landmarks = map(str, [';'.join(map(str, [lnd.label, ';'.join(map(str, lnd.pos)), lnd.visible, lnd.confidence])) for lnds in [landmarks for lps in obj.landmarks.values() for landmarks in lps.values()] for lnd in lnds])
                    num_landmarks = len([lnd for lnds in [landmarks for lps in obj.landmarks.values() for landmarks in lps.values()] for lnd in lnds])
                    fs.write(';' + str(obj.id) + ';' + str(obj.bb) + ';' + str(obj.headpose.tolist() if hasattr(obj, 'headpose') else np.eye(3).tolist()) + ';' + str(num_landmarks))
                    if num_landmarks > 0:
                        fs.write(';' + ';'.join(landmarks))
            fs.write('\n')

    def save(self, dirname, pred):
        import os
        import json
        datasets = [subclass().get_names() for subclass in Database.__subclasses__()]
        parts = Database.__subclasses__()[next((idx for idx, subset in enumerate(datasets) if self.database in subset), None)]().get_landmarks()
        for img_idx, img_pred in enumerate(pred.images):
            # Create a blank json that matched the labeler provided jsons with default values
            output_json = dict({'images': [], 'annotations': [], 'mapping': []})
            output_json['images'].append(dict({'id': img_idx, 'file_name': os.path.basename(img_pred.filename), 'width': int(img_pred.tile[2]-img_pred.tile[0]), 'height': int(img_pred.tile[3]-img_pred.tile[1]), 'date_captured': img_pred.timestamp}))
            for obj in img_pred.objects:
                landmarks = list(map(dict, [dict({'label': int(lnd.label), 'pos': list(map(float, lnd.pos)), 'visible': bool(lnd.visible), 'confidence': float(lnd.confidence)}) for lnds in [landmarks for lps in obj.landmarks.values() for landmarks in lps.values()] for lnd in lnds]))
                output_json['annotations'].append(dict({'id': str(obj.id), 'image_id': img_idx, 'bbox': list(map(float, [obj.bb[0], obj.bb[1], obj.bb[2]-obj.bb[0], obj.bb[3]-obj.bb[1]])), 'pose': list(map(float, Rotation.from_matrix(obj.headpose).as_euler('YXZ', degrees=True) if hasattr(obj, 'headpose') else [-1.0, -1.0, -1.0])), 'landmarks': landmarks, 'iscrowd': int(len(img_pred.objects) > 1)}))
            output_json['mapping'].append(dict({str(lp.value): list(map(int, parts[lp])) for lp in parts}))
            # Save COCO annotation file
            root, extension = os.path.splitext(img_pred.filename)
            with open(dirname + os.path.basename(root) + '.json', 'w') as ofs:
                json.dump(output_json, ofs)

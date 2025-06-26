#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import abc
import numpy as np
from .annotations import GenericGroup, GenericImage, GenericObject, GenericCategory, GenericAttribute, GenericLandmark
from .categories import Name, Category as Oi


def get_palette(n):
    """
    Returns the default color map for visualizing each category.
    """
    palette = [0] * (n*3)
    for j in range(0, n):
        lab = j
        palette[j*3+0] = 0
        palette[j*3+1] = 0
        palette[j*3+2] = 0
        i = 0
        while lab:
            palette[j*3+0] |= (((lab >> 0) & 1) << (7-i))
            palette[j*3+1] |= (((lab >> 1) & 1) << (7-i))
            palette[j*3+2] |= (((lab >> 2) & 1) << (7-i))
            i += 1
            lab >>= 3
    return list(zip(*[iter(palette)]*3))


class Database(abc.ABC):
    """
    Declare a common interface for the different data sets.
    """
    def __init__(self):
        self._names = ''
        self._landmarks = {}
        self._categories = {}
        self._colors = []

    @abc.abstractmethod
    def load_filename(self, path, db, line):
        pass

    def get_names(self):
        return self._names

    def get_landmarks(self):
        return self._landmarks

    def get_categories(self):
        return self._categories

    def get_colors(self):
        return self._colors


class Mnist(Database):
    def __init__(self):
        from images_framework.categories.characters import Character as Oc
        super().__init__()
        self._names = ['mnist']
        self._categories = {0: Oc.CHARACTER.ZERO, 1: Oc.CHARACTER.ONE, 2: Oc.CHARACTER.TWO, 3: Oc.CHARACTER.THREE, 4: Oc.CHARACTER.FOUR, 5: Oc.CHARACTER.FIVE, 6: Oc.CHARACTER.SIX, 7: Oc.CHARACTER.SEVEN, 8: Oc.CHARACTER.EIGHT, 9: Oc.CHARACTER.NINE}
        self._colors = get_palette(len(self._categories))

    def load_filename(self, path, db, line):
        from PIL import Image
        seq = GenericGroup()
        parts = line.strip().split(';')
        image = GenericImage(path + parts[0])
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        obj = GenericObject()
        obj.bb = (0, 0, width, height)
        obj.add_category(GenericCategory(self._categories[int(parts[1])]))
        image.add_object(obj)
        seq.add_image(image)
        return seq


class Fill50K(Database):
    def __init__(self):
        super().__init__()
        self._names = ['fill50k']
        self._categories = {0: Oi.BACKGROUND}
        self._colors = [(0, 255, 0)]

    def load_filename(self, path, db, line):
        import os
        from PIL import Image
        from pathlib import Path
        from .annotations import DiffusionObject
        seq = GenericGroup()
        parts = line.strip().split(';')
        image = GenericImage(path + parts[0])
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        obj = DiffusionObject()
        obj.bb = (0, 0, width, height)
        obj.add_category(GenericCategory(Oi.BACKGROUND))
        obj.control = path + parts[1]
        obj.prompt = parts[2]
        dirname = path + 'prompt/'
        Path(dirname).mkdir(parents=True, exist_ok=True)
        obj.prompt = dirname + os.path.splitext(os.path.basename(image.filename))[0] + '.txt'
        with open(obj.prompt, 'w', encoding='utf-8') as ofs: 
            ofs.write(parts[2])
        image.add_object(obj)
        seq.add_image(image)
        return seq


class HPGEN(Database):
    def __init__(self):
        super().__init__()
        self._names = ['hpgen']
        self._categories = {0: Oi.FACE}
        self._colors = [(0, 255, 0)]

    def load_filename(self, path, db, line):
        from PIL import Image
        from scipy.spatial.transform import Rotation
        from .annotations import PersonObject
        seq = GenericGroup()
        parts = line.strip().split(';')
        image = GenericImage(path + parts[0])
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        obj = PersonObject()
        obj.bb = (float(parts[2]), float(parts[3]), float(parts[2])+float(parts[4]), float(parts[3])+float(parts[5]))
        obj.add_category(GenericCategory(Name(parts[1])))  # Set identity as category to split the validation set
        # obj.headpose = Rotation.from_euler('YXZ', [float(parts[6]), float(parts[7]), float(parts[8])], degrees=True).as_matrix()
        obj.headpose = Rotation.from_euler('YXZ', [-float(parts[6]), float(parts[7]), 0.0], degrees=True).as_matrix()
        image.add_object(obj)
        seq.add_image(image)
        return seq


class COCO(Database):
    def __init__(self):
        from images_framework.alignment.landmarks import FaceLandmarkPart as Pf, HandLandmarkPart as Ph, BodyLandmarkPart as Pb
        from images_framework.categories.vehicles import Vehicle as Ov
        from images_framework.categories.outdoor import Outdoor as Oo
        from images_framework.categories.animals import Animal as Oa
        from images_framework.categories.accessories import Accessory as Oq
        from images_framework.categories.sports import Sport as Os
        from images_framework.categories.kitchen import Kitchen as Ok
        from images_framework.categories.food import Food as Of
        from images_framework.categories.furniture import Furniture as Ow
        from images_framework.categories.electronic import Electronic as Oe
        from images_framework.categories.appliance import Appliance as Oj
        from images_framework.categories.indoor import Indoor as Oy
        super().__init__()
        self._names = ['coco']
        self._landmarks = {Pf.NOSE: (0,), Pf.LEYE: (1,), Pf.REYE: (2,), Pf.LEAR: (3,), Pf.REAR: (4,), Pb.LSHOULDER: (5,), Pb.RSHOULDER: (6,), Pb.LELBOW: (7,), Pb.RELBOW: (8,), Ph.LWRIST: (9,), Ph.RWRIST: (10,), Pb.LHIP: (11,), Pb.RHIP: (12,), Pb.LKNEE: (13,), Pb.RKNEE: (14,), Pb.LANKLE: (15,), Pb.RANKLE: (16,)}
        self._categories = {1: Oi.PERSON, 2: Ov.VEHICLE.BICYCLE, 3: Ov.VEHICLE.CAR, 4: Ov.VEHICLE.MOTORCYCLE, 5: Ov.VEHICLE.AIRPLANE, 6: Ov.VEHICLE.BUS, 7: Ov.VEHICLE.TRAIN, 8: Ov.VEHICLE.TRUCK, 9: Ov.VEHICLE.BOAT, 10: Oo.OUTDOOR.TRAFFIC_LIGHT, 11: Oo.OUTDOOR.FIRE_HYDRANT, 12: Oo.OUTDOOR.STREET_SIGN, 13: Oo.OUTDOOR.STOP_SIGN, 14: Oo.OUTDOOR.PARKING_METER, 15: Oo.OUTDOOR.BENCH, 16: Oa.ANIMAL.BIRD, 17: Oa.ANIMAL.CAT, 18: Oa.ANIMAL.DOG, 19: Oa.ANIMAL.HORSE, 20: Oa.ANIMAL.SHEEP, 21: Oa.ANIMAL.COW, 22: Oa.ANIMAL.ELEPHANT, 23: Oa.ANIMAL.BEAR, 24: Oa.ANIMAL.ZEBRA, 25: Oa.ANIMAL.GIRAFFE, 26: Oq.ACCESSORY.HAT, 27: Oq.ACCESSORY.BACKPACK, 28: Oq.ACCESSORY.UMBRELLA, 29: Oq.ACCESSORY.SHOE, 30: Oq.ACCESSORY.EYE_GLASSES, 31: Oq.ACCESSORY.HANDBAG, 32: Oq.ACCESSORY.TIE, 33: Oq.ACCESSORY.SUITCASE, 34: Os.SPORTS.FRISBEE, 35: Os.SPORTS.SKIS, 36: Os.SPORTS.SNOWBOARD, 37: Os.SPORTS.SPORTS_BALL, 38: Os.SPORTS.KITE, 39: Os.SPORTS.BASEBALL_BAT, 40: Os.SPORTS.BASEBALL_GLOVE, 41: Os.SPORTS.SKATEBOARD, 42: Os.SPORTS.SURFBOARD, 43: Os.SPORTS.TENNIS_RACKET, 44: Ok.KITCHEN.BOTTLE, 45: Ok.KITCHEN.PLATE, 46: Ok.KITCHEN.WINE_GLASS, 47: Ok.KITCHEN.CUP, 48: Ok.KITCHEN.FORK, 49: Ok.KITCHEN.KNIFE, 50: Ok.KITCHEN.SPOON, 51: Ok.KITCHEN.BOWL, 52: Of.FOOD.BANANA, 53: Of.FOOD.APPLE, 54: Of.FOOD.SANDWICH, 55: Of.FOOD.ORANGE, 56: Of.FOOD.BROCCOLI, 57: Of.FOOD.CARROT, 58: Of.FOOD.HOT_DOG, 59: Of.FOOD.PIZZA, 60: Of.FOOD.DONUT, 61: Of.FOOD.CAKE, 62: Ow.FURNITURE.CHAIR, 63: Ow.FURNITURE.COUCH, 64: Ow.FURNITURE.POTTED_PLANT, 65: Ow.FURNITURE.BED, 66: Ow.FURNITURE.MIRROR, 67: Ow.FURNITURE.DINING_TABLE, 68: Ow.FURNITURE.WINDOW, 69: Ow.FURNITURE.DESK, 70: Ow.FURNITURE.TOILET, 71: Ow.FURNITURE.DOOR, 72: Oe.ELECTRONIC.TV, 73: Oe.ELECTRONIC.LAPTOP, 74: Oe.ELECTRONIC.MOUSE, 75: Oe.ELECTRONIC.REMOTE, 76: Oe.ELECTRONIC.KEYBOARD, 77: Oe.ELECTRONIC.CELL_PHONE, 78: Oj.APPLIANCE.MICROWAVE, 79: Oj.APPLIANCE.OVEN, 80: Oj.APPLIANCE.TOASTER, 81: Oj.APPLIANCE.SINK, 82: Oj.APPLIANCE.REFRIGERATOR, 83: Oj.APPLIANCE.BLENDER, 84: Oy.INDOOR.BOOK, 85: Oy.INDOOR.CLOCK, 86: Oy.INDOOR.VASE, 87: Oy.INDOOR.SCISSORS, 88: Oy.INDOOR.TEDDY_BEAR, 89: Oy.INDOOR.HAIR_DRIER, 90: Oy.INDOOR.TOOTHBRUSH, 91: Oy.INDOOR.HAIR_BRUSH}
        self._colors = get_palette(len(self._categories))

    def load_filename(self, path, db, line):
        import json
        import itertools
        from PIL import Image
        # from ast import literal_eval
        from datetime import datetime
        from .annotations import PersonObject
        from images_framework.alignment.landmarks import lps
        seq = GenericGroup()
        parts = line.strip().split(';')
        image = GenericImage(path + parts[0])
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        image.timestamp = datetime.strptime(parts[2], '%Y-%m-%d %H:%M:%S')
        for idx in range(0, int(parts[3])):
            bbox = np.array(json.loads(parts[(5*idx)+5]), dtype=float)
            # contours = literal_eval(parts[(5*idx)+6])
            landmarks = np.array(json.loads(parts[(5*idx)+8]), dtype=int)
            obj = GenericObject() if landmarks.size == 0 else PersonObject()
            obj.id = int(parts[(5*idx)+4])
            obj.bb = (float(bbox[0]), float(bbox[1]), float(bbox[0]+bbox[2]), float(bbox[1]+bbox[3]))
            # obj.multipolygon = [np.array([[[pt[0], pt[1]]] for pt in list(zip(contour[::2], contour[1::2]))], dtype=float) for contour in contours]
            obj.add_category(GenericCategory(list(self._categories.values())[int(parts[(5*idx)+7])-1]))
            if not isinstance(obj, PersonObject):
                continue
            for label in list(itertools.chain.from_iterable(self._landmarks.values())):
                lp = list(self._landmarks.keys())[next((ids for ids, xs in enumerate(self._landmarks.values()) for x in xs if x == label), None)]
                pos = (int(landmarks[(3*label)]), int(landmarks[(3*label)+1]))
                vis = int(landmarks[(3*label)+2])
                if vis == 0:  # landmark is not in the image
                    continue
                obj.add_landmark(GenericLandmark(label, lp, pos, bool(vis == 2)), lps[type(lp)])
            image.add_object(obj)
        seq.add_image(image)
        return seq


class Agora(Database):
    def __init__(self):
        from images_framework.alignment.landmarks import FaceLandmarkPart as Pf, HandLandmarkPart as Ph, BodyLandmarkPart as Pb
        super().__init__()
        self._names = ['agora']
        self._landmarks = {Pf.REYEBROW: (4, 124, 5, 126, 6), Pf.LEYEBROW: (1, 119, 2, 121, 3), Pf.NOSE: (128, 129, 130, 17, 16, 133, 134, 135, 18), Pf.REYE: (11, 144, 145, 12, 147, 148), Pf.LEYE: (7, 138, 139, 8, 141, 142), Pf.TMOUTH: (20, 150, 151, 22, 153, 154, 21, 165, 164, 163, 162, 161), Pf.BMOUTH: (156, 157, 23, 159, 160, 168, 167, 166)}
        # self._landmarks = {Pf.NOSE: (110, 111,), Pf.LEYE: (109,), Pf.REYE: (112,), Pb.LSHOULDER: (5,), Pb.RSHOULDER: (6,), Pb.LELBOW: (7,), Pb.RELBOW: (8,), Ph.LWRIST: (9,), Ph.RWRIST: (10,), Pb.LHIP: (11,), Pb.RHIP: (12,), Pb.LKNEE: (13,), Pb.RKNEE: (14,), Pb.LANKLE: (15,), Pb.RANKLE: (16,), Pb.LTOE: (15, 105,), Pb.RTOE: (16, 106,), Pb.NECK: (17,), Pb.CHEST: (107, 108,), Pb.ABDOMEN: (101, 102, 103, 104,)}
        self._categories = {0: Oi.PERSON}
        self._colors = [(0, 255, 0)]

    def load_filename(self, path, db, line):
        import cv2
        import json
        from PIL import Image
        from scipy.spatial.transform import Rotation
        from .annotations import PersonObject
        from images_framework.alignment.landmarks import lps, PersonLandmarkPart as Pl
        seq = GenericGroup()
        parts = line.strip().split(';')
        image = GenericImage(path + parts[0])
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        for idx_obj in range(0, int(parts[1])):
            obj = PersonObject()
            obj.id = int(parts[(7*idx_obj)+2])
            obj.bb = (int(parts[(7*idx_obj)+3]), int(parts[(7*idx_obj)+4]), int(parts[(7*idx_obj)+3])+int(parts[(7*idx_obj)+5]), int(parts[(7*idx_obj)+4])+int(parts[(7*idx_obj)+6]))
            rot_matrix = np.reshape(np.matrix(parts[(7*idx_obj)+7], dtype=np.float32), (4, 4))
            euler = Rotation.from_matrix(np.transpose(rot_matrix[:3, :3])).as_euler('YXZ', degrees=True)
            obj.headpose = Rotation.from_euler('YXZ', [euler[0], euler[1], -euler[2]], degrees=True).as_matrix()
            obj.add_category(GenericCategory(Oi.PERSON))
            landmarks = np.array(json.loads(parts[(7*idx_obj)+8]), dtype=float).reshape(-1, 2)  # (51, 2)
            indices = [4, 124, 5, 126, 6, 1, 119, 2, 121, 3, 128, 129, 130, 17, 16, 133, 134, 135, 18, 11, 144, 145, 12, 147, 148, 7, 138, 139, 8, 141, 142, 20, 150, 151, 22, 153, 154, 21, 165, 164, 163, 162, 161, 156, 157, 23, 159, 160, 168, 167, 166]
            for idx_lnd, label in enumerate(indices):
                lp = list(self._landmarks.keys())[next((ids for ids, xs in enumerate(self._landmarks.values()) for x in xs if x == label), None)]
                pos = (int(landmarks[idx_lnd][0]), int(landmarks[idx_lnd][1]))
                obj.add_landmark(GenericLandmark(label, lp, pos, True), lps[type(lp)])
            # landmarks = np.array(json.loads(parts[(2*idx_obj)+3]), dtype=float).reshape(-1, 3)  # (127, 3)
            # skeleton, hands, face = landmarks[:25], landmarks[25:56], landmarks[56:]
            # indices = [101, 11, 12, 102, 13, 14, 103, 15, 16, 104, 105, 106, 17, 107, 108, 112, 5, 6, 7, 8, 9, 10, 109, 110, 111]
            # for idx_lnd, label in enumerate(indices):
            #     lp = list(self._landmarks.keys())[next((ids for ids, xs in enumerate(self._landmarks.values()) for x in xs if x == label), None)]
            #     pos = (int(skeleton[idx_lnd][0]), int(skeleton[idx_lnd][1]))
            #     vis = int(skeleton[idx_lnd][2])
            #     obj.add_landmark(GenericLandmark(label, lp, pos, bool(vis)), lps[type(lp)])
            obj.bb = cv2.boundingRect(landmarks[:, :2].astype(int))
            obj.bb = (obj.bb[0], obj.bb[1], obj.bb[0]+obj.bb[2], obj.bb[1]+obj.bb[3])
            image.add_object(obj)
        seq.add_image(image)
        return seq


class PTS68(Database):
    def __init__(self):
        from images_framework.alignment.landmarks import FaceLandmarkPart as Pf
        super().__init__()
        self._names = ['300w_public', '300w_private']
        self._landmarks = {Pf.LEYEBROW: (1, 119, 2, 121, 3), Pf.REYEBROW: (4, 124, 5, 126, 6), Pf.LEYE: (7, 138, 139, 8, 141, 142), Pf.REYE: (11, 144, 145, 12, 147, 148), Pf.NOSE: (128, 129, 130, 17, 16, 133, 134, 135, 18), Pf.TMOUTH: (20, 150, 151, 22, 153, 154, 21, 165, 164, 163, 162, 161), Pf.BMOUTH: (156, 157, 23, 159, 160, 168, 167, 166), Pf.LEAR: (101, 102, 103, 104, 105, 106), Pf.REAR: (112, 113, 114, 115, 116, 117), Pf.CHIN: (107, 108, 24, 110, 111)}
        self._categories = {0: Oi.FACE}
        self._colors = [(0, 255, 0)]

    def load_filename(self, path, db, line):
        import os
        from PIL import Image
        from pathlib import Path
        from .annotations import DiffusionObject
        from images_framework.alignment.landmarks import lps
        seq = GenericGroup()
        parts = line.strip().split(';')
        image = GenericImage(path + parts[0])
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        obj = DiffusionObject()
        obj.bb = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        obj.add_category(GenericCategory(Oi.FACE))
        indices = [101, 102, 103, 104, 105, 106, 107, 108, 24, 110, 111, 112, 113, 114, 115, 116, 117, 1, 119, 2, 121, 3, 4, 124, 5, 126, 6, 128, 129, 130, 17, 16, 133, 134, 135, 18, 7, 138, 139, 8, 141, 142, 11, 144, 145, 12, 147, 148, 20, 150, 151, 22, 153, 154, 21, 156, 157, 23, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168]
        for idx in range(0, len(indices)):
            label = indices[idx]
            lp = list(self._landmarks.keys())[next((ids for ids, xs in enumerate(self._landmarks.values()) for x in xs if x == label), None)]
            pos = (float(parts[(2*idx)+5]), float(parts[(2*idx)+6]))
            obj.add_landmark(GenericLandmark(label, lp, pos, True), lps[type(lp)])
            # dirname = path + 'landmarks/'
            # Path(dirname).mkdir(parents=True, exist_ok=True)
            # obj.control = dirname + os.path.splitext(os.path.basename(image.filename))[0] + '_' + '_'.join(str(int(elem)) for elem in obj.bb) + '.png'
            # dirname = path + 'prompt/'
            # Path(dirname).mkdir(parents=True, exist_ok=True)
            # obj.prompt = dirname + os.path.splitext(os.path.basename(image.filename))[0] + '_' + '_'.join(str(int(elem)) for elem in obj.bb) + '.txt'
        image.add_object(obj)
        seq.add_image(image)
        return seq


class COFW(Database):
    def __init__(self):
        from images_framework.alignment.landmarks import FaceLandmarkPart as Pf
        super().__init__()
        self._names = ['cofw']
        self._landmarks = {Pf.LEYEBROW: (1, 101, 3, 102), Pf.REYEBROW: (4, 103, 6, 104), Pf.LEYE: (7, 9, 8, 10, 105), Pf.REYE: (11, 13, 12, 14, 106), Pf.NOSE: (16, 17, 18, 107), Pf.TMOUTH: (20, 22, 21, 108), Pf.BMOUTH: (109, 23), Pf.CHIN: (24,)}
        self._categories = {0: Oi.FACE}
        self._colors = [(0, 255, 0)]

    def load_filename(self, path, db, line):
        import os
        from PIL import Image
        from pathlib import Path
        from .annotations import DiffusionObject
        from images_framework.alignment.landmarks import lps
        seq = GenericGroup()
        parts = line.strip().split(';')
        image = GenericImage(path + parts[0])
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        obj = DiffusionObject()
        obj.bb = (float(parts[1]), float(parts[2]), float(parts[1])+float(parts[3]), float(parts[2])+float(parts[4]))
        obj.add_category(GenericCategory(Oi.FACE))
        indices = [1, 6, 3, 4, 101, 102, 103, 104, 7, 12, 8, 11, 9, 10, 13, 14, 105, 106, 16, 18, 17, 107, 20, 21, 22, 108, 109, 23, 24]
        for idx in range(0, len(indices)):
            label = indices[idx]
            lp = list(self._landmarks.keys())[next((ids for ids, xs in enumerate(self._landmarks.values()) for x in xs if x == label), None)]
            pos = (float(parts[(3*idx)+5]), float(parts[(3*idx)+6]))
            vis = float(parts[(3*idx)+7]) == 0.0
            obj.add_landmark(GenericLandmark(label, lp, pos, vis), lps[type(lp)])
            # dirname = path + 'landmarks/'
            # Path(dirname).mkdir(parents=True, exist_ok=True)
            # obj.control = dirname + os.path.splitext(os.path.basename(image.filename))[0] + '_' + '_'.join(str(int(elem)) for elem in obj.bb) + '.png'
            # dirname = path + 'prompt/'
            # Path(dirname).mkdir(parents=True, exist_ok=True)
            # obj.prompt = dirname + os.path.splitext(os.path.basename(image.filename))[0] + '_' + '_'.join(str(int(elem)) for elem in obj.bb) + '.txt'
        image.add_object(obj)
        seq.add_image(image)
        return seq


class AFLW(Database):
    def __init__(self):
        from images_framework.alignment.landmarks import FaceLandmarkPart as Pf
        super().__init__()
        self._names = ['aflw', 'AFLW']
        self._landmarks = {Pf.LEYEBROW: (1, 2, 3), Pf.REYEBROW: (4, 5, 6), Pf.LEYE: (7, 101, 8), Pf.REYE: (11, 102, 12), Pf.NOSE: (16, 17, 18), Pf.TMOUTH: (20, 103, 21), Pf.LEAR: (15,), Pf.REAR: (19,), Pf.CHIN: (24,)}
        self._categories = {0: Oi.FACE}
        self._colors = [(0, 255, 0)]

    def load_filename(self, path, db, line):
        import os
        from PIL import Image
        from pathlib import Path
        from scipy.spatial.transform import Rotation
        from .annotations import DiffusionObject
        from images_framework.alignment.landmarks import lps
        seq = GenericGroup()
        parts = line.strip().split(';')
        image = GenericImage(path + parts[0])
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        obj = DiffusionObject()
        obj.bb = (int(parts[1]), int(parts[2]), int(parts[1])+int(parts[3]), int(parts[2])+int(parts[4]))
        obj.add_category(GenericCategory(Oi.FACE))
        obj.headpose = Rotation.from_euler('YXZ', [float(parts[5]), float(parts[6]), float(parts[7])], degrees=True).as_matrix()
        obj.add_attribute(GenericAttribute('gender', 'male' if parts[8] == 'm' else 'female'))
        obj.add_attribute(GenericAttribute('glasses', bool(parts[9])))
        num_landmarks = int(parts[10])
        indices = [1, 2, 3, 4, 5, 6, 7, 101, 8, 11, 102, 12, 15, 16, 17, 18, 19, 20, 103, 21, 24]
        for idx in range(0, num_landmarks):
            label = indices[int(parts[(3*idx)+11])-1]
            lp = list(self._landmarks.keys())[next((ids for ids, xs in enumerate(self._landmarks.values()) for x in xs if x == label), None)]
            pos = (float(parts[(3*idx)+12]), float(parts[(3*idx)+13]))
            obj.add_landmark(GenericLandmark(label, lp, pos, True), lps[type(lp)])
            # dirname = path + 'landmarks/'
            # Path(dirname).mkdir(parents=True, exist_ok=True)
            # obj.control = dirname + os.path.splitext(os.path.basename(image.filename))[0] + '_' + '_'.join(str(int(elem)) for elem in obj.bb) + '.png'
            # dirname = path + 'prompt/'
            # Path(dirname).mkdir(parents=True, exist_ok=True)
            # obj.prompt = dirname + os.path.splitext(os.path.basename(image.filename))[0] + '_' + '_'.join(str(int(elem)) for elem in obj.bb) + '.txt'
        image.add_object(obj)
        seq.add_image(image)
        return seq


class WFLW(Database):
    def __init__(self):
        from images_framework.alignment.landmarks import FaceLandmarkPart as Pf
        super().__init__()
        self._names = ['wflw']
        self._landmarks = {Pf.LEYEBROW: (1, 134, 2, 136, 3, 138, 139, 140, 141), Pf.REYEBROW: (6, 147, 148, 149, 150, 4, 143, 5, 145), Pf.LEYE: (7, 161, 9, 163, 8, 165, 10, 167, 196), Pf.REYE: (11, 169, 13, 171, 12, 173, 14, 175, 197), Pf.NOSE: (151, 152, 153, 17, 16, 156, 157, 158, 18), Pf.TMOUTH: (20, 177, 178, 22, 180, 181, 21, 192, 191, 190, 189, 188), Pf.BMOUTH: (187, 186, 23, 184, 183, 193, 194, 195), Pf.LEAR: (100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110), Pf.REAR: (122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132), Pf.CHIN: (111, 112, 113, 114, 115, 24, 117, 118, 119, 120, 121)}
        self._categories = {0: Oi.FACE}
        self._colors = [(0, 255, 0)]

    def load_filename(self, path, db, line):
        import os
        from PIL import Image
        from pathlib import Path
        from .annotations import DiffusionObject
        from images_framework.alignment.landmarks import lps
        seq = GenericGroup()
        parts = line.strip().split(';')
        image = GenericImage(path + parts.pop(0))
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        num_faces = int(parts.pop(0))
        for idx in range(0, num_faces):
            obj = DiffusionObject()
            x, y, w, h = parts.pop(0), parts.pop(0), parts.pop(0), parts.pop(0)
            obj.bb = (int(x), int(y), int(x)+int(w), int(y)+int(h))
            obj.add_category(GenericCategory(Oi.FACE))
            pose, expression, illumination, makeup, occlusion, blur = parts.pop(0), parts.pop(0), parts.pop(0), parts.pop(0), parts.pop(0), parts.pop(0)
            indices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 24, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 1, 134, 2, 136, 3, 138, 139, 140, 141, 4, 143, 5, 145, 6, 147, 148, 149, 150, 151, 152, 153, 17, 16, 156, 157, 158, 18, 7, 161, 9, 163, 8, 165, 10, 167, 11, 169, 13, 171, 12, 173, 14, 175, 20, 177, 178, 22, 180, 181, 21, 183, 184, 23, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197]
            for idx in range(0, len(indices)):
                label = indices[idx]
                lp = list(self._landmarks.keys())[next((ids for ids, xs in enumerate(self._landmarks.values()) for x in xs if x == label), None)]
                pos = (float(parts.pop(0)), float(parts.pop(0)))
                obj.add_landmark(GenericLandmark(label, lp, pos, True), lps[type(lp)])
                # dirname = path + 'landmarks/'
                # Path(dirname).mkdir(parents=True, exist_ok=True)
                # obj.control = dirname + os.path.splitext(os.path.basename(image.filename))[0] + '_' + '_'.join(str(int(elem)) for elem in obj.bb) + '.png'
                # dirname = path + 'prompt/'
                # Path(dirname).mkdir(parents=True, exist_ok=True)
                # obj.prompt = dirname + os.path.splitext(os.path.basename(image.filename))[0] + '_' + '_'.join(str(int(elem)) for elem in obj.bb) + '.txt'
            image.add_object(obj)
        seq.add_image(image)
        return seq


class CatHeads(Database):
    def __init__(self):
        from images_framework.alignment.landmarks import FaceLandmarkPart as Pf
        super().__init__()
        self._names = ['catheads']
        self._landmarks = {Pf.LEYE: (101,), Pf.REYE: (102,), Pf.TMOUTH: (103,), Pf.LEAR: (104, 105, 106), Pf.REAR: (107, 108, 109)}
        self._categories = {0: Oi.ANIMAL}
        self._colors = [(0, 255, 0)]

    def load_filename(self, path, db, line):
        import cv2
        import itertools
        from PIL import Image
        from .annotations import PersonObject
        from images_framework.alignment.landmarks import lps, PersonLandmarkPart as Pl
        seq = GenericGroup()
        parts = line.strip().split(';')
        image = GenericImage(path + parts.pop(0))
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        num_landmarks = int(parts.pop(0))
        obj = PersonObject()
        obj.add_category(GenericCategory(Oi.ANIMAL))
        indices = [101, 102, 103, 104, 105, 106, 107, 108, 109]
        for idx in range(0, len(indices)):
            label = indices[idx]
            lp = list(self._landmarks.keys())[next((ids for ids, xs in enumerate(self._landmarks.values()) for x in xs if x == label), None)]
            pos = (int(parts.pop(0)), int(parts.pop(0)))
            obj.add_landmark(GenericLandmark(label, lp, pos, True), lps[type(lp)])
        obj.bb = cv2.boundingRect(np.array([[pt.pos for pt in list(itertools.chain.from_iterable(obj.landmarks[Pl.FACE.value].values()))]]).astype(int))
        obj.bb = (obj.bb[0], obj.bb[1], obj.bb[0]+obj.bb[2], obj.bb[1]+obj.bb[3])
        image.add_object(obj)
        seq.add_image(image)
        return seq


class FaceSynthetics(Database):
    def __init__(self):
        from images_framework.alignment.landmarks import FaceLandmarkPart as Pf
        super().__init__()
        self._names = ['face_synthetics', 'FaceSynthetics']
        self._landmarks = {Pf.LEYEBROW: (1, 119, 2, 121, 3), Pf.REYEBROW: (4, 124, 5, 126, 6), Pf.LEYE: (7, 138, 139, 8, 141, 142, 169), Pf.REYE: (11, 144, 145, 12, 147, 148, 170), Pf.NOSE: (128, 129, 130, 17, 16, 133, 134, 135, 18), Pf.TMOUTH: (20, 150, 151, 22, 153, 154, 21, 165, 164, 163, 162, 161), Pf.BMOUTH: (156, 157, 23, 159, 160, 168, 167, 166), Pf.LEAR: (101, 102, 103, 104, 105, 106), Pf.REAR: (112, 113, 114, 115, 116, 117), Pf.CHIN: (107, 108, 24, 110, 111)}
        # self._landmarks = {Pf.LEYEBROW: (1, 119, 2, 121, 3), Pf.REYEBROW: (4, 124, 5, 126, 6), Pf.LEYE: (7, 138, 139, 8, 141, 142), Pf.REYE: (11, 144, 145, 12, 147, 148), Pf.NOSE: (128, 129, 130, 17, 16, 133, 134, 135, 18), Pf.TMOUTH: (20, 150, 151, 22, 153, 154, 21, 165, 164, 163, 162, 161), Pf.BMOUTH: (156, 157, 23, 159, 160, 168, 167, 166), Pf.LEAR: (101, 102, 103, 104, 105, 106), Pf.REAR: (112, 113, 114, 115, 116, 117), Pf.CHIN: (107, 108, 24, 110, 111)}
        # self._categories = {0: Name('BACKGROUND'), 1: Name('SKIN'), 2: Name('NOSE'), 3: Name('RIGHT_EYE'), 4: Name('LEFT_EYE'), 5: Name('RIGHT_BROW'), 6: Name('LEFT_BROW'), 7: Name('RIGHT_EAR'), 8: Name('LEFT_EAR'), 9: Name('MOUTH_INTERIOR'), 10: Name('TOP_LIP'), 11: Name('BOTTOM_LIP'), 12: Name('NECK'), 13: Name('HAIR'), 14: Name('BEARD'), 15: Name('CLOTHING'), 16: Name('GLASSES'), 17: Name('HEADWEAR'), 18: Name('FACEWEAR'), 255: Name('IGNORE')}
        # self._colors = get_palette(len(self._categories))
        self._categories = {0: Oi.FACE}
        self._colors = [(0, 255, 0)]

    def load_filename(self, path, db, line):
        import os
        import cv2
        import itertools
        from PIL import Image
        from pathlib import Path
        from .annotations import DiffusionObject
        from images_framework.alignment.landmarks import lps, PersonLandmarkPart as Pl
        seq = GenericGroup()
        parts = line.strip().split(';')
        image = GenericImage(path + parts[0])
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        # Segmentation
        # from .utils import load_geoimage, mask2contours
        # segmentation, _ = load_geoimage(path + parts[1])
        # for idx in self._categories.keys():
        #     obj = GenericObject()
        #     mask = np.where((segmentation == idx), 255, 0).astype(np.uint8)
        #     obj.multipolygon = mask2contours(mask)
        #     obj.add_category(GenericCategory(self._categories[idx]))
        #     image.add_object(obj)
        # Landmarks
        obj = DiffusionObject()
        obj.add_category(GenericCategory(Oi.FACE))
        indices = [101, 102, 103, 104, 105, 106, 107, 108, 24, 110, 111, 112, 113, 114, 115, 116, 117, 1, 119, 2, 121, 3, 4, 124, 5, 126, 6, 128, 129, 130, 17, 16, 133, 134, 135, 18, 7, 138, 139, 8, 141, 142, 11, 144, 145, 12, 147, 148, 20, 150, 151, 22, 153, 154, 21, 156, 157, 23, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170]
        # indices = [101, 102, 103, 104, 105, 106, 107, 108, 24, 110, 111, 112, 113, 114, 115, 116, 117, 1, 119, 2, 121, 3, 4, 124, 5, 126, 6, 128, 129, 130, 17, 16, 133, 134, 135, 18, 7, 138, 139, 8, 141, 142, 11, 144, 145, 12, 147, 148, 20, 150, 151, 22, 153, 154, 21, 156, 157, 23, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168]
        for idx in range(0, len(indices)):
            label = indices[idx]
            lp = list(self._landmarks.keys())[next((ids for ids, xs in enumerate(self._landmarks.values()) for x in xs if x == label), None)]
            pos = (float(parts[(2*idx)+3]), float(parts[(2*idx)+4]))
            obj.add_landmark(GenericLandmark(label, lp, pos, True), lps[type(lp)])
        obj.bb = cv2.boundingRect(np.array([[pt.pos for pt in list(itertools.chain.from_iterable(obj.landmarks[Pl.FACE.value].values()))]]).astype(int))
        obj.bb = (obj.bb[0], obj.bb[1], obj.bb[0]+obj.bb[2], obj.bb[1]+obj.bb[3])
        # dirname = path + 'landmarks/'
        # Path(dirname).mkdir(parents=True, exist_ok=True)
        # obj.control = dirname + os.path.splitext(os.path.basename(image.filename))[0] + '_' + '_'.join(str(int(elem)) for elem in obj.bb) + '.png'
        # dirname = path + 'prompt/'
        # Path(dirname).mkdir(parents=True, exist_ok=True)
        # obj.prompt = dirname + os.path.splitext(os.path.basename(image.filename))[0] + '_' + '_'.join(str(int(elem)) for elem in obj.bb) + '.txt'
        image.add_object(obj)
        seq.add_image(image)
        return seq


class DAD(Database):
    def __init__(self):
        from images_framework.alignment.landmarks import FaceLandmarkPart as Pf
        super().__init__()
        self._names = ['dad']
        self._landmarks = {Pf.LEYEBROW: (1983, 2189, 3708, 336, 335, 3153, 3705, 2178, 3684, 3741, 3148, 3696, 2585, 2565, 2567, 3764), Pf.REYEBROW: (570, 694, 3865, 17, 16, 2134, 3863, 673, 3851, 3880, 2121, 3859, 1448, 1428, 1430, 3893), Pf.LEYE: (2441, 2446, 2382, 2381, 2383, 2496, 3690, 2493, 2491, 2465, 3619, 3632, 2505, 2273, 2276, 2355, 2295, 2359, 2267, 2271, 2403, 2437), Pf.REYE: (1183, 1194, 1033, 1023, 1034, 1345, 3856, 1342, 1340, 1243, 3827, 3833, 1354, 824, 827, 991, 883, 995, 814, 822, 1096, 1175), Pf.NOSE: (3540, 3704, 3555, 3560, 3561, 3501, 3526, 3563, 2793, 2751, 3092, 3099, 3102, 2205, 2193, 2973, 2868, 2921, 2920, 1676, 1623, 2057, 2064, 2067, 723, 702, 1895, 1757, 1818, 1817, 3515, 3541), Pf.TMOUTH: (2828, 2832, 2833, 2850, 2813, 2811, 2774, 3546, 1657, 1694, 1696, 1735, 1716, 1715, 1711, 1719, 1748, 1740, 1667, 1668, 3533, 2785, 2784, 2855, 2863, 2836), Pf.BMOUTH: (2891, 2890, 2892, 2928, 2937, 3509, 1848, 1826, 1789, 1787, 1788, 1579, 1773, 1774, 1795, 1802, 1865, 3503, 2948, 2905, 2898, 2881, 2880, 2715), Pf.LEAR: (3386, 3381, 1962, 2213, 2259, 2257, 2954, 3171, 2003), Pf.REAR: (3554, 576, 2159, 1872, 798, 802, 731, 567, 3577, 3582), Pf.CHIN: (3390, 3391, 3396, 3400, 3599, 3593, 3588), Pf.FOREHEAD: (3068, 2196, 2091, 3524, 628, 705, 2030)}
        self._categories = {0: Oi.FACE}
        self._colors = [(0, 255, 0)]

    def load_filename(self, path, db, line):
        import os
        import cv2
        import itertools
        from PIL import Image
        from pathlib import Path
        from scipy.spatial.transform import Rotation
        from .annotations import DiffusionObject
        from images_framework.alignment.landmarks import lps, PersonLandmarkPart as Pl
        seq = GenericGroup()
        parts = line.strip().split(';')
        image = GenericImage(path + parts[0])
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        obj = DiffusionObject()
        # obj.bb = (int(round(float(parts[1]))), int(round(float(parts[2]))), int(round(float(parts[1])))+int(round(float(parts[3]))), int(round(float(parts[2])))+int(round(float(parts[4]))))
        obj.add_category(GenericCategory(Oi.FACE))
        if len(parts) != 12:  # train, val
            flame_vertices3d = np.reshape(np.matrix(parts[5], dtype=np.float32), (5023, 3))
            model_view_matrix = np.reshape(np.matrix(parts[6], dtype=np.float32), (4, 4))
            projection_matrix = np.reshape(np.matrix(parts[7], dtype=np.float32), (4, 4))
            euler = Rotation.from_matrix(model_view_matrix[:3, :3]).as_euler('YXZ', degrees=True)
            obj.headpose = Rotation.from_euler('YXZ', [euler[0], -euler[1], -euler[2]], degrees=True).as_matrix()
            flame_vertices3d_homo = np.concatenate((flame_vertices3d, np.ones_like(flame_vertices3d[:, [0]])), -1)
            flame_vertices3d_world_homo = np.transpose(np.matmul(model_view_matrix, np.transpose(flame_vertices3d_homo)))
            flame_vertices2d_homo = np.transpose(np.matmul(projection_matrix, np.transpose(flame_vertices3d_world_homo)))
            flame_vertices2d = flame_vertices2d_homo[:, :2] / flame_vertices2d_homo[:, [3]]
            for idx in list(itertools.chain.from_iterable(self._landmarks.values())):
                lp = list(self._landmarks.keys())[next((ids for ids, xs in enumerate(self._landmarks.values()) for x in xs if x == idx), None)]
                pos = (float(flame_vertices2d[idx, 0]), height-float(flame_vertices2d[idx, 1]))
                obj.add_landmark(GenericLandmark(idx, lp, pos, True), lps[type(lp)])
        if len(parts) != 8:  # train, test
            obj.add_attribute(GenericAttribute('quality', parts[len(parts)-7]))
            obj.add_attribute(GenericAttribute('gender',  parts[len(parts)-6]))
            obj.add_attribute(GenericAttribute('expression', bool(parts[len(parts)-5] == 'True')))
            obj.add_attribute(GenericAttribute('age', parts[len(parts)-4]))
            obj.add_attribute(GenericAttribute('occlusions', bool(parts[len(parts)-3] == 'True')))
            obj.add_attribute(GenericAttribute('pose', parts[len(parts)-2]))
            obj.add_attribute(GenericAttribute('standard_light', bool(parts[len(parts)-1] == 'True')))
        obj.bb = cv2.boundingRect(np.array([[pt.pos for pt in list(itertools.chain.from_iterable(obj.landmarks[Pl.FACE.value].values()))]]).astype(int))
        obj.bb = (obj.bb[0], obj.bb[1], obj.bb[0]+obj.bb[2], obj.bb[1]+obj.bb[3])
        # dirname = path + 'landmarks/'
        # Path(dirname).mkdir(parents=True, exist_ok=True)
        # obj.control = dirname + os.path.splitext(os.path.basename(image.filename))[0] + '_' + '_'.join(str(int(elem)) for elem in obj.bb) + '.png'
        # dirname = path + 'prompt/'
        # Path(dirname).mkdir(parents=True, exist_ok=True)
        # obj.prompt = dirname + os.path.splitext(os.path.basename(image.filename))[0] + '_' + '_'.join(str(int(elem)) for elem in obj.bb) + '.txt'
        image.add_object(obj)
        seq.add_image(image)
        return seq


class AFLW2000(Database):
    def __init__(self):
        from images_framework.alignment.landmarks import FaceLandmarkPart as Pf
        super().__init__()
        self._names = ['300wlp', 'aflw2000']
        self._landmarks = {Pf.LEYEBROW: (1, 119, 2, 121, 3), Pf.REYEBROW: (4, 124, 5, 126, 6), Pf.LEYE: (7, 138, 139, 8, 141, 142), Pf.REYE: (11, 144, 145, 12, 147, 148), Pf.NOSE: (128, 129, 130, 17, 16, 133, 134, 135, 18), Pf.TMOUTH: (20, 150, 151, 22, 153, 154, 21, 165, 164, 163, 162, 161), Pf.BMOUTH: (156, 157, 23, 159, 160, 168, 167, 166), Pf.LEAR: (101, 102, 103, 104, 105, 106), Pf.REAR: (112, 113, 114, 115, 116, 117), Pf.CHIN: (107, 108, 24, 110, 111)}
        self._categories = {0: Oi.FACE}
        self._colors = [(0, 255, 0)]

    def load_filename(self, path, db, line):
        import os
        import cv2
        import itertools
        from PIL import Image
        from pathlib import Path
        from scipy.spatial.transform import Rotation
        from .annotations import DiffusionObject
        from images_framework.alignment.landmarks import lps, PersonLandmarkPart as Pl
        seq = GenericGroup()
        parts = line.strip().split(';')
        image = GenericImage(path + parts[0])
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        obj = DiffusionObject()
        obj.add_category(GenericCategory(Name(parts[1])))  # Set identity as category to split the validation set
        euler = [float(parts[3]), float(parts[2]), float(parts[4])]
        obj.headpose = Rotation.from_euler('XYZ', euler, degrees=True).as_matrix()
        # Skip images with angles outside the range (-99, 99)
        # if np.any(np.abs(euler) > 99):
        #     return seq
        indices = [101, 102, 103, 104, 105, 106, 107, 108, 24, 110, 111, 112, 113, 114, 115, 116, 117, 1, 119, 2, 121, 3, 4, 124, 5, 126, 6, 128, 129, 130, 17, 16, 133, 134, 135, 18, 7, 138, 139, 8, 141, 142, 11, 144, 145, 12, 147, 148, 20, 150, 151, 22, 153, 154, 21, 156, 157, 23, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168]
        for idx in range(0, len(indices)):
            label = indices[idx]
            lp = list(self._landmarks.keys())[next((ids for ids, xs in enumerate(self._landmarks.values()) for x in xs if x == label), None)]
            pos = (int(round(float(parts[(2*idx)+5]))), int(round(float(parts[(2*idx)+6]))))
            obj.add_landmark(GenericLandmark(label, lp, pos, True), lps[type(lp)])
        obj.bb = cv2.boundingRect(np.array([[pt.pos for pt in list(itertools.chain.from_iterable(obj.landmarks[Pl.FACE.value].values()))]]).astype(int))
        obj.bb = (obj.bb[0], obj.bb[1], obj.bb[0]+obj.bb[2], obj.bb[1]+obj.bb[3])
        # dirname = path + 'landmarks/'
        # Path(dirname).mkdir(parents=True, exist_ok=True)
        # obj.control = dirname + os.path.splitext(os.path.basename(image.filename))[0] + '_' + '_'.join(str(int(elem)) for elem in obj.bb) + '.png'
        # dirname = path + 'prompt/'
        # Path(dirname).mkdir(parents=True, exist_ok=True)
        # obj.prompt = dirname + os.path.splitext(os.path.basename(image.filename))[0] + '_' + '_'.join(str(int(elem)) for elem in obj.bb) + '.txt'
        image.add_object(obj)
        seq.add_image(image)
        return seq


class Pointing04(Database):
    def __init__(self):
        super().__init__()
        self._names = ['pointing04']
        self._categories = {0: Oi.FACE}
        self._colors = [(0, 255, 0)]

    def load_filename(self, path, db, line):
        from PIL import Image
        from scipy.spatial.transform import Rotation
        from .annotations import PersonObject
        seq = GenericGroup()
        parts = line.strip().split(';')
        image = GenericImage(path + parts[0])
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        obj = PersonObject()
        obj.add_category(GenericCategory(Name(parts[1])))  # Set identity as category to split the validation set
        obj.bb = (int(parts[2]), int(parts[3]), int(parts[2])+int(parts[4]), int(parts[3])+int(parts[5]))
        obj.headpose = Rotation.from_euler('YXZ', [-float(parts[6]), float(parts[7]), 0.0], degrees=True).as_matrix()
        image.add_object(obj)
        seq.add_image(image)
        return seq


class Biwi(Database):
    def __init__(self):
        super().__init__()
        self._names = ['biwi']
        self._categories = {0: Oi.FACE}
        self._colors = [(0, 255, 0)]

    def load_filename(self, path, db, line):
        from PIL import Image
        from scipy.spatial.transform import Rotation
        from .annotations import PersonObject
        seq = GenericGroup()
        parts = line.strip().split(';')
        image = GenericImage(path + parts[0])
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        obj = PersonObject()
        obj.bb = (int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
        obj.add_category(GenericCategory(Oi.FACE))
        obj.headpose = Rotation.from_euler('XYZ', [float(parts[6]), float(parts[5]), float(parts[7])], degrees=True).as_matrix()  # biwi_ann_mtcnn.txt
        # obj.headpose = Rotation.from_euler('XYZ', [-float(parts[6]), float(parts[5]), float(parts[7])], degrees=True).as_matrix().transpose() * np.array([[1, -1, 1], [-1, 1, -1], [1, -1, 1]])  # biwi_ann_test.txt
        image.add_object(obj)
        seq.add_image(image)
        return seq


class Panoptic(Database):
    def __init__(self):
        super().__init__()
        self._names = ['panoptic']
        self._categories = {0: Oi.FACE}
        self._colors = [(0, 255, 0)]

    def load_filename(self, path, db, line):
        from PIL import Image
        from scipy.spatial.transform import Rotation
        from .annotations import PersonObject
        seq = GenericGroup()
        parts = line.strip().split(';')
        image = GenericImage(path + parts[0])
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        obj = PersonObject()
        obj.bb = (int(parts[1]), int(parts[2]), int(parts[1])+int(parts[3]), int(parts[2])+int(parts[4]))
        obj.add_category(GenericCategory(Oi.FACE))
        obj.headpose = Rotation.from_euler('XYZ', [float(parts[6]), float(parts[5]), float(parts[7])], degrees=True).as_matrix()
        image.add_object(obj)
        seq.add_image(image)
        return seq


class WIDER(Database):
    def __init__(self):
        super().__init__()
        self._names = ['wider']
        self._categories = {0: Oi.FACE}
        self._colors = [(0, 255, 0)]

    def load_filename(self, path, db, line):
        import json
        from PIL import Image
        from .annotations import PersonObject
        seq = GenericGroup()
        parts = line.strip().split(';')
        image = GenericImage(path + parts[0])
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        num_faces = int(parts[2])
        for idx in range(0, num_faces):
            obj = PersonObject()
            bbox = np.array(json.loads(parts[(3+idx)]), dtype=float)
            obj.bb = (int(round(float(bbox[0]))), int(round(float(bbox[1]))), int(round(float(bbox[0]+bbox[2]))), int(round(float(bbox[1]+bbox[3]))))
            obj.add_category(GenericCategory(self._categories[0]))
            image.add_object(obj)
        seq.add_image(image)
        return seq


class ArckPadel(Database):
    def __init__(self):
        super().__init__()
        self._names = ['arckpadel']
        self._categories = {0: Oi.BALL, 1: Oi.RACKET, 2: Oi.PERSON}
        self._colors = [(0, 255, 0), (0, 0, 255), (0, 255, 255)]

    def load_filename(self, path, db, line):
        import os
        from PIL import Image
        seq = GenericGroup()
        parts = line.strip().split(';')
        num_images = int(parts[1])
        if len(parts) == 2:
            return seq
        for idx in range(0, num_images):
            ann_file = path + parts[2+idx]
            root, extension = os.path.splitext(ann_file)
            image = GenericImage(path + root + '.png')
            width, height = Image.open(image.filename).size
            image.tile = np.array([0, 0, width, height])
            if os.path.exists(ann_file):
                import xml.etree.ElementTree as ET
                tree = ET.parse(ann_file)
                root = tree.getroot()
                for obj in root.findall('object'):
                    bbox = obj.find('bndbox')
                    obj = GenericObject()
                    obj.bb = (int(bbox.find('xmin').text), int(bbox.find('ymin').text), int(bbox.find('xmax').text), int(bbox.find('ymax').text))
                    obj.add_category(GenericCategory(self._categories[0]))
                    image.add_object(obj)
            seq.add_image(image)
        return seq


class XView(Database):
    def __init__(self):
        from images_framework.categories.vehicles import Vehicle as Ov
        from images_framework.categories.buildings import Building as Ob
        super().__init__()
        self._names = ['xview', 'XView']
        self._categories = {13: Oi.VEHICLE.FIXED_WING_AIRCRAFT.CARGO_PLANE, 15: Oi.VEHICLE.HELICOPTER, 18: Oi.VEHICLE.PASSENGER_VEHICLE.SMALL_CAR, 19: Oi.VEHICLE.PASSENGER_VEHICLE.BUS, 20: Oi.VEHICLE.TRUCK, 41: Oi.VEHICLE.MARITIME_VESSEL.MOTORBOAT, 47: Oi.VEHICLE.MARITIME_VESSEL.FISHING_VESSEL, 60: Oi.VEHICLE.ENGINEERING_VEHICLE.DUMP_TRUCK, 64: Oi.VEHICLE.ENGINEERING_VEHICLE.EXCAVATOR, 71: Oi.BUILDING, 86: Oi.STORAGE_TANK, 89: Oi.SHIPPING_CONTAINER}
        # self._categories = {11: Ov.VEHICLE.FIXED_WING_AIRCRAFT, 12: Ov.VEHICLE.FIXED_WING_AIRCRAFT.SMALL_AIRCRAFT, 13: Ov.VEHICLE.FIXED_WING_AIRCRAFT.CARGO_PLANE, 15: Ov.VEHICLE.HELICOPTER, 17: Ov.VEHICLE.PASSENGER_VEHICLE, 18: Ov.VEHICLE.PASSENGER_VEHICLE.SMALL_CAR, 19: Ov.VEHICLE.PASSENGER_VEHICLE.BUS, 20: Ov.VEHICLE.TRUCK.PICKUP_TRUCK, 21: Ov.VEHICLE.TRUCK.UTILITY_TRUCK, 23: Ov.VEHICLE.TRUCK, 24: Ov.VEHICLE.TRUCK.CARGO_TRUCK, 25: Ov.VEHICLE.TRUCK.TRUCK_BOX, 26: Ov.VEHICLE.TRUCK.TRUCK_TRACTOR, 27: Ov.VEHICLE.TRUCK.TRAILER, 28: Ov.VEHICLE.TRUCK.TRUCK_FLATBED, 29: Ov.VEHICLE.TRUCK.TRUCK_LIQUID, 32: Ov.VEHICLE.ENGINEERING_VEHICLE.CRANE_TRUCK, 33: Ov.VEHICLE.RAILWAY_VEHICLE, 34: Ov.VEHICLE.RAILWAY_VEHICLE.PASSENGER_CAR, 35: Ov.VEHICLE.RAILWAY_VEHICLE.CARGO_CAR, 36: Ov.VEHICLE.RAILWAY_VEHICLE.FLAT_CAR, 37: Ov.VEHICLE.RAILWAY_VEHICLE.TANK_CAR, 38: Ov.VEHICLE.RAILWAY_VEHICLE.LOCOMOTIVE, 40: Ov.VEHICLE.MARITIME_VESSEL, 41: Ov.VEHICLE.MARITIME_VESSEL.MOTORBOAT, 42: Ov.VEHICLE.MARITIME_VESSEL.SAILBOAT, 44: Ov.VEHICLE.MARITIME_VESSEL.TUGBOAT, 45: Ov.VEHICLE.MARITIME_VESSEL.BARGE, 47: Ov.VEHICLE.MARITIME_VESSEL.FISHING_VESSEL, 49: Ov.VEHICLE.MARITIME_VESSEL.FERRY, 50: Ov.VEHICLE.MARITIME_VESSEL.YATCH, 51: Ov.VEHICLE.MARITIME_VESSEL.CONTAINER_SHIP, 52: Ov.VEHICLE.MARITIME_VESSEL.OIL_TANKER, 53: Ov.VEHICLE.ENGINEERING_VEHICLE, 54: Ov.VEHICLE.ENGINEERING_VEHICLE.TOWER_CRANE, 55: Ov.VEHICLE.ENGINEERING_VEHICLE.CONTAINER_CRANE, 56: Ov.VEHICLE.ENGINEERING_VEHICLE.REACH_STACKER, 57: Ov.VEHICLE.ENGINEERING_VEHICLE.STRADDLE_CARRIER, 59: Ov.VEHICLE.ENGINEERING_VEHICLE.MOBILE_CRANE, 60: Ov.VEHICLE.ENGINEERING_VEHICLE.DUMP_TRUCK, 61: Ov.VEHICLE.ENGINEERING_VEHICLE.HAUL_TRUCK, 62: Ov.VEHICLE.ENGINEERING_VEHICLE.SCRAPER_TRACTOR, 63: Ov.VEHICLE.ENGINEERING_VEHICLE.FRONT_LOADER, 64: Ov.VEHICLE.ENGINEERING_VEHICLE.EXCAVATOR, 65: Ov.VEHICLE.ENGINEERING_VEHICLE.CEMENT_MIXER, 66: Ov.VEHICLE.ENGINEERING_VEHICLE.GROUND_GRADER, 71: Ob.BUILDING.HUT_TENT, 72: Ob.BUILDING.SHED, 73: Oi.BUILDING, 74: Ob.BUILDING.AIRCRAFT_HANGAR, 76: Ob.BUILDING.DAMAGED_BUILDING, 77: Ob.BUILDING.FACILITY, 79: Oi.CONSTRUCTION_SITE, 83: Ov.VEHICLE.VEHICLE_LOT, 84: Oi.HELIPAD, 86: Oi.STORAGE_TANK, 89: Oi.SHIPPING_CONTAINER_LOT, 91: Oi.SHIPPING_CONTAINER, 93: Oi.PYLON, 94: Oi.TOWER_STRUCTURE}
        self._colors = get_palette(len(self._categories))

    def load_filename(self, path, db, line):
        from PIL import Image
        from .annotations import AerialImage
        seq = GenericGroup()
        parts = line.strip().split(';')
        image = AerialImage(path + parts[0])
        num_predictions = int(parts[1])
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        image.gsd = 0.3
        for idx in range(0, num_predictions):
            obj = GenericObject()
            obj.id = int(parts[(3*idx)+2])
            pts = parts[(3*idx)+3].split(',')
            obj.bb = (int(pts[0]), int(pts[1]), int(pts[2]), int(pts[3]))
            cat = int(parts[(3*idx)+4])
            if cat not in self._categories.keys():
                continue
            obj.add_category(GenericCategory(self._categories[cat]))
            # if self._categories[cat] is not Ov.VEHICLE.PASSENGER_VEHICLE.SMALL_CAR:
            #     continue
            image.add_object(obj)
        if len(image.objects) > 0:
            seq.add_image(image)
        return seq


class XView2(Database):
    def __init__(self):
        from images_framework.categories.buildings import Building as Ob
        super().__init__()
        self._names = ['xview2', 'XView2']
        self._categories = {'building': Oi.BUILDING, 'un-classified': Ob.BUILDING.UNCLASSIFIED, 'no-damage': Ob.BUILDING.NO_DAMAGE, 'minor-damage': Ob.BUILDING.MINOR_DAMAGE, 'major-damage': Ob.BUILDING.MAJOR_DAMAGE, 'destroyed': Ob.BUILDING.DESTROYED}
        self._colors = get_palette(len(self._categories))

    def load_filename(self, path, db, line):
        import json
        from shapely import wkt
        from .utils import geometry2numpy
        from .annotations import AerialImage
        seq = GenericGroup()
        for time in ['pre_', 'post_']:
            filepath = line.strip()
            pos = filepath.find('pre_')
            image = AerialImage(path + filepath[:pos] + time + filepath[pos+4:])
            pos = filepath.find('/') + 1
            mid = filepath[:pos]
            end = filepath[pos:]
            json_file = path + mid + 'labels' + end[6:-16] + time + 'disaster.json'
            with open(json_file) as ifs:
                json_data = json.load(ifs)
            ifs.close()
            image.tile = np.array([0, 0, json_data['metadata']['width'], json_data['metadata']['height']])
            image.gsd = json_data['metadata']['gsd']
            image.nadir_angle = json_data['metadata']['off_nadir_angle']
            image.timestamp = json_data['metadata']['capture_date']
            for feat in json_data['features']['xy']:
                geom = wkt.loads(feat['wkt'])
                if geom.is_empty:
                    continue
                obj = GenericObject()
                obj.id = feat['properties']['uid']
                obj.bb = (int(geom.bounds[0]), int(geom.bounds[1]), int(geom.bounds[2]), int(geom.bounds[3]))
                obj.multipolygon = [contour for contour in geometry2numpy(geom)]
                if json_file.find('post_disaster') > -1:
                    obj.add_category(GenericCategory(self._categories[feat['properties']['subtype']]))
                else:
                    obj.add_category(GenericCategory(self._categories[feat['properties']['feature_type']]))
                image.add_object(obj)
            seq.add_image(image)
        return seq


class DOTA(Database):
    def __init__(self):
        from images_framework.categories.vehicles import Vehicle as Ov
        super().__init__()
        self._names = ['dota_1.0', 'dota_1.5', 'dota_2.0', 'dota']
        self._categories = {'ship': Ov.VEHICLE.SHIP, 'storage-tank': Oi.STORAGE_TANK, 'baseball-diamond': Oi.BASEBALL_DIAMOND, 'tennis-court': Oi.TENNIS_COURT, 'basketball-court': Oi.BASKETBALL_COURT, 'ground-track-field': Oi.GROUND_TRACK_FIELD, 'bridge': Oi.BRIDGE, 'large-vehicle': Ov.VEHICLE.LARGE_VEHICLE, 'small-vehicle': Ov.VEHICLE.SMALL_VEHICLE, 'helicopter': Ov.VEHICLE.HELICOPTER, 'swimming-pool': Oi.SWIMMING_POOL, 'roundabout': Oi.ROUNDABOUT, 'soccer-ball-field': Oi.SOCCER_BALL_FIELD, 'plane': Ov.VEHICLE.PLANE, 'harbor': Oi.HARBOR, 'container-crane': Ov.VEHICLE.ENGINEERING_VEHICLE.CONTAINER_CRANE, 'airport': Oi.AIRPORT, 'helipad': Oi.HELIPAD}
        self._colors = get_palette(len(self._categories))

    def load_filename(self, path, db, line):
        import rasterio
        from .annotations import AerialImage
        seq = GenericGroup()
        filepath = line.strip()
        image = AerialImage(path + filepath)
        pos = filepath.find('/')
        mid = filepath[:pos]
        end = filepath[pos:]
        if mid == 'train' or mid == 'val':
            if db in 'dota_1.0':
                obb = '/labelTxt-v1.0/labelTxt'
                hbb = '/labelTxt-v1.0/' + mid.capitalize() + '_Task2_gt/' + mid +'set_reclabelTxt'
            elif db in 'dota_1.5':
                obb = '/labelTxt-v1.5/DOTA-v1.5_' + mid
                hbb = '/labelTxt-v1.5/DOTA-v1.5_' + mid + '_hbb'
            else:
                obb = '/labelTxt-v2.0/DOTA-v2.0_' + mid
                hbb = '/labelTxt-v2.0/DOTA-v2.0_' + mid + '_hbb'
            with open(path + mid + obb + end[7:-3] + 'txt') as ifs:
                lines_obb = ifs.readlines()
            ifs.close()
            with open(path + mid + hbb + end[7:-3] + 'txt') as ifs:
                lines_hbb = ifs.readlines()
            ifs.close()
            if db in 'dota_1.0':
                gsd = lines_obb[1].strip()
                lines_obb = lines_obb[2:]
            elif db in 'dota_1.5':
                gsd = lines_obb[1].strip()
                lines_obb, lines_hbb = lines_obb[2:], lines_hbb[2:]
            else:
                meta = '/labelTxt-v2.0/' + mid + '_meta/meta'
                with open(path + mid + meta + end[7:-3] + 'txt') as ifs:
                    lines_meta = ifs.readlines()
                ifs.close()
                gsd = lines_meta[2].strip()
        else:
            lines_obb, lines_hbb = [], []
            meta = '/test-dev_meta/meta'
            with open(path + mid + meta + end[7:-3] + 'txt') as ifs:
                lines_meta = ifs.readlines()
            ifs.close()
            gsd = lines_meta[2].strip()
        src_raster = rasterio.open(image.filename, 'r')
        width = src_raster.width
        height = src_raster.height
        image.tile = np.array([0, 0, width, height])
        image.gsd = -1 if gsd == 'gsd:null' or gsd == 'gsd:None' else float(gsd[gsd.find(':')+1:])
        for idx in range(len(lines_obb)):
            elem = lines_obb[idx].strip().split(' ')
            elem_hbb = lines_hbb[idx].strip().split(' ')
            obj = GenericObject()
            obj.bb = (float(elem_hbb[0]), float(elem_hbb[1]), float(elem_hbb[4]), float(elem_hbb[5]))
            obj.obb = (float(elem[0]), float(elem[1]), float(elem[2]), float(elem[3]), float(elem[4]), float(elem[5]), float(elem[6]), float(elem[7]))
            obj.add_category(GenericCategory(self._categories[elem[8]]))
            obj.confidence = 1 - int(elem[9])  # 0 represents a difficult object
            image.add_object(obj)
        seq.add_image(image)
        return seq


class COWC(Database):
    def __init__(self):
        from images_framework.categories.vehicles import Vehicle as Ov
        super().__init__()
        self._names = ['cowc']
        self._categories = {0: Ov.VEHICLE.CAR}
        self._colors = [(0, 255, 0)]

    def load_filename(self, path, db, line):
        import rasterio
        from .annotations import AerialImage
        seq = GenericGroup()
        parts = line.strip().split(',')
        image = AerialImage(path + parts[0])
        num_vehicles = int(parts[1])
        src_raster = rasterio.open(image.filename, 'r')
        width = src_raster.width
        height = src_raster.height
        image.tile = np.array([0, 0, width, height])
        image.gsd = 0.15
        for idx in range(0, num_vehicles):
            obj = GenericObject()
            center = [int(parts[(3*idx)+2]), int(parts[(3*idx)+3])]
            # Bounding boxes were fixed at size 48 pixels which is the maximum length of a car
            obj.bb = (center[0]-12, center[1]-12, center[0]+12, center[1]+12)
            obj.obb = (obj.bb[0], obj.bb[1], obj.bb[2], obj.bb[1], obj.bb[2], obj.bb[3], obj.bb[0], obj.bb[3])
            obj.add_category(GenericCategory(self._categories[0]))
            image.add_object(obj)
        seq.add_image(image)
        return seq


class CARPK(Database):
    def __init__(self):
        from images_framework.categories.vehicles import Vehicle as Ov
        super().__init__()
        self._names = ['carpk']
        self._categories = {0: Ov.VEHICLE.CAR}
        self._colors = [(0, 255, 0)]

    def load_filename(self, path, db, line):
        import os
        from PIL import Image
        from .annotations import AerialImage
        seq = GenericGroup()
        parts = line.strip().split(';')
        image = AerialImage(path + parts[0])
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        image.gsd = 0.05
        root, extension = os.path.splitext(image.filename)
        txt_file = path + 'Annotations/' + os.path.basename(root) + '.txt'
        with open(txt_file) as ifs:
            lines = ifs.readlines()
        ifs.close()
        for ln in lines:
            pts = ln.strip().split(' ')
            obj = GenericObject()
            obj.bb = (int(pts[0]), int(pts[1]), int(pts[2]), int(pts[3]))
            obj.add_category(GenericCategory(self._categories[0]))
            image.add_object(obj)
        seq.add_image(image)
        return seq


class DRL(Database):
    def __init__(self):
        from images_framework.categories.vehicles import Vehicle as Ov
        super().__init__()
        self._names = ['drl']
        self._categories = {'pkw': Ov.VEHICLE.CAR, 'truck': Ov.VEHICLE.TRUCK, 'bus': Ov.VEHICLE.BUS}
        self._colors = get_palette(len(self._categories))

    def load_filename(self, path, db, line):
        import os
        from PIL import Image
        from .annotations import AerialImage
        seq = GenericGroup()
        parts = line.strip().split(';')
        image = AerialImage(path + parts[0])
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        image.gsd = 0.13
        for category in self._categories.keys():
            txt_file = path + parts[0][:-4] + '_' + category + '.samp'
            if not os.path.isfile(txt_file):
                continue
            with open(txt_file) as ifs:
                lines = ifs.readlines()
            ifs.close()
            for ln in lines:
                elems = ln.strip().split(' ')
                if elems[0].startswith('#') or elems[0].startswith('@'):
                    continue
                obj = GenericObject()
                obj.id = int(elems[0])
                center = (int(elems[2]), int(elems[3]))
                obj.bb = (center[0]-int(elems[4]), center[1]-int(elems[5]), center[0]+int(elems[4]), center[1]+int(elems[5]))
                angle = np.radians(float(elems[6]))
                pts = np.array(((obj.bb[0]-center[0], obj.bb[2]-center[0], obj.bb[2]-center[0], obj.bb[0]-center[0]), (obj.bb[1]-center[1], obj.bb[1]-center[1], obj.bb[3]-center[1], obj.bb[3]-center[1])))
                rot = np.array(((np.cos(angle), np.sin(angle)), (-np.sin(angle), np.cos(angle))))
                pts_proj = np.matmul(rot, pts)
                obj.obb = (pts_proj[0, 0]+center[0], pts_proj[1, 0]+center[1], pts_proj[0, 1]+center[0], pts_proj[1, 1]+center[1], pts_proj[0, 2]+center[0], pts_proj[1, 2]+center[1], pts_proj[0, 3]+center[0], pts_proj[1, 3]+center[1])
                obj.add_category(GenericCategory(self._categories[category]))
                image.add_object(obj)
        seq.add_image(image)
        return seq


class NWPU(Database):
    def __init__(self):
        from images_framework.categories.vehicles import Vehicle as Ov
        super().__init__()
        self._names = ['nwpu']
        self._categories = {1: Ov.VEHICLE.PLANE, 2: Ov.VEHICLE.SHIP, 3: Oi.STORAGE_TANK, 4: Oi.BASEBALL_DIAMOND, 5: Oi.TENNIS_COURT, 6: Oi.BASKETBALL_COURT, 7: Oi.GROUND_TRACK_FIELD, 8: Oi.HARBOR, 9: Oi.BRIDGE, 10: Ov.VEHICLE.CAR}
        self._colors = get_palette(len(self._categories))

    def load_filename(self, path, db, line):
        import os
        from PIL import Image
        seq = GenericGroup()
        parts = line.strip().split(';')
        image = GenericImage(path + parts[0])
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        root, extension = os.path.splitext(image.filename)
        txt_file = path + 'ground_truth/' + os.path.basename(root) + '.txt'
        with open(txt_file) as ifs:
            lines = ifs.readlines()
        ifs.close()
        for ln in lines:
            elem = ln.strip().split(',')
            elem = [el.replace('(', '').replace(')', '') for el in elem]
            obj = GenericObject()
            obj.bb = (int(elem[0]), int(elem[1]), int(elem[2]), int(elem[3]))
            obj.add_category(GenericCategory(self._categories[int(elem[4])]))
            image.add_object(obj)
        seq.add_image(image)
        return seq


class SpaceNet(Database):
    def __init__(self):
        super().__init__()
        self._names = ['spacenet']
        self._categories = {0: Oi.BUILDING}
        self._colors = [(0, 255, 0)]

    def load_filename(self, path, db, line):
        from PIL import Image
        from shapely import wkt
        from .utils import geometry2numpy
        from .annotations import AerialImage
        seq = GenericGroup()
        parts = line.strip().split(';')
        filepath = parts[0]
        image = AerialImage(path + filepath)
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        if len(parts) > 1:
            for i in range(1, len(parts), 2):
                geom = wkt.loads(parts[i+1])
                if geom.is_empty:
                    continue
                obj = GenericObject()
                obj.id = str(parts[i])
                obj.bb = (int(geom.bounds[0]), int(geom.bounds[1]), int(geom.bounds[2]), int(geom.bounds[3]))
                obj.multipolygon = [contour for contour in geometry2numpy(geom)]
                obj.add_category(GenericCategory(self._categories[0]))
                image.add_object(obj)
        seq.add_image(image)
        return seq


class Cityscapes(Database):
    def __init__(self):
        super().__init__()
        self._names = ['cityscapes', 'Cityscapes']
        self._categories = {num: Name(str(num)) for num in range(19)}
        self._colors = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)]

    def load_filename(self, path, db, line):
        import cv2
        from PIL import Image
        from .utils import load_geoimage, mask2contours
        label_mapping = {0: -1, 1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1, 7: 0, 8: 1, 9: -1, 10: -1, 11: 2, 12: 3, 13: 4, 14: -1, 15: -1, 16: -1, 17: 5, 18: -1, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: -1, 30: -1, 31: 16, 32: 17, 33: 18}
        seq = GenericGroup()
        parts = line.strip().split('\t')
        filepath = parts[0]
        image = GenericImage(path + filepath)
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        if len(parts) > 1:
            aux_filepath = parts[1]
            aux_image = GenericImage(path + aux_filepath)
            img, _ = load_geoimage(aux_image.filename)
            temp = img.copy()
            for key, value in label_mapping.items():
                img[temp == key] = value
            categories = list(np.unique(img))
            categories.remove(255)
            contours, labels = [], []
            for category in categories:
                mask = np.where((img == category), 255, 0).astype(np.uint8)
                for contour in mask2contours(mask):
                    contours.append(contour)
                    labels.append(str(category))
            for index in range(len(contours)):
                obj = GenericObject()
                bbox = cv2.boundingRect(contours[index])
                obj.bb = (bbox[0], bbox[1], bbox[2]+bbox[0], bbox[3]+bbox[1])
                obj.multipolygon = [contours[index]]
                obj.add_category(GenericCategory(Name(labels[index])))
                image.add_object(obj)
        seq.add_image(image)
        return seq


class LIP(Database):
    def __init__(self):
        super().__init__()
        self._names = ['lip', 'LIP']
        self._categories = {num: Name(str(num)) for num in range(20)}
        self._colors = [(0.0, 0.0, 0.0), (127.5, 0.0, 0.0), (254.00390625, 0.0, 0.0), (0.0, 84.66796875, 0.0), (169.3359375, 0.0, 50.80078125), (254.00390625, 84.66796875, 0.0), (0.0, 0.0, 84.66796875), (0.0, 118.53515625, 220.13671875), (84.66796875, 84.66796875, 0.0), (0.0, 84.66796875, 84.66796875), (84.66796875, 50.80078125, 0.0), (51.796875, 85.6640625, 127.5), (0.0, 127.5, 0.0), (0.0, 0.0, 254.00390625), (50.80078125, 169.3359375, 220.13671875), (0.0, 254.00390625, 254.00390625), (84.66796875, 254.00390625, 169.3359375), (169.3359375, 254.00390625, 84.66796875), (254.00390625, 254.00390625, 0.0), (254.00390625, 169.3359375, 0.0)]

    def load_filename(self, path, db, line):
        import cv2
        from PIL import Image
        from .utils import load_geoimage, mask2contours
        seq = GenericGroup()
        parts = line.strip().split(' ')
        filepath = parts[0]
        image = GenericImage(path + filepath)
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        if len(parts) > 1:
            aux_filepath = parts[1]
            aux_image = GenericImage(path + aux_filepath)
            img, _ = load_geoimage(aux_image.filename)
            categories = list(np.unique(img))
            contours, labels = [], []
            for category in categories:
                mask = np.where((img == category), 255, 0).astype(np.uint8)
                for contour in mask2contours(mask):
                    contours.append(contour)
                    labels.append(str(category))
            for index in range(len(contours)):
                obj = GenericObject()
                bbox = cv2.boundingRect(contours[index])
                obj.bb = (bbox[0], bbox[1], bbox[2]+bbox[0], bbox[3]+bbox[1])
                obj.multipolygon = [contours[index]]
                obj.add_category(GenericCategory(Name(labels[index])))
                image.add_object(obj)
        seq.add_image(image)
        return seq


class SegESolarScene(Database):
    def __init__(self):
        super().__init__()
        self._names = ['seg_geoai_panels', 'SegESolarScene']
        self._categories = {'bg': Oi.BACKGROUND, 'fg': Oi.SOLAR_PANEL}
        self._colors = [(0, 255, 255), (0, 255, 0)]

    def load_filename(self, path, db, line):
        from PIL import Image
        from shapely import wkt
        from .utils import geometry2numpy
        from .annotations import AerialImage
        seq = GenericGroup()
        parts = line.strip().split(';')
        filepath = parts[0]
        image = AerialImage(path + filepath)
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        if len(parts) > 1:
            for i in range(1, len(parts), 2):
                geom = wkt.loads(parts[i])
                if geom.is_empty:
                    continue
                obj = GenericObject()
                obj.bb = (int(geom.bounds[0]), int(geom.bounds[1]), int(geom.bounds[2]), int(geom.bounds[3]))
                obj.multipolygon = [contour for contour in geometry2numpy(geom)]
                obj.add_category(GenericCategory(self._categories[parts[i+1]]))
                image.add_object(obj)
        seq.add_image(image)
        return seq


class SegGeoAIPanels(Database):
    def __init__(self):
        super().__init__()
        self._names = ['SegGeoAIPanels']
        self._categories = {0: Oi.SOLAR_PANEL}
        self._colors = [(0, 255, 0)]

    def load_filename(self, path, db, line):
        from PIL import Image
        from shapely import wkt
        from .utils import geometry2numpy
        from .annotations import AerialImage
        seq = GenericGroup()
        parts = line.strip().split(';')
        filepath = parts[0]
        image = AerialImage(path + filepath)
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        if len(parts) > 1:
            for i in range(1, len(parts)):
                geom = wkt.loads(parts[i])
                if geom.is_empty:
                    continue
                obj = GenericObject()
                obj.bb = (int(geom.bounds[0]), int(geom.bounds[1]), int(geom.bounds[2]), int(geom.bounds[3]))
                obj.multipolygon = [contour for contour in geometry2numpy(geom)]
                obj.add_category(GenericCategory(self._categories[0]))
                image.add_object(obj)
        seq.add_image(image)
        return seq


class RecGeoAIPanels(Database):
    def __init__(self):
        from images_framework.categories.panels import Panel as Op
        super().__init__()
        self._names = ['RecGeoAIPanels']
        self._categories = {'1CC': Op.SOLAR_PANEL.HOT_CELL, 'SMC': Op.SOLAR_PANEL.HOT_CELL_CHAIN, 'CCS': Op.SOLAR_PANEL.SEVERAL_HOT_CELLS, '1PC': Op.SOLAR_PANEL.HOT_SPOT, 'PCS': Op.SOLAR_PANEL.SEVERAL_HOT_SPOTS, 'PID': Op.SOLAR_PANEL.POTENTIAL_INDUCED_DEGRADATION, 'DRT': Op.SOLAR_PANEL.DIRTY_PANEL, 'BRK': Op.SOLAR_PANEL.BROKEN_PANEL, 'DSC': Op.SOLAR_PANEL.DISCONNECTED_PANEL, 'SDW': Op.SOLAR_PANEL.SHADES, 'SDWD': Op.SOLAR_PANEL.SHADES_HOT_CELL_CHAIN, 'NDM': Op.SOLAR_PANEL.NO_DAMAGE}
        self._colors = get_palette(len(self._categories))

    def load_filename(self, path, db, line):
        from PIL import Image
        from shapely import wkt
        from .utils import geometry2numpy
        from .annotations import AerialImage
        seq = GenericGroup()
        parts = line.strip().split(';')
        filepath = parts[0]
        image = AerialImage(path + filepath)
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        if len(parts) > 1:
            for i in range(1, len(parts), 2):
                geom = wkt.loads(parts[i])
                if geom.is_empty:
                    continue
                obj = GenericObject()
                obj.bb = (int(geom.bounds[0]), int(geom.bounds[1]), int(geom.bounds[2]), int(geom.bounds[3]))
                obj.multipolygon = [contour for contour in geometry2numpy(geom)]
                obj.add_category(GenericCategory(self._categories[parts[i+1]]))
                image.add_object(obj)
        seq.add_image(image)
        return seq


class StanfordCars(Database):
    def __init__(self):
        from images_framework.categories.cars import Car as Oc
        super().__init__()
        self._names = ['stanford_cars', 'StanfordCars']
        self._categories = {0: Oc.VEHICLE.CAR.AMGENERAL_HUMMER_SUV_2000, 1: Oc.VEHICLE.CAR.ACURA_RL_SEDAN_2012, 2: Oc.VEHICLE.CAR.ACURA_TL_SEDAN_2012, 3: Oc.VEHICLE.CAR.ACURA_TL_TYPES_2008, 4: Oc.VEHICLE.CAR.ACURA_TSX_SEDAN_2012, 5: Oc.VEHICLE.CAR.ACURA_INTEGRA_TYPER_2001, 6: Oc.VEHICLE.CAR.ACURA_ZDX_HATCHBACK_2012, 7: Oc.VEHICLE.CAR.ASTONMARTIN_V8VANTAGE_CONVERTIBLE_2012, 8: Oc.VEHICLE.CAR.ASTONMARTIN_V8VANTAGE_COUPE_2012, 9: Oc.VEHICLE.CAR.ASTONMARTIN_VIRAGE_CONVERTIBLE_2012, 10: Oc.VEHICLE.CAR.ASTONMARTIN_VIRAGE_COUPE_2012, 11: Oc.VEHICLE.CAR.AUDI_RS4_CONVERTIBLE_2008,  12: Oc.VEHICLE.CAR.AUDI_A5_COUPE_2012, 13: Oc.VEHICLE.CAR.AUDI_TTS_COUPE_2012, 14: Oc.VEHICLE.CAR.AUDI_R8_COUPE_2012,  15: Oc.VEHICLE.CAR.AUDI_V8_SEDAN_1994, 16: Oc.VEHICLE.CAR.AUDI_100_SEDAN_1994, 17: Oc.VEHICLE.CAR.AUDI_100_WAGON_1994,  18: Oc.VEHICLE.CAR.AUDI_TT_HATCHBACK_2011, 19: Oc.VEHICLE.CAR.AUDI_S6_SEDAN_2011, 20: Oc.VEHICLE.CAR.AUDI_S5_CONVERTIBLE_2012,  21: Oc.VEHICLE.CAR.AUDI_S5_COUPE_2012, 22: Oc.VEHICLE.CAR.AUDI_S4_SEDAN_2012, 23: Oc.VEHICLE.CAR.AUDI_S4_SEDAN_2007,  24: Oc.VEHICLE.CAR.AUDI_TTRS_COUPE_2012, 25: Oc.VEHICLE.CAR.BMW_ACTIVEHYBRID5_SEDAN_2012,  26: Oc.VEHICLE.CAR.BMW_SERIES1_CONVERTIBLE_2012, 27: Oc.VEHICLE.CAR.BMW_SERIES1_COUPE_2012,  28: Oc.VEHICLE.CAR.BMW_SERIES3_SEDAN_2012, 29: Oc.VEHICLE.CAR.BMW_SERIES3_WAGON_2012,  30: Oc.VEHICLE.CAR.BMW_SERIES6_CONVERTIBLE_2007, 31: Oc.VEHICLE.CAR.BMW_X5_SUV_2007, 32: Oc.VEHICLE.CAR.BMW_X6_SUV_2012,  33: Oc.VEHICLE.CAR.BMW_M3_COUPE_2012, 34: Oc.VEHICLE.CAR.BMW_M5_SEDAN_2010, 35: Oc.VEHICLE.CAR.BMW_M6_CONVERTIBLE_2010,  36: Oc.VEHICLE.CAR.BMW_X3_SUV_2012, 37: Oc.VEHICLE.CAR.BMW_Z4_CONVERTIBLE_2012,  38: Oc.VEHICLE.CAR.BENTLEY_CONTINENTAL_CONVERTIBLE_2012, 39: Oc.VEHICLE.CAR.BENTLEY_ARNAGE_SEDAN_2009, 40: Oc.VEHICLE.CAR.BENTLEY_MULSANNE_SEDAN_2011, 41: Oc.VEHICLE.CAR.BENTLEY_CONTINENTAL_COUPE_2012, 42: Oc.VEHICLE.CAR.BENTLEY_CONTINENTAL_COUPE_2007, 43: Oc.VEHICLE.CAR.BENTLEY_CONTINENTAL_SEDAN_2007, 44: Oc.VEHICLE.CAR.BUGATTI_VEYRON_CONVERTIBLE_2009, 45: Oc.VEHICLE.CAR.BUGATTI_VEYRON_COUPE_2009,  46: Oc.VEHICLE.CAR.BUICK_REGAL_GS_2012, 47: Oc.VEHICLE.CAR.BUICK_RAINIER_SUV_2007, 48: Oc.VEHICLE.CAR.BUICK_VERANO_SEDAN_2012,  49: Oc.VEHICLE.CAR.BUICK_ENCLAVE_SUV_2012, 50: Oc.VEHICLE.CAR.CADILLAC_CTSV_SEDAN_2012, 51: Oc.VEHICLE.CAR.CADILLAC_SRX_SUV_2012,  52: Oc.VEHICLE.CAR.CADILLAC_ESCALADE_CREWCAB_2007, 53: Oc.VEHICLE.CAR.CHEVROLET_SILVERADO1500HYBRID_CREWCAB_2012,  54: Oc.VEHICLE.CAR.CHEVROLET_CORVETTE_CONVERTIBLE_2012, 55: Oc.VEHICLE.CAR.CHEVROLET_CORVETTE_ZR1_2012,  56: Oc.VEHICLE.CAR.CHEVROLET_CORVETTE_Z06_2007, 57: Oc.VEHICLE.CAR.CHEVROLET_TRAVERSE_SUV_2012,  58: Oc.VEHICLE.CAR.CHEVROLET_CAMARO_CONVERTIBLE_2012, 59: Oc.VEHICLE.CAR.CHEVROLET_HHR_SS_2010,  60: Oc.VEHICLE.CAR.CHEVROLET_IMPALA_SEDAN_2007, 61: Oc.VEHICLE.CAR.CHEVROLET_TAHOEHYBRID_SUV_2012, 62: Oc.VEHICLE.CAR.CHEVROLET_SONIC_SEDAN_2012, 63: Oc.VEHICLE.CAR.CHEVROLET_EXPRESS_CARGOVAN_2007, 64: Oc.VEHICLE.CAR.CHEVROLET_AVALANCHE_CREWCAB_2012, 65: Oc.VEHICLE.CAR.CHEVROLET_COBALT_SS_2010, 66: Oc.VEHICLE.CAR.CHEVROLET_MALIBUHYBRID_SEDAN_2010, 67: Oc.VEHICLE.CAR.CHEVROLET_TRAINBLAZER_SS_2009, 68: Oc.VEHICLE.CAR.CHEVROLET_SILVERADO2500HD_REGULARCAB_2012,  69: Oc.VEHICLE.CAR.CHEVROLET_SILVERADO1500CLASSIC_EXTENDEDCAB_2007, 70: Oc.VEHICLE.CAR.CHEVROLET_EXPRESS_VAN_2007,  71: Oc.VEHICLE.CAR.CHEVROLET_MONTECARLO_COUPE_2007, 72: Oc.VEHICLE.CAR.CHEVROLET_MALIBU_SEDAN_2007,  73: Oc.VEHICLE.CAR.CHEVROLET_SILVERADO1500_EXTENDEDCAB_2012, 74: Oc.VEHICLE.CAR.CHEVROLET_SILVERADO1500_REGULARCAB_2012,  75: Oc.VEHICLE.CAR.CHRYSLER_ASPEN_SUV_2009, 76: Oc.VEHICLE.CAR.CHRYSLER_SEBRING_CONVERTIBLE_2010,  77: Oc.VEHICLE.CAR.CHRYSLER_TOWN_MINIVAN_2012, 78: Oc.VEHICLE.CAR.CHRYSLER_300_STR8_2010,  79: Oc.VEHICLE.CAR.CHRYSLER_CROSSFIRE_CONVERTIBLE_2008, 80: Oc.VEHICLE.CAR.CHRYSLER_PTCRUISER_CONVERTIBLE_2008,  81: Oc.VEHICLE.CAR.DAEWOO_NUBIRA_WAGON_2002, 82: Oc.VEHICLE.CAR.DODGE_CALIBER_WAGON_2012,  83: Oc.VEHICLE.CAR.DODGE_CALIBER_WAGON_2007, 84: Oc.VEHICLE.CAR.DODGE_CARAVAN_MINIVAN_1997,  85: Oc.VEHICLE.CAR.DODGE_RAM_CREWCAB_2010, 86: Oc.VEHICLE.CAR.DODGE_RAM_QUADCAB_2009,  87: Oc.VEHICLE.CAR.DODGE_SPRINTER_CARGOVAN_2009, 88: Oc.VEHICLE.CAR.DODGE_JOURNEY_SUV_2012,  89: Oc.VEHICLE.CAR.DODGE_DAKOTA_CREWCAB_2010, 90: Oc.VEHICLE.CAR.DODGE_DAKOTA_CLUBCAB_2007, 91: Oc.VEHICLE.CAR.DODGE_MAGNUM_WAGON_2008, 92: Oc.VEHICLE.CAR.DODGE_CHALLENGER_SRT8_2011,  93: Oc.VEHICLE.CAR.DODGE_DURANGO_SUV_2012, 94: Oc.VEHICLE.CAR.DODGE_DURANGO_SUV_2007,  95: Oc.VEHICLE.CAR.DODGE_CHARGER_SEDAN_2012, 96: Oc.VEHICLE.CAR.DODGE_CHARGER_SRT8_2009, 97: Oc.VEHICLE.CAR.EAGLE_TALON_HATCHBACK_1998, 98: Oc.VEHICLE.CAR.FIAT_500_ABARTH_2012,  99: Oc.VEHICLE.CAR.FIAT_500_CONVERTIBLE_2012, 100: Oc.VEHICLE.CAR.FERRARI_FF_COUPE_2012,  101: Oc.VEHICLE.CAR.FERRARI_CALIFORNIA_CONVERTIBLE_2012, 102: Oc.VEHICLE.CAR.FERRARI_ITALIA_CONVERTIBLE_2012,  103: Oc.VEHICLE.CAR.FERRARI_ITALIA_COUPE_2012, 104: Oc.VEHICLE.CAR.FISKER_KARMA_SEDAN_2012,  105: Oc.VEHICLE.CAR.FORD_F450_CREWCAB_2012, 106: Oc.VEHICLE.CAR.FORD_MUSTANG_CONVERTIBLE_2007, 107: Oc.VEHICLE.CAR.FORD_FREESTAR_MINIVAN_2007, 108: Oc.VEHICLE.CAR.FORD_EXPEDITION_SUV_2009,  109: Oc.VEHICLE.CAR.FORD_EDGE_SUV_2012, 110: Oc.VEHICLE.CAR.FORD_RANGER_SUPERCAB_2011, 111: Oc.VEHICLE.CAR.FORD_GT_COUPE_2006,  112: Oc.VEHICLE.CAR.FORD_F150_REGULARCAB_2012, 113: Oc.VEHICLE.CAR.FORD_F150_REGULARCAB_2007,  114: Oc.VEHICLE.CAR.FORD_FOCUS_SEDAN_2007, 115: Oc.VEHICLE.CAR.FORD_ESERIES_WAGON_2012, 116: Oc.VEHICLE.CAR.FORD_FIESTA_SEDAN_2012, 117: Oc.VEHICLE.CAR.GMC_TERRAIN_SUV_2012, 118: Oc.VEHICLE.CAR.GMC_SAVANA_VAN_2012, 119: Oc.VEHICLE.CAR.GMC_YUKONHYBRID_SUV_2012, 120: Oc.VEHICLE.CAR.GMC_ACADIA_SUV_2012,  121: Oc.VEHICLE.CAR.GMC_CANYON_EXTENDEDCAB_2012, 122: Oc.VEHICLE.CAR.GMC_METRO_CONVERTIBLE_1993,  123: Oc.VEHICLE.CAR.HUMMER_H3T_CREWCAB_2010, 124: Oc.VEHICLE.CAR.HUMMER_H2SUT_CREWCAB_2009,  125: Oc.VEHICLE.CAR.HONDA_ODYSSEY_MINIVAN_2012, 126: Oc.VEHICLE.CAR.HONDA_ODYSSEY_MINIVAN_2007,  127: Oc.VEHICLE.CAR.HONDA_ACCORD_COUPE_2012, 128: Oc.VEHICLE.CAR.HONDA_ACCORD_SEDAN_2012,  129: Oc.VEHICLE.CAR.HYUNDAI_VELOSTER_HATCHBACK_2012, 130: Oc.VEHICLE.CAR.HYUNDAI_SANTAFE_SUV_2012, 131: Oc.VEHICLE.CAR.HYUNDAI_TUCSON_SUV_2012, 132: Oc.VEHICLE.CAR.HYUNDAI_VERACRUZ_SUV_2012, 133: Oc.VEHICLE.CAR.HYUNDAI_SONATAHYBRID_SEDAN_2012, 134: Oc.VEHICLE.CAR.HYUNDAI_ELANTRA_SEDAN_2007, 135: Oc.VEHICLE.CAR.HYUNDAI_ACCENT_SEDAN_2012, 136: Oc.VEHICLE.CAR.HYUNDAI_GENESIS_SEDAN_2012, 137: Oc.VEHICLE.CAR.HYUNDAI_SONATA_SEDAN_2012, 138: Oc.VEHICLE.CAR.HYUNDAI_ELANTRA_HATCHBACK_2012, 139: Oc.VEHICLE.CAR.HYUNDAI_AZERA_SEDAN_2012, 140: Oc.VEHICLE.CAR.INFINITI_G_COUPE_2012, 141: Oc.VEHICLE.CAR.INFINITI_QX56_SUV_2011, 142: Oc.VEHICLE.CAR.ISUZU_ASCENDER_SUV_2008, 143: Oc.VEHICLE.CAR.JAGUAR_XK_XKR_2012, 144: Oc.VEHICLE.CAR.JEEP_PATRIOT_SUV_2012, 145: Oc.VEHICLE.CAR.JEEP_WRANGLER_SUV_2012, 146: Oc.VEHICLE.CAR.JEEP_LIBERTY_SUV_2012, 147: Oc.VEHICLE.CAR.JEEP_GRANDCHEROKEE_SUV_2012, 148: Oc.VEHICLE.CAR.JEEP_COMPASS_SUV_2012, 149: Oc.VEHICLE.CAR.LAMBORGHINI_REVENTON_COUPE_2012, 150: Oc.VEHICLE.CAR.LAMBORGHINI_AVENTADOR_COUPE_2012, 151: Oc.VEHICLE.CAR.LAMBORGHINI_GALLARDO_SUPERLEGGERA_2012, 152: Oc.VEHICLE.CAR.LAMBORGHINI_DIABLO_COUPE_2001, 153: Oc.VEHICLE.CAR.LANDROVER_RANGEROVER_SUV_2012, 154: Oc.VEHICLE.CAR.LANDROVER_LR2_SUV_2012, 155: Oc.VEHICLE.CAR.LINCOLN_TOWN_SEDAN_2011, 156: Oc.VEHICLE.CAR.MINI_COOPER_CONVERTIBLE_2012, 157: Oc.VEHICLE.CAR.MAYBACH_LANDAULET_CONVERTIBLE_2012, 158: Oc.VEHICLE.CAR.MAZDA_TRIBUTE_SUV_2011, 159: Oc.VEHICLE.CAR.MCLAREN_MP4_COUPE_2012, 160: Oc.VEHICLE.CAR.MERCEDES_300_CONVERTIBLE_1993, 161: Oc.VEHICLE.CAR.MERCEDES_C_SEDAN_2012, 162: Oc.VEHICLE.CAR.MERCEDES_SL_COUPE_2009, 163: Oc.VEHICLE.CAR.MERCEDES_E_SEDAN_2012, 164: Oc.VEHICLE.CAR.MERCEDES_S_SEDAN_2012, 165: Oc.VEHICLE.CAR.MERCEDES_SPRINTER_VAN_2012, 166: Oc.VEHICLE.CAR.MITSUBISHI_LANCER_SEDAN_2012, 167: Oc.VEHICLE.CAR.NISSAN_LEAF_HATCHBACK_2012, 168: Oc.VEHICLE.CAR.NISSAN_NV_VAN_2012, 169: Oc.VEHICLE.CAR.NISSAN_JUKE_HATCHBACK_2012, 170: Oc.VEHICLE.CAR.NISSAN_240SX_COUPE_1998, 171: Oc.VEHICLE.CAR.PLYMOUTH_NEON_COUPE_1999, 172: Oc.VEHICLE.CAR.PORSCHE_PANAMERA_SEDAN_2012, 173: Oc.VEHICLE.CAR.RAM_CV_MINIVAN_2012, 174: Oc.VEHICLE.CAR.ROLLSROYCE_PHANTOM_COUPE_2012, 175: Oc.VEHICLE.CAR.ROLLSROYCE_GHOST_SEDAN_2012, 176: Oc.VEHICLE.CAR.ROLLSROYCE_PHANTOM_SEDAN_2012, 177: Oc.VEHICLE.CAR.SCION_XD_HATCHBACK_2012, 178: Oc.VEHICLE.CAR.SPYKER_C8_CONVERTIBLE_2009, 179: Oc.VEHICLE.CAR.SPYKER_C8_COUPE_2009, 180: Oc.VEHICLE.CAR.SUZUKI_AERIO_SEDAN_2007, 181: Oc.VEHICLE.CAR.SUZUKI_KIZASHI_SEDAN_2012, 182: Oc.VEHICLE.CAR.SUZUKI_SX4_HATCHBACK_2012, 183: Oc.VEHICLE.CAR.SUZUKI_SX4_SEDAN_2012, 184: Oc.VEHICLE.CAR.TESLA_MODELS_SEDAN_2012, 185: Oc.VEHICLE.CAR.TOYOTA_SEQUOIA_SUV_2012, 186: Oc.VEHICLE.CAR.TOYOTA_CAMRY_SEDAN_2012, 187: Oc.VEHICLE.CAR.TOYOTA_COROLLA_SEDAN_2012, 188: Oc.VEHICLE.CAR.TOYOTA_4RUNNER_SUV_2012, 189: Oc.VEHICLE.CAR.VOLKSWAGEN_GOLF_HATCHBACK_2012, 190: Oc.VEHICLE.CAR.VOLKSWAGEN_GOLF_HATCHBACK_1991, 191: Oc.VEHICLE.CAR.VOLKSWAGEN_BEETLE_HATCHBACK_2012, 192: Oc.VEHICLE.CAR.VOLV0_C30_HATCHBACK_2012, 193: Oc.VEHICLE.CAR.VOLV0_240_SEDAN_1993, 194: Oc.VEHICLE.CAR.VOLVO_XC90_SUV_2007, 195: Oc.VEHICLE.CAR.SMART_FORTWO_CONVERTIBLE_2012}
        self._colors = get_palette(len(self._categories))

    def load_filename(self, path, db, line):
        from PIL import Image
        seq = GenericGroup()
        parts = line.strip().split(';')
        filepath = parts[0]
        image = GenericImage(path + filepath)
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        obj = GenericObject()
        obj.bb = (int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
        obj.add_category(GenericCategory(self._categories[int(parts[5])]))
        image.add_object(obj)
        seq.add_image(image)
        return seq


class WorldView3(Database):
    def __init__(self):
        from images_framework.categories.vehicles import Vehicle as Ov
        super().__init__()
        self._names = ['maxar', 'cuende']
        self._categories = {0: Ov.VEHICLE.CAR}
        self._colors = [(0, 255, 0)]

    def load_filename(self, path, db, line):
        import rasterio
        from .annotations import AerialImage
        seq = GenericGroup()
        parts = line.strip().split(';')
        filepath = parts[0]
        image = AerialImage(path + filepath)
        src_raster = rasterio.open(image.filename, 'r')
        width = src_raster.width
        height = src_raster.height
        image.tile = np.array([0, 0, width, height])
        image.gsd = 0.3
        if len(parts) > 1:
            import xml.etree.ElementTree as ET
            filepath = parts[1]
            tree = ET.parse(path + filepath)
            root = tree.getroot()
            for obj in root.findall('object'):
                bbox = obj.find('bndbox')
                obj = GenericObject()
                obj.bb = (int(bbox.find('xmin').text), int(bbox.find('ymin').text), int(bbox.find('xmax').text), int(bbox.find('ymax').text))
                obj.add_category(GenericCategory(self._categories[0]))
                image.add_object(obj)
        seq.add_image(image)
        return seq

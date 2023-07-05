#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import abc
import numpy as np
from .annotations import GenericGroup, GenericImage, GenericObject, GenericCategory
from .categories import Name, FaceLandmarkPart as Lp, Category as Oi


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
        self._mapping = {}
        self._categories = []
        self._colors = []

    @abc.abstractmethod
    def load_filename(self, path, db, line):
        pass

    def get_names(self):
        return self._names

    def get_mapping(self):
        return self._mapping

    def get_categories(self):
        return self._categories

    def get_colors(self):
        return self._colors


class PTS68(Database):
    def __init__(self):
        super().__init__()
        self._names = ['300w_public', '300w_private']
        self._mapping = {Lp.LEYEBROW: (1, 119, 2, 121, 3), Lp.REYEBROW: (4, 124, 5, 126, 6), Lp.LEYE: (7, 138, 139, 8, 141, 142), Lp.REYE: (11, 144, 145, 12, 147, 148), Lp.NOSE: (128, 129, 130, 17, 16, 133, 134, 135, 18), Lp.TMOUTH: (20, 150, 151, 22, 153, 154, 21, 165, 164, 163, 162, 161), Lp.BMOUTH: (156, 157, 23, 159, 160, 168, 167, 166), Lp.LEAR: (101, 102, 103, 104, 105, 106), Lp.REAR: (112, 113, 114, 115, 116, 117), Lp.CHIN: (107, 108, 24, 110, 111)}
        self._categories = [Oi.FACE]
        self._colors = [(0, 255, 0)]

    def load_filename(self, path, db, line):
        from PIL import Image
        from .annotations import FaceObject, FaceLandmark
        seq = GenericGroup()
        parts = line.strip().split(';')
        image = GenericImage(path + parts[0])
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        obj = FaceObject()
        obj.bb = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        obj.add_category(GenericCategory(Oi.FACE))
        indices = [101, 102, 103, 104, 105, 106, 107, 108, 24, 110, 111, 112, 113, 114, 115, 116, 117, 1, 119, 2, 121, 3, 4, 124, 5, 126, 6, 128, 129, 130, 17, 16, 133, 134, 135, 18, 7, 138, 139, 8, 141, 142, 11, 144, 145, 12, 147, 148, 20, 150, 151, 22, 153, 154, 21, 156, 157, 23, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168]
        for idx in range(0, len(indices)):
            label = indices[idx]
            lp = list(self._mapping.keys())[next((ids for ids, xs in enumerate(self._mapping.values()) for x in xs if x == label), None)]
            pos = (float(parts[(2*idx)+5]), float(parts[(2*idx)+6]))
            obj.add_landmark(FaceLandmark(label, lp, pos, True))
        image.add_object(obj)
        seq.add_image(image)
        return seq


class COFW(Database):
    def __init__(self):
        super().__init__()
        self._names = ['cofw']
        self._mapping = {Lp.LEYEBROW: (1, 101, 3, 102), Lp.REYEBROW: (4, 103, 6, 104), Lp.LEYE: (7, 9, 8, 10, 105), Lp.REYE: (11, 13, 12, 14, 106), Lp.NOSE: (16, 17, 18, 107), Lp.TMOUTH: (20, 22, 21, 108), Lp.BMOUTH: (109, 23), Lp.CHIN: (24,)}
        self._categories = [Oi.FACE]
        self._colors = [(0, 255, 0)]

    def load_filename(self, path, db, line):
        from PIL import Image
        from .annotations import FaceObject, FaceLandmark
        seq = GenericGroup()
        parts = line.strip().split(';')
        image = GenericImage(path + parts[0])
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        obj = FaceObject()
        obj.bb = (float(parts[1]), float(parts[2]), float(parts[1])+float(parts[3]), float(parts[2])+float(parts[4]))
        obj.add_category(GenericCategory(Oi.FACE))
        indices = [1, 6, 3, 4, 101, 102, 103, 104, 7, 12, 8, 11, 9, 10, 13, 14, 105, 106, 16, 18, 17, 107, 20, 21, 22, 108, 109, 23, 24]
        for idx in range(0, len(indices)):
            label = indices[idx]
            lp = list(self._mapping.keys())[next((ids for ids, xs in enumerate(self._mapping.values()) for x in xs if x == label), None)]
            pos = (float(parts[(3*idx)+5]), float(parts[(3*idx)+6]))
            vis = float(parts[(3*idx)+7]) == 0.0
            obj.add_landmark(FaceLandmark(label, lp, pos, vis))
        image.add_object(obj)
        seq.add_image(image)
        return seq


class AFLW(Database):
    def __init__(self):
        super().__init__()
        self._names = ['aflw']
        self._mapping = {Lp.LEYEBROW: (1, 2, 3), Lp.REYEBROW: (4, 5, 6), Lp.LEYE: (7, 101, 8), Lp.REYE: (11, 102, 12), Lp.NOSE: (16, 17, 18), Lp.TMOUTH: (20, 103, 21), Lp.LEAR: (15,), Lp.REAR: (19,), Lp.CHIN: (24,)}
        self._categories = [Oi.FACE]
        self._colors = [(0, 255, 0)]

    def load_filename(self, path, db, line):
        from PIL import Image
        from scipy.spatial.transform import Rotation
        from .annotations import FaceObject, FaceAttribute, FaceLandmark
        seq = GenericGroup()
        parts = line.strip().split(';')
        image = GenericImage(path + parts[0])
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        obj = FaceObject()
        obj.bb = (int(parts[1]), int(parts[2]), int(parts[1])+int(parts[3]), int(parts[2])+int(parts[4]))
        obj.add_category(GenericCategory(Oi.FACE))
        obj.headpose = Rotation.from_euler('YXZ', [float(parts[5]), float(parts[6]), float(parts[7])], degrees=True).as_matrix()
        obj.add_attribute(FaceAttribute('gender', 'male' if parts[8] == 'm' else 'female'))
        obj.add_attribute(FaceAttribute('glasses', bool(parts[9])))
        num_landmarks = int(parts[10])
        indices = [1, 2, 3, 4, 5, 6, 7, 101, 8, 11, 102, 12, 15, 16, 17, 18, 19, 20, 103, 21, 24]
        for idx in range(0, num_landmarks):
            label = indices[int(parts[(3*idx)+11])-1]
            lp = list(self._mapping.keys())[next((ids for ids, xs in enumerate(self._mapping.values()) for x in xs if x == label), None)]
            pos = (float(parts[(3*idx)+12]), float(parts[(3*idx)+13]))
            obj.add_landmark(FaceLandmark(label, lp, pos, True))
        image.add_object(obj)
        seq.add_image(image)
        return seq


class WFLW(Database):
    def __init__(self):
        super().__init__()
        self._names = ['wflw']
        self._mapping = {Lp.LEYEBROW: (1, 134, 2, 136, 3, 138, 139, 140, 141), Lp.REYEBROW: (6, 147, 148, 149, 150, 4, 143, 5, 145), Lp.LEYE: (7, 161, 9, 163, 8, 165, 10, 167, 196), Lp.REYE: (11, 169, 13, 171, 12, 173, 14, 175, 197), Lp.NOSE: (151, 152, 153, 17, 16, 156, 157, 158, 18), Lp.TMOUTH: (20, 177, 178, 22, 180, 181, 21, 192, 191, 190, 189, 188), Lp.BMOUTH: (187, 186, 23, 184, 183, 193, 194, 195), Lp.LEAR: (100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110), Lp.REAR: (122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132), Lp.CHIN: (111, 112, 113, 114, 115, 24, 117, 118, 119, 120, 121)}
        self._categories = [Oi.FACE]
        self._colors = [(0, 255, 0)]

    def load_filename(self, path, db, line):
        from PIL import Image
        from .annotations import FaceObject, FaceLandmark
        seq = GenericGroup()
        parts = line.strip().split(';')
        image = GenericImage(path + parts[0])
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        obj = FaceObject()
        obj.bb = (int(parts[1]), int(parts[2]), int(parts[1])+int(parts[3]), int(parts[2])+int(parts[4]))
        obj.add_category(GenericCategory(Oi.FACE))
        indices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 24, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 1, 134, 2, 136, 3, 138, 139, 140, 141, 4, 143, 5, 145, 6, 147, 148, 149, 150, 151, 152, 153, 17, 16, 156, 157, 158, 18, 7, 161, 9, 163, 8, 165, 10, 167, 11, 169, 13, 171, 12, 173, 14, 175, 20, 177, 178, 22, 180, 181, 21, 183, 184, 23, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197]
        for idx in range(0, len(indices)):
            label = indices[idx]
            lp = list(self._mapping.keys())[next((ids for ids, xs in enumerate(self._mapping.values()) for x in xs if x == label), None)]
            pos = (float(parts[(2*idx)+11]), float(parts[(2*idx)+12]))
            obj.add_landmark(FaceLandmark(label, lp, pos, True))
        image.add_object(obj)
        seq.add_image(image)
        return seq


class DAD(Database):
    def __init__(self):
        super().__init__()
        self._names = ['dad']
        self._mapping = {Lp.LEYEBROW: (1983, 2189, 3708, 336, 335, 3153, 3705, 2178, 3684, 3741, 3148, 3696, 2585, 2565, 2567, 3764), Lp.REYEBROW: (570, 694, 3865, 17, 16, 2134, 3863, 673, 3851, 3880, 2121, 3859, 1448, 1428, 1430, 3893), Lp.LEYE: (2441, 2446, 2382, 2381, 2383, 2496, 3690, 2493, 2491, 2465, 3619, 3632, 2505, 2273, 2276, 2355, 2295, 2359, 2267, 2271, 2403, 2437), Lp.REYE: (1183, 1194, 1033, 1023, 1034, 1345, 3856, 1342, 1340, 1243, 3827, 3833, 1354, 824, 827, 991, 883, 995, 814, 822, 1096, 1175), Lp.NOSE: (3540, 3704, 3555, 3560, 3561, 3501, 3526, 3563, 2793, 2751, 3092, 3099, 3102, 2205, 2193, 2973, 2868, 2921, 2920, 1676, 1623, 2057, 2064, 2067, 723, 702, 1895, 1757, 1818, 1817, 3515, 3541), Lp.TMOUTH: (2828, 2832, 2833, 2850, 2813, 2811, 2774, 3546, 1657, 1694, 1696, 1735, 1716, 1715, 1711, 1719, 1748, 1740, 1667, 1668, 3533, 2785, 2784, 2855, 2863, 2836), Lp.BMOUTH: (2891, 2890, 2892, 2928, 2937, 3509, 1848, 1826, 1789, 1787, 1788, 1579, 1773, 1774, 1795, 1802, 1865, 3503, 2948, 2905, 2898, 2881, 2880, 2715), Lp.LEAR: (3386, 3381, 1962, 2213, 2259, 2257, 2954, 3171, 2003), Lp.REAR: (3554, 576, 2159, 1872, 798, 802, 731, 567, 3577, 3582), Lp.CHIN: (3390, 3391, 3396, 3400, 3599, 3593, 3588), Lp.FOREHEAD: (3068, 2196, 2091, 3524, 628, 705, 2030)}
        self._categories = [Oi.FACE]
        self._colors = [(0, 255, 0)]

    def load_filename(self, path, db, line):
        import itertools
        from PIL import Image
        from scipy.spatial.transform import Rotation
        from .annotations import FaceObject, FaceAttribute, FaceLandmark
        seq = GenericGroup()
        parts = line.strip().split(';')
        image = GenericImage(path + parts[0])
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        obj = FaceObject()
        obj.bb = (int(round(float(parts[1]))), int(round(float(parts[2]))), int(round(float(parts[1])))+int(round(float(parts[3]))), int(round(float(parts[2])))+int(round(float(parts[4]))))
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
            for idx in list(itertools.chain.from_iterable(self._mapping.values())):
                lp = list(self._mapping.keys())[next((ids for ids, xs in enumerate(self._mapping.values()) for x in xs if x == idx), None)]
                pos = (float(flame_vertices2d[idx, 0]), height-float(flame_vertices2d[idx, 1]))
                obj.add_landmark(FaceLandmark(idx, lp, pos, True))
        if len(parts) != 8:  # train, test
            obj.add_attribute(FaceAttribute('quality', parts[len(parts)-7]))
            obj.add_attribute(FaceAttribute('gender',  parts[len(parts)-6]))
            obj.add_attribute(FaceAttribute('expression', bool(parts[len(parts)-5] == 'True')))
            obj.add_attribute(FaceAttribute('age', parts[len(parts)-4]))
            obj.add_attribute(FaceAttribute('occlusions', bool(parts[len(parts)-3] == 'True')))
            obj.add_attribute(FaceAttribute('pose', parts[len(parts)-2]))
            obj.add_attribute(FaceAttribute('standard_light', bool(parts[len(parts)-1] == 'True')))
        image.add_object(obj)
        seq.add_image(image)
        return seq


class AFLW2000(Database):
    def __init__(self):
        super().__init__()
        self._names = ['300wlp', 'aflw2000']
        self._mapping = {Lp.LEYEBROW: (1, 119, 2, 121, 3), Lp.REYEBROW: (4, 124, 5, 126, 6), Lp.LEYE: (7, 138, 139, 8, 141, 142), Lp.REYE: (11, 144, 145, 12, 147, 148), Lp.NOSE: (128, 129, 130, 17, 16, 133, 134, 135, 18), Lp.TMOUTH: (20, 150, 151, 22, 153, 154, 21, 165, 164, 163, 162, 161), Lp.BMOUTH: (156, 157, 23, 159, 160, 168, 167, 166), Lp.LEAR: (101, 102, 103, 104, 105, 106), Lp.REAR: (112, 113, 114, 115, 116, 117), Lp.CHIN: (107, 108, 24, 110, 111)}
        self._categories = [Oi.FACE]
        self._colors = [(0, 255, 0)]

    def load_filename(self, path, db, line):
        import cv2
        import itertools
        from PIL import Image
        from scipy.spatial.transform import Rotation
        from .annotations import FaceObject, FaceLandmark
        seq = GenericGroup()
        parts = line.strip().split(';')
        image = GenericImage(path + parts[0])
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        obj = FaceObject()
        obj.add_category(GenericCategory(Oi.FACE))
        euler = [float(parts[2]), float(parts[1]), float(parts[3])]
        obj.headpose = Rotation.from_euler('XYZ', euler, degrees=True).as_matrix()
        # Skip images with angles outside the range (-99, 99)
        if np.any(np.abs(euler) > 99):
            return seq
        indices = [101, 102, 103, 104, 105, 106, 107, 108, 24, 110, 111, 112, 113, 114, 115, 116, 117, 1, 119, 2, 121, 3, 4, 124, 5, 126, 6, 128, 129, 130, 17, 16, 133, 134, 135, 18, 7, 138, 139, 8, 141, 142, 11, 144, 145, 12, 147, 148, 20, 150, 151, 22, 153, 154, 21, 156, 157, 23, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168]
        for idx in range(0, len(indices)):
            label = indices[idx]
            lp = list(self._mapping.keys())[next((ids for ids, xs in enumerate(self._mapping.values()) for x in xs if x == label), None)]
            pos = (int(round(float(parts[(2*idx)+4]))), int(round(float(parts[(2*idx)+5]))))
            obj.add_landmark(FaceLandmark(label, lp, pos, True))
        obj.bb = cv2.boundingRect(np.array([[pt.pos for pt in list(itertools.chain.from_iterable(obj.landmarks.values()))]]))
        obj.bb = (obj.bb[0], obj.bb[1], obj.bb[0]+obj.bb[2], obj.bb[1]+obj.bb[3])
        image.add_object(obj)
        seq.add_image(image)
        return seq


class Biwi(Database):
    def __init__(self):
        super().__init__()
        self._names = ['biwi']
        self._categories = [Oi.FACE]
        self._colors = [(0, 255, 0)]

    def load_filename(self, path, db, line):
        from PIL import Image
        from scipy.spatial.transform import Rotation
        from .annotations import FaceObject
        seq = GenericGroup()
        parts = line.strip().split(';')
        image = GenericImage(path + parts[0])
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        obj = FaceObject()
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
        self._categories = [Oi.FACE]
        self._colors = [(0, 255, 0)]

    def load_filename(self, path, db, line):
        from PIL import Image
        from scipy.spatial.transform import Rotation
        from .annotations import FaceObject
        seq = GenericGroup()
        parts = line.strip().split(';')
        image = GenericImage(path + parts[0])
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        obj = FaceObject()
        obj.bb = (int(parts[1]), int(parts[2]), int(parts[1])+int(parts[3]), int(parts[2])+int(parts[4]))
        obj.add_category(GenericCategory(Oi.FACE))
        obj.headpose = Rotation.from_euler('XYZ', [float(parts[6]), float(parts[5]), float(parts[7])], degrees=True).as_matrix()
        image.add_object(obj)
        seq.add_image(image)
        return seq


class XView(Database):
    def __init__(self):
        super().__init__()
        self._names = ['xview']
        self._mapping = {11: Oi.FIXED_WING_AIRCRAFT, 12: Oi.FIXED_WING_AIRCRAFT.SMALL_AIRCRAFT, 13: Oi.FIXED_WING_AIRCRAFT.CARGO_PLANE, 15: Oi.HELICOPTER, 17: Oi.PASSENGER_VEHICLE, 18: Oi.PASSENGER_VEHICLE.SMALL_CAR, 19: Oi.PASSENGER_VEHICLE.BUS, 20: Oi.TRUCK.PICKUP_TRUCK, 21: Oi.TRUCK.UTILITY_TRUCK, 23: Oi.TRUCK, 24: Oi.TRUCK.CARGO_TRUCK, 25: Oi.TRUCK.TRUCK_BOX, 26: Oi.TRUCK.TRUCK_TRACTOR, 27: Oi.TRUCK.TRAILER, 28: Oi.TRUCK.TRUCK_FLATBED, 29: Oi.TRUCK.TRUCK_LIQUID, 32: Oi.ENGINEERING_VEHICLE.CRANE_TRUCK, 33: Oi.RAILWAY_VEHICLE, 34: Oi.RAILWAY_VEHICLE.PASSENGER_CAR, 35: Oi.RAILWAY_VEHICLE.CARGO_CAR, 36: Oi.RAILWAY_VEHICLE.FLAT_CAR, 37: Oi.RAILWAY_VEHICLE.TANK_CAR, 38: Oi.RAILWAY_VEHICLE.LOCOMOTIVE, 40: Oi.MARITIME_VESSEL, 41: Oi.MARITIME_VESSEL.MOTORBOAT, 42: Oi.MARITIME_VESSEL.SAILBOAT, 44: Oi.MARITIME_VESSEL.TUGBOAT, 45: Oi.MARITIME_VESSEL.BARGE, 47: Oi.MARITIME_VESSEL.FISHING_VESSEL, 49: Oi.MARITIME_VESSEL.FERRY, 50: Oi.MARITIME_VESSEL.YATCH, 51: Oi.MARITIME_VESSEL.CONTAINER_SHIP, 52: Oi.MARITIME_VESSEL.OIL_TANKER, 53: Oi.ENGINEERING_VEHICLE, 54: Oi.ENGINEERING_VEHICLE.TOWER_CRANE, 55: Oi.ENGINEERING_VEHICLE.CONTAINER_CRANE, 56: Oi.ENGINEERING_VEHICLE.REACH_STACKER, 57: Oi.ENGINEERING_VEHICLE.STRADDLE_CARRIER, 59: Oi.ENGINEERING_VEHICLE.MOBILE_CRANE, 60: Oi.ENGINEERING_VEHICLE.DUMP_TRUCK, 61: Oi.ENGINEERING_VEHICLE.HAUL_TRUCK, 62: Oi.ENGINEERING_VEHICLE.SCRAPER_TRACTOR, 63: Oi.ENGINEERING_VEHICLE.FRONT_LOADER, 64: Oi.ENGINEERING_VEHICLE.EXCAVATOR, 65: Oi.ENGINEERING_VEHICLE.CEMENT_MIXER, 66: Oi.ENGINEERING_VEHICLE.GROUND_GRADER, 71: Oi.BUILDING.HUT_TENT, 72: Oi.BUILDING.SHED, 73: Oi.BUILDING, 74: Oi.BUILDING.AIRCRAFT_HANGAR, 76: Oi.BUILDING.DAMAGED_BUILDING, 77: Oi.BUILDING.FACILITY, 79: Oi.CONSTRUCTION_SITE, 83: Oi.VEHICLE_LOT, 84: Oi.HELIPAD, 86: Oi.STORAGE_TANK, 89: Oi.SHIPPING_CONTAINER_LOT, 91: Oi.SHIPPING_CONTAINER, 93: Oi.PYLON, 94: Oi.TOWER_STRUCTURE}
        self._categories = list(self._mapping.values())
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
            if cat not in self._mapping.keys():
                continue
            obj.add_category(GenericCategory(self._mapping[cat]))
            # if self._mapping[cat] is not Oi.PASSENGER_VEHICLE.SMALL_CAR:
            #     continue
            image.add_object(obj)
        if len(image.objects) > 0:
            seq.add_image(image)
        return seq


class XView2(Database):
    def __init__(self):
        import images_framework.categories.panels
        super().__init__()
        self._names = ['xview2']
        self._mapping = {'building': Oi.BUILDING, 'un-classified': Oi.BUILDING.UNCLASSIFIED, 'no-damage': Oi.BUILDING.NO_DAMAGE, 'minor-damage': Oi.BUILDING.MINOR_DAMAGE, 'major-damage': Oi.BUILDING.MAJOR_DAMAGE, 'destroyed': Oi.BUILDING.DESTROYED}
        self._categories = list(self._mapping.values())
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
                    obj.add_category(GenericCategory(self._mapping[feat['properties']['subtype']]))
                else:
                    obj.add_category(GenericCategory(self._mapping[feat['properties']['feature_type']]))
                image.add_object(obj)
            seq.add_image(image)
        return seq


class DOTA(Database):
    def __init__(self):
        super().__init__()
        self._names = ['dota_1.0', 'dota_1.5', 'dota_2.0']
        self._mapping = {'ship': Oi.SHIP, 'storage-tank': Oi.STORAGE_TANK, 'baseball-diamond': Oi.BASEBALL_DIAMOND, 'tennis-court': Oi.TENNIS_COURT, 'basketball-court': Oi.BASKETBALL_COURT, 'ground-track-field': Oi.GROUND_TRACK_FIELD, 'bridge': Oi.BRIDGE, 'large-vehicle': Oi.LARGE_VEHICLE, 'small-vehicle': Oi.SMALL_VEHICLE, 'helicopter': Oi.HELICOPTER, 'swimming-pool': Oi.SWIMMING_POOL, 'roundabout': Oi.ROUNDABOUT, 'soccer-ball-field': Oi.SOCCER_BALL_FIELD, 'plane': Oi.PLANE, 'harbor': Oi.HARBOR, 'container-crane': Oi.ENGINEERING_VEHICLE.CONTAINER_CRANE, 'airport': Oi.AIRPORT, 'helipad': Oi.HELIPAD}
        self._categories = list(self._mapping.values())
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
            obj.add_category(GenericCategory(self._mapping[elem[8]]))
            obj.confidence = 1 - int(elem[9])  # 0 represents a difficult object
            image.add_object(obj)
        seq.add_image(image)
        return seq


class COWC(Database):
    def __init__(self):
        super().__init__()
        self._names = ['cowc']
        self._categories = [Oi.SMALL_VEHICLE]
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
        super().__init__()
        self._names = ['carpk']
        self._categories = [Oi.SMALL_VEHICLE]
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
        super().__init__()
        self._names = ['drl']
        self._mapping = {'pkw': Oi.PASSENGER_VEHICLE.SMALL_CAR, 'truck': Oi.TRUCK, 'bus': Oi.PASSENGER_VEHICLE.BUS}
        self._categories = list(self._mapping.values())
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
        for category in self._mapping.keys():
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
                obj.add_category(GenericCategory(self._mapping[category]))
                image.add_object(obj)
        seq.add_image(image)
        return seq


class NWPU(Database):
    def __init__(self):
        super().__init__()
        self._names = ['nwpu']
        self._mapping = {1: Oi.PLANE, 2: Oi.SHIP, 3: Oi.STORAGE_TANK, 4: Oi.BASEBALL_DIAMOND, 5: Oi.TENNIS_COURT, 6: Oi.BASKETBALL_COURT, 7: Oi.GROUND_TRACK_FIELD, 8: Oi.HARBOR, 9: Oi.BRIDGE, 10: Oi.SMALL_VEHICLE}
        self._categories = list(self._mapping.values())
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
            obj.add_category(GenericCategory(self._mapping[int(elem[4])]))
            image.add_object(obj)
        seq.add_image(image)
        return seq


class SpaceNet(Database):
    def __init__(self):
        super().__init__()
        self._names = ['spacenet']
        self._categories = [Oi.BUILDING]
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
        self._names = ['cityscapes']
        self._categories = [Name(str(num)) for num in range(19)]
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
        self._names = ['lip']
        self._categories = [Name(str(num)) for num in range(20)]
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
        self._names = ['seg_esolar_scene']
        self._mapping = {'bg': Oi.BACKGROUND, 'fg': Oi.SOLAR_PANEL}
        self._categories = list(self._mapping.values())
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
                obj.add_category(GenericCategory(self._mapping[parts[i+1]]))
                image.add_object(obj)
        seq.add_image(image)
        return seq


class SegGeoAIPanels(Database):
    def __init__(self):
        super().__init__()
        self._names = ['seg_geoai_panels']
        self._categories = [Oi.SOLAR_PANEL]
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
        import images_framework.categories.panels
        super().__init__()
        self._names = ['rec_geoai_panels']
        self._mapping = {'1CC': Oi.SOLAR_PANEL.HOT_CELL, 'SMC': Oi.SOLAR_PANEL.HOT_CELL_CHAIN, 'CCS': Oi.SOLAR_PANEL.SEVERAL_HOT_CELLS, '1PC': Oi.SOLAR_PANEL.HOT_SPOT, 'PCS': Oi.SOLAR_PANEL.SEVERAL_HOT_SPOTS, 'PID': Oi.SOLAR_PANEL.POTENTIAL_INDUCED_DEGRADATION, 'DRT': Oi.SOLAR_PANEL.DIRTY_PANEL, 'BRK': Oi.SOLAR_PANEL.BROKEN_PANEL, 'DSC': Oi.SOLAR_PANEL.DISCONNECTED_PANEL, 'SDW': Oi.SOLAR_PANEL.SHADES, 'SDWD': Oi.SOLAR_PANEL.SHADES_HOT_CELL_CHAIN, 'NDM': Oi.SOLAR_PANEL.NO_DAMAGE}
        self._categories = list(self._mapping.values())
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
                obj.add_category(GenericCategory(self._mapping[parts[i+1]]))
                image.add_object(obj)
        seq.add_image(image)
        return seq


class StanfordCars(Database):
    def __init__(self):
        import images_framework.categories.cars
        super().__init__()
        self._names = ['stanford_cars']
        self._mapping = {0: Oi.VEHICLE.CAR.AMGENERAL_HUMMER_SUV_2000, 1: Oi.VEHICLE.CAR.ACURA_RL_SEDAN_2012, 2: Oi.VEHICLE.CAR.ACURA_TL_SEDAN_2012, 3: Oi.VEHICLE.CAR.ACURA_TL_TYPES_2008, 4: Oi.VEHICLE.CAR.ACURA_TSX_SEDAN_2012, 5: Oi.VEHICLE.CAR.ACURA_INTEGRA_TYPER_2001, 6: Oi.VEHICLE.CAR.ACURA_ZDX_HATCHBACK_2012, 7: Oi.VEHICLE.CAR.ASTONMARTIN_V8VANTAGE_CONVERTIBLE_2012, 8: Oi.VEHICLE.CAR.ASTONMARTIN_V8VANTAGE_COUPE_2012, 9: Oi.VEHICLE.CAR.ASTONMARTIN_VIRAGE_CONVERTIBLE_2012, 10: Oi.VEHICLE.CAR.ASTONMARTIN_VIRAGE_COUPE_2012, 11: Oi.VEHICLE.CAR.AUDI_RS4_CONVERTIBLE_2008,  12: Oi.VEHICLE.CAR.AUDI_A5_COUPE_2012, 13: Oi.VEHICLE.CAR.AUDI_TTS_COUPE_2012, 14: Oi.VEHICLE.CAR.AUDI_R8_COUPE_2012,  15: Oi.VEHICLE.CAR.AUDI_V8_SEDAN_1994, 16: Oi.VEHICLE.CAR.AUDI_100_SEDAN_1994, 17: Oi.VEHICLE.CAR.AUDI_100_WAGON_1994,  18: Oi.VEHICLE.CAR.AUDI_TT_HATCHBACK_2011, 19: Oi.VEHICLE.CAR.AUDI_S6_SEDAN_2011, 20: Oi.VEHICLE.CAR.AUDI_S5_CONVERTIBLE_2012,  21: Oi.VEHICLE.CAR.AUDI_S5_COUPE_2012, 22: Oi.VEHICLE.CAR.AUDI_S4_SEDAN_2012, 23: Oi.VEHICLE.CAR.AUDI_S4_SEDAN_2007,  24: Oi.VEHICLE.CAR.AUDI_TTRS_COUPE_2012, 25: Oi.VEHICLE.CAR.BMW_ACTIVEHYBRID5_SEDAN_2012,  26: Oi.VEHICLE.CAR.BMW_SERIES1_CONVERTIBLE_2012, 27: Oi.VEHICLE.CAR.BMW_SERIES1_COUPE_2012,  28: Oi.VEHICLE.CAR.BMW_SERIES3_SEDAN_2012, 29: Oi.VEHICLE.CAR.BMW_SERIES3_WAGON_2012,  30: Oi.VEHICLE.CAR.BMW_SERIES6_CONVERTIBLE_2007, 31: Oi.VEHICLE.CAR.BMW_X5_SUV_2007, 32: Oi.VEHICLE.CAR.BMW_X6_SUV_2012,  33: Oi.VEHICLE.CAR.BMW_M3_COUPE_2012, 34: Oi.VEHICLE.CAR.BMW_M5_SEDAN_2010, 35: Oi.VEHICLE.CAR.BMW_M6_CONVERTIBLE_2010,  36: Oi.VEHICLE.CAR.BMW_X3_SUV_2012, 37: Oi.VEHICLE.CAR.BMW_Z4_CONVERTIBLE_2012,  38: Oi.VEHICLE.CAR.BENTLEY_CONTINENTAL_CONVERTIBLE_2012, 39: Oi.VEHICLE.CAR.BENTLEY_ARNAGE_SEDAN_2009, 40: Oi.VEHICLE.CAR.BENTLEY_MULSANNE_SEDAN_2011, 41: Oi.VEHICLE.CAR.BENTLEY_CONTINENTAL_COUPE_2012, 42: Oi.VEHICLE.CAR.BENTLEY_CONTINENTAL_COUPE_2007, 43: Oi.VEHICLE.CAR.BENTLEY_CONTINENTAL_SEDAN_2007, 44: Oi.VEHICLE.CAR.BUGATTI_VEYRON_CONVERTIBLE_2009, 45: Oi.VEHICLE.CAR.BUGATTI_VEYRON_COUPE_2009,  46: Oi.VEHICLE.CAR.BUICK_REGAL_GS_2012, 47: Oi.VEHICLE.CAR.BUICK_RAINIER_SUV_2007, 48: Oi.VEHICLE.CAR.BUICK_VERANO_SEDAN_2012,  49: Oi.VEHICLE.CAR.BUICK_ENCLAVE_SUV_2012, 50: Oi.VEHICLE.CAR.CADILLAC_CTSV_SEDAN_2012, 51: Oi.VEHICLE.CAR.CADILLAC_SRX_SUV_2012,  52: Oi.VEHICLE.CAR.CADILLAC_ESCALADE_CREWCAB_2007, 53: Oi.VEHICLE.CAR.CHEVROLET_SILVERADO1500HYBRID_CREWCAB_2012,  54: Oi.VEHICLE.CAR.CHEVROLET_CORVETTE_CONVERTIBLE_2012, 55: Oi.VEHICLE.CAR.CHEVROLET_CORVETTE_ZR1_2012,  56: Oi.VEHICLE.CAR.CHEVROLET_CORVETTE_Z06_2007, 57: Oi.VEHICLE.CAR.CHEVROLET_TRAVERSE_SUV_2012,  58: Oi.VEHICLE.CAR.CHEVROLET_CAMARO_CONVERTIBLE_2012, 59: Oi.VEHICLE.CAR.CHEVROLET_HHR_SS_2010,  60: Oi.VEHICLE.CAR.CHEVROLET_IMPALA_SEDAN_2007, 61: Oi.VEHICLE.CAR.CHEVROLET_TAHOEHYBRID_SUV_2012, 62: Oi.VEHICLE.CAR.CHEVROLET_SONIC_SEDAN_2012, 63: Oi.VEHICLE.CAR.CHEVROLET_EXPRESS_CARGOVAN_2007, 64: Oi.VEHICLE.CAR.CHEVROLET_AVALANCHE_CREWCAB_2012, 65: Oi.VEHICLE.CAR.CHEVROLET_COBALT_SS_2010, 66: Oi.VEHICLE.CAR.CHEVROLET_MALIBUHYBRID_SEDAN_2010, 67: Oi.VEHICLE.CAR.CHEVROLET_TRAINBLAZER_SS_2009, 68: Oi.VEHICLE.CAR.CHEVROLET_SILVERADO2500HD_REGULARCAB_2012,  69: Oi.VEHICLE.CAR.CHEVROLET_SILVERADO1500CLASSIC_EXTENDEDCAB_2007, 70: Oi.VEHICLE.CAR.CHEVROLET_EXPRESS_VAN_2007,  71: Oi.VEHICLE.CAR.CHEVROLET_MONTECARLO_COUPE_2007, 72: Oi.VEHICLE.CAR.CHEVROLET_MALIBU_SEDAN_2007,  73: Oi.VEHICLE.CAR.CHEVROLET_SILVERADO1500_EXTENDEDCAB_2012, 74: Oi.VEHICLE.CAR.CHEVROLET_SILVERADO1500_REGULARCAB_2012,  75: Oi.VEHICLE.CAR.CHRYSLER_ASPEN_SUV_2009, 76: Oi.VEHICLE.CAR.CHRYSLER_SEBRING_CONVERTIBLE_2010,  77: Oi.VEHICLE.CAR.CHRYSLER_TOWN_MINIVAN_2012, 78: Oi.VEHICLE.CAR.CHRYSLER_300_STR8_2010,  79: Oi.VEHICLE.CAR.CHRYSLER_CROSSFIRE_CONVERTIBLE_2008, 80: Oi.VEHICLE.CAR.CHRYSLER_PTCRUISER_CONVERTIBLE_2008,  81: Oi.VEHICLE.CAR.DAEWOO_NUBIRA_WAGON_2002, 82: Oi.VEHICLE.CAR.DODGE_CALIBER_WAGON_2012,  83: Oi.VEHICLE.CAR.DODGE_CALIBER_WAGON_2007, 84: Oi.VEHICLE.CAR.DODGE_CARAVAN_MINIVAN_1997,  85: Oi.VEHICLE.CAR.DODGE_RAM_CREWCAB_2010, 86: Oi.VEHICLE.CAR.DODGE_RAM_QUADCAB_2009,  87: Oi.VEHICLE.CAR.DODGE_SPRINTER_CARGOVAN_2009, 88: Oi.VEHICLE.CAR.DODGE_JOURNEY_SUV_2012,  89: Oi.VEHICLE.CAR.DODGE_DAKOTA_CREWCAB_2010, 90: Oi.VEHICLE.CAR.DODGE_DAKOTA_CLUBCAB_2007, 91: Oi.VEHICLE.CAR.DODGE_MAGNUM_WAGON_2008, 92: Oi.VEHICLE.CAR.DODGE_CHALLENGER_SRT8_2011,  93: Oi.VEHICLE.CAR.DODGE_DURANGO_SUV_2012, 94: Oi.VEHICLE.CAR.DODGE_DURANGO_SUV_2007,  95: Oi.VEHICLE.CAR.DODGE_CHARGER_SEDAN_2012, 96: Oi.VEHICLE.CAR.DODGE_CHARGER_SRT8_2009, 97: Oi.VEHICLE.CAR.EAGLE_TALON_HATCHBACK_1998, 98: Oi.VEHICLE.CAR.FIAT_500_ABARTH_2012,  99: Oi.VEHICLE.CAR.FIAT_500_CONVERTIBLE_2012, 100: Oi.VEHICLE.CAR.FERRARI_FF_COUPE_2012,  101: Oi.VEHICLE.CAR.FERRARI_CALIFORNIA_CONVERTIBLE_2012, 102: Oi.VEHICLE.CAR.FERRARI_ITALIA_CONVERTIBLE_2012,  103: Oi.VEHICLE.CAR.FERRARI_ITALIA_COUPE_2012, 104: Oi.VEHICLE.CAR.FISKER_KARMA_SEDAN_2012,  105: Oi.VEHICLE.CAR.FORD_F450_CREWCAB_2012, 106: Oi.VEHICLE.CAR.FORD_MUSTANG_CONVERTIBLE_2007, 107: Oi.VEHICLE.CAR.FORD_FREESTAR_MINIVAN_2007, 108: Oi.VEHICLE.CAR.FORD_EXPEDITION_SUV_2009,  109: Oi.VEHICLE.CAR.FORD_EDGE_SUV_2012, 110: Oi.VEHICLE.CAR.FORD_RANGER_SUPERCAB_2011, 111: Oi.VEHICLE.CAR.FORD_GT_COUPE_2006,  112: Oi.VEHICLE.CAR.FORD_F150_REGULARCAB_2012, 113: Oi.VEHICLE.CAR.FORD_F150_REGULARCAB_2007,  114: Oi.VEHICLE.CAR.FORD_FOCUS_SEDAN_2007, 115: Oi.VEHICLE.CAR.FORD_ESERIES_WAGON_2012, 116: Oi.VEHICLE.CAR.FORD_FIESTA_SEDAN_2012, 117: Oi.VEHICLE.CAR.GMC_TERRAIN_SUV_2012, 118: Oi.VEHICLE.CAR.GMC_SAVANA_VAN_2012, 119: Oi.VEHICLE.CAR.GMC_YUKONHYBRID_SUV_2012, 120: Oi.VEHICLE.CAR.GMC_ACADIA_SUV_2012,  121: Oi.VEHICLE.CAR.GMC_CANYON_EXTENDEDCAB_2012, 122: Oi.VEHICLE.CAR.GMC_METRO_CONVERTIBLE_1993,  123: Oi.VEHICLE.CAR.HUMMER_H3T_CREWCAB_2010, 124: Oi.VEHICLE.CAR.HUMMER_H2SUT_CREWCAB_2009,  125: Oi.VEHICLE.CAR.HONDA_ODYSSEY_MINIVAN_2012, 126: Oi.VEHICLE.CAR.HONDA_ODYSSEY_MINIVAN_2007,  127: Oi.VEHICLE.CAR.HONDA_ACCORD_COUPE_2012, 128: Oi.VEHICLE.CAR.HONDA_ACCORD_SEDAN_2012,  129: Oi.VEHICLE.CAR.HYUNDAI_VELOSTER_HATCHBACK_2012, 130: Oi.VEHICLE.CAR.HYUNDAI_SANTAFE_SUV_2012, 131: Oi.VEHICLE.CAR.HYUNDAI_TUCSON_SUV_2012, 132: Oi.VEHICLE.CAR.HYUNDAI_VERACRUZ_SUV_2012, 133: Oi.VEHICLE.CAR.HYUNDAI_SONATAHYBRID_SEDAN_2012, 134: Oi.VEHICLE.CAR.HYUNDAI_ELANTRA_SEDAN_2007, 135: Oi.VEHICLE.CAR.HYUNDAI_ACCENT_SEDAN_2012, 136: Oi.VEHICLE.CAR.HYUNDAI_GENESIS_SEDAN_2012, 137: Oi.VEHICLE.CAR.HYUNDAI_SONATA_SEDAN_2012, 138: Oi.VEHICLE.CAR.HYUNDAI_ELANTRA_HATCHBACK_2012, 139: Oi.VEHICLE.CAR.HYUNDAI_AZERA_SEDAN_2012, 140: Oi.VEHICLE.CAR.INFINITI_G_COUPE_2012, 141: Oi.VEHICLE.CAR.INFINITI_QX56_SUV_2011, 142: Oi.VEHICLE.CAR.ISUZU_ASCENDER_SUV_2008, 143: Oi.VEHICLE.CAR.JAGUAR_XK_XKR_2012, 144: Oi.VEHICLE.CAR.JEEP_PATRIOT_SUV_2012, 145: Oi.VEHICLE.CAR.JEEP_WRANGLER_SUV_2012, 146: Oi.VEHICLE.CAR.JEEP_LIBERTY_SUV_2012, 147: Oi.VEHICLE.CAR.JEEP_GRANDCHEROKEE_SUV_2012, 148: Oi.VEHICLE.CAR.JEEP_COMPASS_SUV_2012, 149: Oi.VEHICLE.CAR.LAMBORGHINI_REVENTON_COUPE_2012, 150: Oi.VEHICLE.CAR.LAMBORGHINI_AVENTADOR_COUPE_2012, 151: Oi.VEHICLE.CAR.LAMBORGHINI_GALLARDO_SUPERLEGGERA_2012, 152: Oi.VEHICLE.CAR.LAMBORGHINI_DIABLO_COUPE_2001, 153: Oi.VEHICLE.CAR.LANDROVER_RANGEROVER_SUV_2012, 154: Oi.VEHICLE.CAR.LANDROVER_LR2_SUV_2012, 155: Oi.VEHICLE.CAR.LINCOLN_TOWN_SEDAN_2011, 156: Oi.VEHICLE.CAR.MINI_COOPER_CONVERTIBLE_2012, 157: Oi.VEHICLE.CAR.MAYBACH_LANDAULET_CONVERTIBLE_2012, 158: Oi.VEHICLE.CAR.MAZDA_TRIBUTE_SUV_2011, 159: Oi.VEHICLE.CAR.MCLAREN_MP4_COUPE_2012, 160: Oi.VEHICLE.CAR.MERCEDES_300_CONVERTIBLE_1993, 161: Oi.VEHICLE.CAR.MERCEDES_C_SEDAN_2012, 162: Oi.VEHICLE.CAR.MERCEDES_SL_COUPE_2009, 163: Oi.VEHICLE.CAR.MERCEDES_E_SEDAN_2012, 164: Oi.VEHICLE.CAR.MERCEDES_S_SEDAN_2012, 165: Oi.VEHICLE.CAR.MERCEDES_SPRINTER_VAN_2012, 166: Oi.VEHICLE.CAR.MITSUBISHI_LANCER_SEDAN_2012, 167: Oi.VEHICLE.CAR.NISSAN_LEAF_HATCHBACK_2012, 168: Oi.VEHICLE.CAR.NISSAN_NV_VAN_2012, 169: Oi.VEHICLE.CAR.NISSAN_JUKE_HATCHBACK_2012, 170: Oi.VEHICLE.CAR.NISSAN_240SX_COUPE_1998, 171: Oi.VEHICLE.CAR.PLYMOUTH_NEON_COUPE_1999, 172: Oi.VEHICLE.CAR.PORSCHE_PANAMERA_SEDAN_2012, 173: Oi.VEHICLE.CAR.RAM_CV_MINIVAN_2012, 174: Oi.VEHICLE.CAR.ROLLSROYCE_PHANTOM_COUPE_2012, 175: Oi.VEHICLE.CAR.ROLLSROYCE_GHOST_SEDAN_2012, 176: Oi.VEHICLE.CAR.ROLLSROYCE_PHANTOM_SEDAN_2012, 177: Oi.VEHICLE.CAR.SCION_XD_HATCHBACK_2012, 178: Oi.VEHICLE.CAR.SPYKER_C8_CONVERTIBLE_2009, 179: Oi.VEHICLE.CAR.SPYKER_C8_COUPE_2009, 180: Oi.VEHICLE.CAR.SUZUKI_AERIO_SEDAN_2007, 181: Oi.VEHICLE.CAR.SUZUKI_KIZASHI_SEDAN_2012, 182: Oi.VEHICLE.CAR.SUZUKI_SX4_HATCHBACK_2012, 183: Oi.VEHICLE.CAR.SUZUKI_SX4_SEDAN_2012, 184: Oi.VEHICLE.CAR.TESLA_MODELS_SEDAN_2012, 185: Oi.VEHICLE.CAR.TOYOTA_SEQUOIA_SUV_2012, 186: Oi.VEHICLE.CAR.TOYOTA_CAMRY_SEDAN_2012, 187: Oi.VEHICLE.CAR.TOYOTA_COROLLA_SEDAN_2012, 188: Oi.VEHICLE.CAR.TOYOTA_4RUNNER_SUV_2012, 189: Oi.VEHICLE.CAR.VOLKSWAGEN_GOLF_HATCHBACK_2012, 190: Oi.VEHICLE.CAR.VOLKSWAGEN_GOLF_HATCHBACK_1991, 191: Oi.VEHICLE.CAR.VOLKSWAGEN_BEETLE_HATCHBACK_2012, 192: Oi.VEHICLE.CAR.VOLV0_C30_HATCHBACK_2012, 193: Oi.VEHICLE.CAR.VOLV0_240_SEDAN_1993, 194: Oi.VEHICLE.CAR.VOLVO_XC90_SUV_2007, 195: Oi.VEHICLE.CAR.SMART_FORTWO_CONVERTIBLE_2012}
        self._categories = list(self._mapping.values())
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
        obj.add_category(GenericCategory(self._mapping[int(parts[5])]))
        image.add_object(obj)
        seq.add_image(image)
        return seq


class WorldView3(Database):
    def __init__(self):
        super().__init__()
        self._names = ['maxar', 'cuende', 'all']
        self._categories = [Oi.SMALL_VEHICLE]
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

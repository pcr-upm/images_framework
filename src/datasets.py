#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@geoaitech.com'

import os
import abc
import json
import rasterio
import numpy as np
from PIL import Image
from shapely import wkt
from satellite_framework.src.utils import load_geoimage, geometry2numpy
from satellite_framework.src.annotations import SatelliteSequence, SatelliteImage, SatelliteObject
from satellite_framework.src.categories import Name, ObjInstance as Oi


def get_palette(n):
    """
    Returns the default color map for visualizing the segmentation task.
    """
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return list(zip(*[iter(palette)]*3))


class Database(abc.ABC):
    """
    Declare a common interface for the different data sets related to satellite imagery.
    """
    def __init__(self):
        self._names = ''
        self._categories = []
        self._colors = []

    @abc.abstractmethod
    def load_filename(self, path, db, line):
        pass

    def get_names(self):
        return self._names

    def get_categories(self):
        return self._categories

    def get_colors(self):
        return self._colors


class XView(Database):
    def __init__(self):
        super().__init__()
        self._names = ['xview']
        self._mapping = {11: Oi.FIXED_WING_AIRCRAFT, 12: Oi.FIXED_WING_AIRCRAFT.SMALL_AIRCRAFT, 13: Oi.FIXED_WING_AIRCRAFT.CARGO_PLANE, 15: Oi.HELICOPTER, 17: Oi.PASSENGER_VEHICLE, 18: Oi.PASSENGER_VEHICLE.SMALL_CAR, 19: Oi.PASSENGER_VEHICLE.BUS, 20: Oi.TRUCK.PICKUP_TRUCK, 21: Oi.TRUCK.UTILITY_TRUCK, 23: Oi.TRUCK, 24: Oi.TRUCK.CARGO_TRUCK, 25: Oi.TRUCK.TRUCK_BOX, 26: Oi.TRUCK.TRUCK_TRACTOR, 27: Oi.TRUCK.TRAILER, 28: Oi.TRUCK.TRUCK_FLATBED, 29: Oi.TRUCK.TRUCK_LIQUID, 32: Oi.ENGINEERING_VEHICLE.CRANE_TRUCK, 33: Oi.RAILWAY_VEHICLE, 34: Oi.RAILWAY_VEHICLE.PASSENGER_CAR, 35: Oi.RAILWAY_VEHICLE.CARGO_CAR, 36: Oi.RAILWAY_VEHICLE.FLAT_CAR, 37: Oi.RAILWAY_VEHICLE.TANK_CAR, 38: Oi.RAILWAY_VEHICLE.LOCOMOTIVE, 40: Oi.MARITIME_VESSEL, 41: Oi.MARITIME_VESSEL.MOTOBOAT, 42: Oi.MARITIME_VESSEL.SAILBOAT, 44: Oi.MARITIME_VESSEL.TUGBOAT, 45: Oi.MARITIME_VESSEL.BARGE, 47: Oi.MARITIME_VESSEL.FISHING_VESSEL, 49: Oi.MARITIME_VESSEL.FERRY, 50: Oi.MARITIME_VESSEL.YATCH, 51: Oi.MARITIME_VESSEL.CONTAINER_SHIP, 52: Oi.MARITIME_VESSEL.OIL_TANKER, 53: Oi.ENGINEERING_VEHICLE, 54: Oi.ENGINEERING_VEHICLE.TOWER_CRANE, 55: Oi.ENGINEERING_VEHICLE.CONTAINER_CRANE, 56: Oi.ENGINEERING_VEHICLE.REACH_STACKER, 57: Oi.ENGINEERING_VEHICLE.STRADDLE_CARRIER, 59: Oi.ENGINEERING_VEHICLE.MOBILE_CRANE, 60: Oi.ENGINEERING_VEHICLE.DUMP_TRUCK, 61: Oi.ENGINEERING_VEHICLE.HAUL_TRUCK, 62: Oi.ENGINEERING_VEHICLE.SCRAPER_TRACTOR, 63: Oi.ENGINEERING_VEHICLE.FRONT_LOADER, 64: Oi.ENGINEERING_VEHICLE.EXCAVATOR, 65: Oi.ENGINEERING_VEHICLE.CEMENT_MIXER, 66: Oi.ENGINEERING_VEHICLE.GROUND_GRADER, 71: Oi.BUILDING.HUT_TENT, 72: Oi.BUILDING.SHED, 73: Oi.BUILDING, 74: Oi.BUILDING.AIRCRAFT_HANGAR, 76: Oi.BUILDING.DAMAGED_BUILDING, 77: Oi.BUILDING.FACILITY, 79: Oi.CONSTRUCTION_SITE, 83: Oi.VEHICLE_LOT, 84: Oi.HELIPAD, 86: Oi.STORAGE_TANK, 89: Oi.SHIPPING_CONTAINER_LOT, 91: Oi.SHIPPING_CONTAINER, 93: Oi.PYLON, 94: Oi.TOWER_STRUCTURE}
        self._categories = list(self._mapping.values())
        self._colors = get_palette(len(self._categories))

    def load_filename(self, path, db, line):
        seq = SatelliteSequence()
        parts = line.strip().split(';')
        image = SatelliteImage(path + parts[0])
        num_predictions = int(parts[1])
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        image.gsd = 0.3
        for idx in range(0, num_predictions):
            obj = SatelliteObject()
            obj.id = int(parts[(3*idx)+2])
            pts = parts[(3*idx)+3].split(',')
            obj.bb = (int(pts[0]), int(pts[1]), int(pts[2]), int(pts[3]))
            cat = int(parts[(3*idx)+4])
            if cat not in self._mapping.keys():
                continue
            obj.categories.add(self._mapping[cat])
            # if self._mapping[cat] is not Oi.PASSENGER_VEHICLE.SMALL_CAR:
            #     continue
            image.add_object(obj)
        if len(image.objects) > 0:
            seq.add_image(image)
        return seq


class XView2(Database):
    def __init__(self):
        super().__init__()
        self._names = ['xview2']
        self._mapping = {'building': Oi.BUILDING, 'un-classified': Oi.BUILDING.UNCLASSIFIED, 'no-damage': Oi.BUILDING.NO_DAMAGE, 'minor-damage': Oi.BUILDING.MINOR_DAMAGE, 'major-damage': Oi.BUILDING.MAJOR_DAMAGE, 'destroyed': Oi.BUILDING.DESTROYED}
        self._categories = list(self._mapping.values())
        self._colors = get_palette(len(self._categories))

    def load_filename(self, path, db, line):
        seq = SatelliteSequence()
        for time in ['pre_', 'post_']:
            filepath = line.strip()
            pos = filepath.find('pre_')
            image = SatelliteImage(path + filepath[:pos] + time + filepath[pos+4:])
            pos = filepath.find('/') + 1
            mid = filepath[:pos]
            end = filepath[pos:]
            json_file = path + mid + 'labels' + end[10:-16] + time + 'disaster.json'
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
                obj = SatelliteObject()
                obj.id = feat['properties']['uid']
                obj.bb = (int(geom.bounds[0]), int(geom.bounds[1]), int(geom.bounds[2]), int(geom.bounds[3]))
                obj.multipolygon = [contour for contour in geometry2numpy(geom)]
                if json_file.find('post_disaster') > -1:
                    obj.categories.add(self._mapping[feat['properties']['subtype']])
                else:
                    obj.categories.add(self._mapping[feat['properties']['feature_type']])
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
        seq = SatelliteSequence()
        filepath = line.strip()
        image = SatelliteImage(path + filepath)
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
            obj = SatelliteObject()
            obj.bb = (float(elem_hbb[0]), float(elem_hbb[1]), float(elem_hbb[4]), float(elem_hbb[5]))
            obj.obb = (float(elem[0]), float(elem[1]), float(elem[2]), float(elem[3]), float(elem[4]), float(elem[5]), float(elem[6]), float(elem[7]))
            obj.categories.add(self._mapping[elem[8]])
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
        seq = SatelliteSequence()
        parts = line.strip().split(',')
        image = SatelliteImage(path + parts[0])
        num_vehicles = int(parts[1])
        src_raster = rasterio.open(image.filename, 'r')
        width = src_raster.width
        height = src_raster.height
        image.tile = np.array([0, 0, width, height])
        image.gsd = 0.15
        for idx in range(0, num_vehicles):
            obj = SatelliteObject()
            center = [int(parts[(3*idx)+2]), int(parts[(3*idx)+3])]
            # Bounding boxes were fixed at size 48 pixels which is the maximum length of a car
            obj.bb = (center[0]-12, center[1]-12, center[0]+12, center[1]+12)
            obj.obb = (obj.bb[0], obj.bb[1], obj.bb[2], obj.bb[1], obj.bb[2], obj.bb[3], obj.bb[0], obj.bb[3])
            obj.categories.add(self._categories[0])
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
        seq = SatelliteSequence()
        parts = line.strip().split(';')
        image = SatelliteImage(path + parts[0])
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
            obj = SatelliteObject()
            obj.bb = (int(pts[0]), int(pts[1]), int(pts[2]), int(pts[3]))
            obj.categories.add(self._categories[0])
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
        seq = SatelliteSequence()
        parts = line.strip().split(';')
        image = SatelliteImage(path + parts[0])
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
                obj = SatelliteObject()
                obj.id = int(elems[0])
                center = (int(elems[2]), int(elems[3]))
                obj.bb = (center[0]-int(elems[4]), center[1]-int(elems[5]), center[0]+int(elems[4]), center[1]+int(elems[5]))
                angle = np.radians(float(elems[6]))
                pts = np.array(((obj.bb[0]-center[0], obj.bb[2]-center[0], obj.bb[2]-center[0], obj.bb[0]-center[0]), (obj.bb[1]-center[1], obj.bb[1]-center[1], obj.bb[3]-center[1], obj.bb[3]-center[1])))
                rot = np.array(((np.cos(angle), np.sin(angle)), (-np.sin(angle), np.cos(angle))))
                pts_proj = np.matmul(rot, pts)
                obj.obb = (pts_proj[0, 0]+center[0], pts_proj[1, 0]+center[1], pts_proj[0, 1]+center[0], pts_proj[1, 1]+center[1], pts_proj[0, 2]+center[0], pts_proj[1, 2]+center[1], pts_proj[0, 3]+center[0], pts_proj[1, 3]+center[1])
                obj.categories.add(self._mapping[category])
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
        seq = SatelliteSequence()
        parts = line.strip().split(';')
        image = SatelliteImage(path + parts[0])
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
            obj = SatelliteObject()
            obj.bb = (int(elem[0]), int(elem[1]), int(elem[2]), int(elem[3]))
            obj.categories.add(self._mapping[int(elem[4])])
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
        seq = SatelliteSequence()
        parts = line.strip().split(';')
        filepath = parts[0]
        image = SatelliteImage(path + filepath)
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        if len(parts) > 1:
            for i in range(1, len(parts), 2):
                geom = wkt.loads(parts[i+1])
                if geom.is_empty:
                    continue
                obj = SatelliteObject()
                obj.id = str(parts[i])
                obj.bb = (int(geom.bounds[0]), int(geom.bounds[1]), int(geom.bounds[2]), int(geom.bounds[3]))
                obj.multipolygon = [contour for contour in geometry2numpy(geom)]
                obj.categories.add(self._categories[0])
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
        from satellite_framework.src.utils import mask2contours
        label_mapping = {0: -1, 1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1, 7: 0, 8: 1, 9: -1, 10: -1, 11: 2, 12: 3, 13: 4, 14: -1, 15: -1, 16: -1, 17: 5, 18: -1, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: -1, 30: -1, 31: 16, 32: 17, 33: 18}
        seq = SatelliteSequence()
        parts = line.strip().split('\t')
        filepath = parts[0]
        image = SatelliteImage(path + filepath)
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        if len(parts) > 1:
            aux_filepath = parts[1]
            aux_image = SatelliteImage(path + aux_filepath)
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
                obj = SatelliteObject()
                bbox = cv2.boundingRect(contours[index])
                obj.bb = (bbox[0], bbox[1], bbox[2]+bbox[0], bbox[3]+bbox[1])
                obj.multipolygon = [contours[index]]
                obj.categories.add(Name(labels[index]))
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
        from satellite_framework.src.utils import mask2contours
        seq = SatelliteSequence()
        parts = line.strip().split(' ')
        filepath = parts[0]
        image = SatelliteImage(path + filepath)
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        if len(parts) > 1:
            aux_filepath = parts[1]
            aux_image = SatelliteImage(path + aux_filepath)
            img, _ = load_geoimage(aux_image.filename)
            categories = list(np.unique(img))
            contours, labels = [], []
            for category in categories:
                mask = np.where((img == category), 255, 0).astype(np.uint8)
                for contour in mask2contours(mask):
                    contours.append(contour)
                    labels.append(str(category))
            for index in range(len(contours)):
                obj = SatelliteObject()
                bbox = cv2.boundingRect(contours[index])
                obj.bb = (bbox[0], bbox[1], bbox[2]+bbox[0], bbox[3]+bbox[1])
                obj.multipolygon = [contours[index]]
                obj.categories.add(Name(labels[index]))
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
        seq = SatelliteSequence()
        parts = line.strip().split(';')
        filepath = parts[0]
        image = SatelliteImage(path + filepath)
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        if len(parts) > 1:
            for i in range(1, len(parts), 2):
                geom = wkt.loads(parts[i])
                if geom.is_empty:
                    continue
                obj = SatelliteObject()
                obj.bb = (int(geom.bounds[0]), int(geom.bounds[1]), int(geom.bounds[2]), int(geom.bounds[3]))
                obj.multipolygon = [contour for contour in geometry2numpy(geom)]
                obj.categories.add(self._mapping[parts[i+1]])
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
        seq = SatelliteSequence()
        parts = line.strip().split(';')
        filepath = parts[0]
        image = SatelliteImage(path + filepath)
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        if len(parts) > 1:
            for i in range(1, len(parts)):
                geom = wkt.loads(parts[i])
                if geom.is_empty:
                    continue
                obj = SatelliteObject()
                obj.bb = (int(geom.bounds[0]), int(geom.bounds[1]), int(geom.bounds[2]), int(geom.bounds[3]))
                obj.multipolygon = [contour for contour in geometry2numpy(geom)]
                obj.categories.add(self._categories[0])
                image.add_object(obj)
        seq.add_image(image)
        return seq


class RecGeoAIPanels(Database):
    def __init__(self):
        super().__init__()
        self._names = ['rec_geoai_panels']
        self._mapping = {'1CC': Oi.SOLAR_PANEL.HOT_CELL, 'SMC': Oi.SOLAR_PANEL.HOT_CELL_CHAIN, 'CCS': Oi.SOLAR_PANEL.SEVERAL_HOT_CELLS, '1PC': Oi.SOLAR_PANEL.HOT_SPOT, 'PCS': Oi.SOLAR_PANEL.SEVERAL_HOT_SPOTS, 'PID': Oi.SOLAR_PANEL.POTENTIAL_INDUCED_DEGRADATION, 'DRT': Oi.SOLAR_PANEL.DIRTY_PANEL, 'BRK': Oi.SOLAR_PANEL.BROKEN_PANEL, 'DSC': Oi.SOLAR_PANEL.DISCONNECTED_PANEL, 'SDW': Oi.SOLAR_PANEL.SHADES, 'SDWD': Oi.SOLAR_PANEL.SHADES_HOT_CELL_CHAIN, 'NDM': Oi.SOLAR_PANEL.NO_DAMAGE}
        self._categories = list(self._mapping.values())
        self._colors = get_palette(len(self._categories))

    def load_filename(self, path, db, line):
        seq = SatelliteSequence()
        parts = line.strip().split(';')
        filepath = parts[0]
        image = SatelliteImage(path + filepath)
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        if len(parts) > 1:
            for i in range(1, len(parts), 2):
                geom = wkt.loads(parts[i])
                if geom.is_empty:
                    continue
                obj = SatelliteObject()
                obj.bb = (int(geom.bounds[0]), int(geom.bounds[1]), int(geom.bounds[2]), int(geom.bounds[3]))
                obj.multipolygon = [contour for contour in geometry2numpy(geom)]
                obj.categories.add(self._mapping[parts[i+1]])
                image.add_object(obj)
        seq.add_image(image)
        return seq


class StanfordCars(Database):
    def __init__(self):
        super().__init__()
        self._names = ['stanford_cars']
        self._mapping = {0: Oi.CAR.AMGENERAL_HUMMER_SUV_2000, 1: Oi.CAR.ACURA_RL_SEDAN_2012, 2: Oi.CAR.ACURA_TL_SEDAN_2012, 3: Oi.CAR.ACURA_TL_TYPES_2008, 4: Oi.CAR.ACURA_TSX_SEDAN_2012, 5: Oi.CAR.ACURA_INTEGRA_TYPER_2001, 6: Oi.CAR.ACURA_ZDX_HATCHBACK_2012, 7: Oi.CAR.ASTONMARTIN_V8VANTAGE_CONVERTIBLE_2012, 8: Oi.CAR.ASTONMARTIN_V8VANTAGE_COUPE_2012, 9: Oi.CAR.ASTONMARTIN_VIRAGE_CONVERTIBLE_2012, 10: Oi.CAR.ASTONMARTIN_VIRAGE_COUPE_2012, 11: Oi.CAR.AUDI_RS4_CONVERTIBLE_2008,  12: Oi.CAR.AUDI_A5_COUPE_2012, 13: Oi.CAR.AUDI_TTS_COUPE_2012, 14: Oi.CAR.AUDI_R8_COUPE_2012,  15: Oi.CAR.AUDI_V8_SEDAN_1994, 16: Oi.CAR.AUDI_100_SEDAN_1994, 17: Oi.CAR.AUDI_100_WAGON_1994,  18: Oi.CAR.AUDI_TT_HATCHBACK_2011, 19: Oi.CAR.AUDI_S6_SEDAN_2011, 20: Oi.CAR.AUDI_S5_CONVERTIBLE_2012,  21: Oi.CAR.AUDI_S5_COUPE_2012, 22: Oi.CAR.AUDI_S4_SEDAN_2012, 23: Oi.CAR.AUDI_S4_SEDAN_2007,  24: Oi.CAR.AUDI_TTRS_COUPE_2012, 25: Oi.CAR.BMW_ACTIVEHYBRID5_SEDAN_2012,  26: Oi.CAR.BMW_SERIES1_CONVERTIBLE_2012, 27: Oi.CAR.BMW_SERIES1_COUPE_2012,  28: Oi.CAR.BMW_SERIES3_SEDAN_2012, 29: Oi.CAR.BMW_SERIES3_WAGON_2012,  30: Oi.CAR.BMW_SERIES6_CONVERTIBLE_2007, 31: Oi.CAR.BMW_X5_SUV_2007, 32: Oi.CAR.BMW_X6_SUV_2012,  33: Oi.CAR.BMW_M3_COUPE_2012, 34: Oi.CAR.BMW_M5_SEDAN_2010, 35: Oi.CAR.BMW_M6_CONVERTIBLE_2010,  36: Oi.CAR.BMW_X3_SUV_2012, 37: Oi.CAR.BMW_Z4_CONVERTIBLE_2012,  38: Oi.CAR.BENTLEY_CONTINENTAL_CONVERTIBLE_2012, 39: Oi.CAR.BENTLEY_ARNAGE_SEDAN_2009, 40: Oi.CAR.BENTLEY_MULSANNE_SEDAN_2011, 41: Oi.CAR.BENTLEY_CONTINENTAL_COUPE_2012, 42: Oi.CAR.BENTLEY_CONTINENTAL_COUPE_2007, 43: Oi.CAR.BENTLEY_CONTINENTAL_SEDAN_2007, 44: Oi.CAR.BUGATTI_VEYRON_CONVERTIBLE_2009, 45: Oi.CAR.BUGATTI_VEYRON_COUPE_2009,  46: Oi.CAR.BUICK_REGAL_GS_2012, 47: Oi.CAR.BUICK_RAINIER_SUV_2007, 48: Oi.CAR.BUICK_VERANO_SEDAN_2012,  49: Oi.CAR.BUICK_ENCLAVE_SUV_2012, 50: Oi.CAR.CADILLAC_CTSV_SEDAN_2012, 51: Oi.CAR.CADILLAC_SRX_SUV_2012,  52: Oi.CAR.CADILLAC_ESCALADE_CREWCAB_2007, 53: Oi.CAR.CHEVROLET_SILVERADO1500HYBRID_CREWCAB_2012,  54: Oi.CAR.CHEVROLET_CORVETTE_CONVERTIBLE_2012, 55: Oi.CAR.CHEVROLET_CORVETTE_ZR1_2012,  56: Oi.CAR.CHEVROLET_CORVETTE_Z06_2007, 57: Oi.CAR.CHEVROLET_TRAVERSE_SUV_2012,  58: Oi.CAR.CHEVROLET_CAMARO_CONVERTIBLE_2012, 59: Oi.CAR.CHEVROLET_HHR_SS_2010,  60: Oi.CAR.CHEVROLET_IMPALA_SEDAN_2007, 61: Oi.CAR.CHEVROLET_TAHOEHYBRID_SUV_2012, 62: Oi.CAR.CHEVROLET_SONIC_SEDAN_2012, 63: Oi.CAR.CHEVROLET_EXPRESS_CARGOVAN_2007, 64: Oi.CAR.CHEVROLET_AVALANCHE_CREWCAB_2012, 65: Oi.CAR.CHEVROLET_COBALT_SS_2010, 66: Oi.CAR.CHEVROLET_MALIBUHYBRID_SEDAN_2010, 67: Oi.CAR.CHEVROLET_TRAINBLAZER_SS_2009, 68: Oi.CAR.CHEVROLET_SILVERADO2500HD_REGULARCAB_2012,  69: Oi.CAR.CHEVROLET_SILVERADO1500CLASSIC_EXTENDEDCAB_2007, 70: Oi.CAR.CHEVROLET_EXPRESS_VAN_2007,  71: Oi.CAR.CHEVROLET_MONTECARLO_COUPE_2007, 72: Oi.CAR.CHEVROLET_MALIBU_SEDAN_2007,  73: Oi.CAR.CHEVROLET_SILVERADO1500_EXTENDEDCAB_2012, 74: Oi.CAR.CHEVROLET_SILVERADO1500_REGULARCAB_2012,  75: Oi.CAR.CHRYSLER_ASPEN_SUV_2009, 76: Oi.CAR.CHRYSLER_SEBRING_CONVERTIBLE_2010,  77: Oi.CAR.CHRYSLER_TOWN_MINIVAN_2012, 78: Oi.CAR.CHRYSLER_300_STR8_2010,  79: Oi.CAR.CHRYSLER_CROSSFIRE_CONVERTIBLE_2008, 80: Oi.CAR.CHRYSLER_PTCRUISER_CONVERTIBLE_2008,  81: Oi.CAR.DAEWOO_NUBIRA_WAGON_2002, 82: Oi.CAR.DODGE_CALIBER_WAGON_2012,  83: Oi.CAR.DODGE_CALIBER_WAGON_2007, 84: Oi.CAR.DODGE_CARAVAN_MINIVAN_1997,  85: Oi.CAR.DODGE_RAM_CREWCAB_2010, 86: Oi.CAR.DODGE_RAM_QUADCAB_2009,  87: Oi.CAR.DODGE_SPRINTER_CARGOVAN_2009, 88: Oi.CAR.DODGE_JOURNEY_SUV_2012,  89: Oi.CAR.DODGE_DAKOTA_CREWCAB_2010, 90: Oi.CAR.DODGE_DAKOTA_CLUBCAB_2007, 91: Oi.CAR.DODGE_MAGNUM_WAGON_2008, 92: Oi.CAR.DODGE_CHALLENGER_SRT8_2011,  93: Oi.CAR.DODGE_DURANGO_SUV_2012, 94: Oi.CAR.DODGE_DURANGO_SUV_2007,  95: Oi.CAR.DODGE_CHARGER_SEDAN_2012, 96: Oi.CAR.DODGE_CHARGER_SRT8_2009, 97: Oi.CAR.EAGLE_TALON_HATCHBACK_1998, 98: Oi.CAR.FIAT_500_ABARTH_2012,  99: Oi.CAR.FIAT_500_CONVERTIBLE_2012, 100: Oi.CAR.FERRARI_FF_COUPE_2012,  101: Oi.CAR.FERRARI_CALIFORNIA_CONVERTIBLE_2012, 102: Oi.CAR.FERRARI_ITALIA_CONVERTIBLE_2012,  103: Oi.CAR.FERRARI_ITALIA_COUPE_2012, 104: Oi.CAR.FISKER_KARMA_SEDAN_2012,  105: Oi.CAR.FORD_F450_CREWCAB_2012, 106: Oi.CAR.FORD_MUSTANG_CONVERTIBLE_2007, 107: Oi.CAR.FORD_FREESTAR_MINIVAN_2007, 108: Oi.CAR.FORD_EXPEDITION_SUV_2009,  109: Oi.CAR.FORD_EDGE_SUV_2012, 110: Oi.CAR.FORD_RANGER_SUPERCAB_2011, 111: Oi.CAR.FORD_GT_COUPE_2006,  112: Oi.CAR.FORD_F150_REGULARCAB_2012, 113: Oi.CAR.FORD_F150_REGULARCAB_2007,  114: Oi.CAR.FORD_FOCUS_SEDAN_2007, 115: Oi.CAR.FORD_ESERIES_WAGON_2012, 116: Oi.CAR.FORD_FIESTA_SEDAN_2012, 117: Oi.CAR.GMC_TERRAIN_SUV_2012, 118: Oi.CAR.GMC_SAVANA_VAN_2012, 119: Oi.CAR.GMC_YUKONHYBRID_SUV_2012, 120: Oi.CAR.GMC_ACADIA_SUV_2012,  121: Oi.CAR.GMC_CANYON_EXTENDEDCAB_2012, 122: Oi.CAR.GMC_METRO_CONVERTIBLE_1993,  123: Oi.CAR.HUMMER_H3T_CREWCAB_2010, 124: Oi.CAR.HUMMER_H2SUT_CREWCAB_2009,  125: Oi.CAR.HONDA_ODYSSEY_MINIVAN_2012, 126: Oi.CAR.HONDA_ODYSSEY_MINIVAN_2007,  127: Oi.CAR.HONDA_ACCORD_COUPE_2012, 128: Oi.CAR.HONDA_ACCORD_SEDAN_2012,  129: Oi.CAR.HYUNDAI_VELOSTER_HATCHBACK_2012, 130: Oi.CAR.HYUNDAI_SANTAFE_SUV_2012, 131: Oi.CAR.HYUNDAI_TUCSON_SUV_2012, 132: Oi.CAR.HYUNDAI_VERACRUZ_SUV_2012, 133: Oi.CAR.HYUNDAI_SONATAHYBRID_SEDAN_2012, 134: Oi.CAR.HYUNDAI_ELANTRA_SEDAN_2007, 135: Oi.CAR.HYUNDAI_ACCENT_SEDAN_2012, 136: Oi.CAR.HYUNDAI_GENESIS_SEDAN_2012, 137: Oi.CAR.HYUNDAI_SONATA_SEDAN_2012, 138: Oi.CAR.HYUNDAI_ELANTRA_HATCHBACK_2012, 139: Oi.CAR.HYUNDAI_AZERA_SEDAN_2012, 140: Oi.CAR.INFINITI_G_COUPE_2012, 141: Oi.CAR.INFINITI_QX56_SUV_2011, 142: Oi.CAR.ISUZU_ASCENDER_SUV_2008, 143: Oi.CAR.JAGUAR_XK_XKR_2012, 144: Oi.CAR.JEEP_PATRIOT_SUV_2012, 145: Oi.CAR.JEEP_WRANGLER_SUV_2012, 146: Oi.CAR.JEEP_LIBERTY_SUV_2012, 147: Oi.CAR.JEEP_GRANDCHEROKEE_SUV_2012, 148: Oi.CAR.JEEP_COMPASS_SUV_2012, 149: Oi.CAR.LAMBORGHINI_REVENTON_COUPE_2012, 150: Oi.CAR.LAMBORGHINI_AVENTADOR_COUPE_2012, 151: Oi.CAR.LAMBORGHINI_GALLARDO_SUPERLEGGERA_2012, 152: Oi.CAR.LAMBORGHINI_DIABLO_COUPE_2001, 153: Oi.CAR.LANDROVER_RANGEROVER_SUV_2012, 154: Oi.CAR.LANDROVER_LR2_SUV_2012, 155: Oi.CAR.LINCOLN_TOWN_SEDAN_2011, 156: Oi.CAR.MINI_COOPER_CONVERTIBLE_2012, 157: Oi.CAR.MAYBACH_LANDAULET_CONVERTIBLE_2012, 158: Oi.CAR.MAZDA_TRIBUTE_SUV_2011, 159: Oi.CAR.MCLAREN_MP4_COUPE_2012, 160: Oi.CAR.MERCEDES_300_CONVERTIBLE_1993, 161: Oi.CAR.MERCEDES_C_SEDAN_2012, 162: Oi.CAR.MERCEDES_SL_COUPE_2009, 163: Oi.CAR.MERCEDES_E_SEDAN_2012, 164: Oi.CAR.MERCEDES_S_SEDAN_2012, 165: Oi.CAR.MERCEDES_SPRINTER_VAN_2012, 166: Oi.CAR.MITSUBISHI_LANCER_SEDAN_2012, 167: Oi.CAR.NISSAN_LEAF_HATCHBACK_2012, 168: Oi.CAR.NISSAN_NV_VAN_2012, 169: Oi.CAR.NISSAN_JUKE_HATCHBACK_2012, 170: Oi.CAR.NISSAN_240SX_COUPE_1998, 171: Oi.CAR.PLYMOUTH_NEON_COUPE_1999, 172: Oi.CAR.PORSCHE_PANAMERA_SEDAN_2012, 173: Oi.CAR.RAM_CV_MINIVAN_2012, 174: Oi.CAR.ROLLSROYCE_PHANTOM_COUPE_2012, 175: Oi.CAR.ROLLSROYCE_GHOST_SEDAN_2012, 176: Oi.CAR.ROLLSROYCE_PHANTOM_SEDAN_2012, 177: Oi.CAR.SCION_XD_HATCHBACK_2012, 178: Oi.CAR.SPYKER_C8_CONVERTIBLE_2009, 179: Oi.CAR.SPYKER_C8_COUPE_2009, 180: Oi.CAR.SUZUKI_AERIO_SEDAN_2007, 181: Oi.CAR.SUZUKI_KIZASHI_SEDAN_2012, 182: Oi.CAR.SUZUKI_SX4_HATCHBACK_2012, 183: Oi.CAR.SUZUKI_SX4_SEDAN_2012, 184: Oi.CAR.TESLA_MODELS_SEDAN_2012, 185: Oi.CAR.TOYOTA_SEQUOIA_SUV_2012, 186: Oi.CAR.TOYOTA_CAMRY_SEDAN_2012, 187: Oi.CAR.TOYOTA_COROLLA_SEDAN_2012, 188: Oi.CAR.TOYOTA_4RUNNER_SUV_2012, 189: Oi.CAR.VOLKSWAGEN_GOLF_HATCHBACK_2012, 190: Oi.CAR.VOLKSWAGEN_GOLF_HATCHBACK_1991, 191: Oi.CAR.VOLKSWAGEN_BEETLE_HATCHBACK_2012, 192: Oi.CAR.VOLV0_C30_HATCHBACK_2012, 193: Oi.CAR.VOLV0_240_SEDAN_1993, 194: Oi.CAR.VOLVO_XC90_SUV_2007, 195: Oi.CAR.SMART_FORTWO_CONVERTIBLE_2012}
        self._categories = list(self._mapping.values())
        self._colors = get_palette(len(self._categories))

    def load_filename(self, path, db, line):
        seq = SatelliteSequence()
        parts = line.strip().split(';')
        filepath = parts[0]
        image = SatelliteImage(path + filepath)
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        obj = SatelliteObject()
        obj.bb = (int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
        obj.categories.add(self._mapping[int(parts[5])])
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
        seq = SatelliteSequence()
        parts = line.strip().split(';')
        filepath = parts[0]
        image = SatelliteImage(path + filepath)
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
                obj = SatelliteObject()
                obj.bb = (int(bbox.find('xmin').text), int(bbox.find('ymin').text), int(bbox.find('xmax').text), int(bbox.find('ymax').text))
                obj.categories.add(self._categories[0])
                image.add_object(obj)
        seq.add_image(image)
        return seq

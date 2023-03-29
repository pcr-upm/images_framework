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


class Detection(Component):
    """
    Represent detection instances in the composition.
    """
    def __init__(self):
        super().__init__(1)
        self.database = None
        self.shapefile = None

    def parse_options(self, params):
        import argparse
        parser = argparse.ArgumentParser(prog='Detection', add_help=False)
        parser.add_argument('--database', required=True, choices=[db.__name__ for db in Database.__subclasses__()],
                            help='Select database model.')
        parser.add_argument('--shapefile', dest='shapefile', action="store_true",
                            help='Save results as vector data.')
        args, unknown = parser.parse_known_args(params)
        print(parser.format_usage())
        self.database = args.database
        self.shapefile = args.shapefile
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
            categories = Database.__subclasses__()[datasets.index(self.database)]().get_categories() if self.database else []
            colors = Database.__subclasses__()[datasets.index(self.database)]().get_colors() if self.database else []
            drawing = dict(zip([cat.name for cat in categories], colors))
            image_idx = [np.array_equal(img_pred.filename, elem) for elem in ann_order].index(True)
            for objs_idx, objs_val in enumerate([ann.images[image_idx].objects, img_pred.objects]):
                for obj in objs_val:
                    values = [drawing[cat.label.name] if cat.label in categories else (0, 255, 0) for cat in obj.categories]
                    color = np.mean(values, axis=0)
                    # Draw rectangle (bb or obb)
                    if obj.obb != (-1, -1, -1, -1, -1, -1, -1, -1):
                        (x1, y1, x2, y2, x3, y3, x4, y4) = obj.obb
                        pts = [np.array([[pt] for pt in [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]], dtype=np.int32)]
                        contour = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int32)
                        thickness = int(round(math.log(max(math.e, np.sqrt(cv2.contourArea(contour))), 2)))
                        viewer.polygon(img_pred, pts, color, -1 if objs_idx == 0 else thickness)
                        viewer.circle(img_pred, (int(round(x1)), int(round(y1))), thickness+1, color)
                    else:
                        (xmin, ymin, xmax, ymax) = obj.bb
                        contour = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=np.int32)
                        thickness = int(round(math.log(max(math.e, np.sqrt(cv2.contourArea(contour))), 2)))
                        viewer.rectangle(img_pred, (int(round(xmin)), int(round(ymin))), (int(round(xmax)), int(round(ymax))), color, -1 if objs_idx == 0 else thickness)

    def evaluate(self, fs, ann, pred):
        # id_component;filename;num_ann;num_pred[;ann_id;ann_bb;ann_obb[;ann_label]][;pred_id;pred_bb;pred_obb[;pred_label;pred_score]]
        ann_order = [img_ann.filename for img_ann in ann.images]  # same order among 'ann' and 'pred' images
        for img_pred in pred.images:
            image_idx = [np.array_equal(img_pred.filename, elem) for elem in ann_order].index(True)
            fs.write(str(self.get_component_class()) + ';' + ann.images[image_idx].filename + ';' + str(len(ann.images[image_idx].objects)) + ';' + str(len(img_pred.objects)))
            for objs_idx, objs_val in enumerate([ann.images[image_idx].objects, img_pred.objects]):
                for obj in objs_val:
                    fs.write(';' + str(obj.id) + ';' + str(obj.bb) + ';' + str(obj.obb))
                    for cat in obj.categories:
                        fs.write(';' + cat.label.name) if objs_idx == 0 else fs.write(';' + cat.label.name + ';' + str(cat.score))
            fs.write('\n')

    def save(self, dirname, pred):
        def generate_shp():
            import gdal
            import rasterio
            import xml.etree.ElementTree as Reader
            from osgeo import ogr, osr
            tree = Reader.parse(dirname + os.path.basename(root) + '.xml')
            ifs = rasterio.open(img_pred.filename, 'r')
            driver = ogr.GetDriverByName('ESRI Shapefile')
            ds = driver.CreateDataSource(dirname + os.path.basename(root) + '.shp')
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(int(osr.SpatialReference(wkt=gdal.Open(img_pred.filename).GetProjection()).GetAttrValue('AUTHORITY', 1)))
            layer = ds.CreateLayer('vehicles', srs, ogr.wkbPolygon)
            layer.CreateField(ogr.FieldDefn('Class', ogr.OFTString))
            for obj in tree.getroot().findall('object'):
                name = obj.find('name')
                bbox = obj.find('bndbox')
                tl = ifs.xy(round(float(bbox.find('ymin').text)), round(float(bbox.find('xmin').text)))
                br = ifs.xy(round(float(bbox.find('ymax').text)), round(float(bbox.find('xmax').text)))
                feature = ogr.Feature(layer.GetLayerDefn())
                feature.SetField('Class', name.text)
                ring = ogr.Geometry(ogr.wkbLinearRing)
                ring.AddPoint(float(tl[0]), float(tl[1]))
                ring.AddPoint(float(br[0]), float(tl[1]))
                ring.AddPoint(float(br[0]), float(br[1]))
                ring.AddPoint(float(tl[0]), float(br[1]))
                ring.AddPoint(float(tl[0]), float(tl[1]))
                polygon = ogr.Geometry(ogr.wkbPolygon)
                polygon.AddGeometry(ring)
                feature.SetGeometry(polygon)
                layer.CreateFeature(feature)

        import os
        from pascal_voc_writer import Writer
        for img_pred in pred.images:
            img = cv2.imread(img_pred.filename)
            writer = Writer(img_pred.filename, img.shape[0], img.shape[1], img.shape[2])
            for obj in img_pred.objects:
                (xmin, ymin, xmax, ymax) = obj.bb
                writer.addObject(obj.categories[0].label.name, xmin, ymin, xmax, ymax)
            # Save results as XML file using Pascal VOC format
            root, extension = os.path.splitext(img_pred.filename)
            writer.save(dirname + os.path.basename(root) + '.xml')
            # Save results as shapefile for geographic information system (GIS) software
            if self.shapefile:
                generate_shp()

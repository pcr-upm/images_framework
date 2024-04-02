#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import os
import cv2
import copy
import warnings
import rasterio
import numpy as np
from enum import Enum


class DepthMode(Enum):
    UBYTE = 1
    UINT = 2
    FLOAT = 3


class ChannelsMode(Enum):
    ONE = 1
    THREE = 2
    MULTI = 3


def load_geoimage(filename, depth=DepthMode.UBYTE, channels=ChannelsMode.ONE):
    warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)
    src_raster = rasterio.open(filename, 'r')
    driver = src_raster.driver
    crs = src_raster.crs
    transform = src_raster.transform
    # RasterIO to OpenCV (see inconsistencies between libjpeg and libjpeg-turbo)
    input_type = src_raster.profile['dtype']
    input_channels = src_raster.count
    img = np.zeros((src_raster.height, src_raster.width, src_raster.count), dtype=input_type)
    for band in range(input_channels):
        img[:, :, band] = src_raster.read(band+1)
    # Standard deviation stretch applies a linear stretch to trim off the extreme values
    if input_type == 'uint16':
        aux = copy.deepcopy(img)
        for band in range(input_channels):
            sigma = 2.0
            mean = aux[:, :, band].mean()
            std = aux[:, :, band].std() * sigma
            new_min = np.nanmin(aux[:, :, band])
            new_max = min(mean+std, np.nanmax(aux[:, :, band]))
            aux[:, :, band] = np.clip(aux[:, :, band], new_min, new_max)
            img[:, :, band] = np.interp(aux[:, :, band], (new_min, new_max), (0, 65535))
    elif input_type == 'int16':
        input_type = 'uint8'
        img = img.astype(input_type)
    # Convert depth
    output_type = 'uint8' if depth is DepthMode.UBYTE else 'uint16' if depth is DepthMode.UINT else 'float'
    if input_type != output_type:
        max_input_value = 255 if input_type == 'uint8' else 65535 if input_type == 'uint16' else 1
        max_output_value = 255 if output_type == 'uint8' else 65535 if output_type == 'uint16' else 1
        for band in range(input_channels):
            img[:, :, band] = np.interp(img[:, :, band], (0, max_input_value), (0, max_output_value))
        img = img.astype(output_type)
    # Convert channels
    output_channels = 1 if channels is ChannelsMode.ONE else 3 if channels is ChannelsMode.THREE else 8
    if input_channels > output_channels:
        if transform != rasterio.Affine.identity():
            aux = copy.deepcopy(img)
            # 8-band multi-spectral imagery (Coastal, Blue, Green, Yellow, Red, Red Edge, NIR1, NIR2)
            if input_channels == 8:
                img[:, :, 0] = aux[:, :, 4]
                img[:, :, 1] = aux[:, :, 2]
                img[:, :, 2] = aux[:, :, 1]
            # 4-band multi-spectral imagery (Blue, Green, Red, NIR1)
            elif input_channels == 4:
                img[:, :, 0] = aux[:, :, 2]
                img[:, :, 1] = aux[:, :, 1]
                img[:, :, 2] = aux[:, :, 0]
        img = copy.deepcopy(img[:, :, 0:3])
        # RGB to panchromatic
        if output_channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.expand_dims(img, axis=-1)
    elif input_channels < output_channels:
        # Panchromatic to multi-band
        aux = np.zeros((img.shape[0], img.shape[1], output_channels), dtype=img.dtype)
        for band in range(output_channels):
            aux[:, :, band] = copy.deepcopy(img[:, :, 0])
        img = copy.deepcopy(aux)
    return img, {'driver': driver, 'crs': crs, 'transform': transform}


def save_geoimage(filename, img, profile):
    # OpenCV to RasterIO
    root, extension = os.path.splitext(filename)
    ofs = rasterio.open(root+'.tif', 'w', driver='GTiff', height=img.shape[0], width=img.shape[1],
                        count=img.shape[2], dtype=img.dtype, crs=profile['crs'], transform=profile['transform'])
    for band in range(img.shape[2]):
        ofs.write_band(band+1, img[:, :, band])
    ofs.close()


def clip_objects(objs, roi):
    boxes = []
    for satellite_obj in objs:
        boxes.append(satellite_obj.bb)
    boxes = np.array(boxes)
    center_x = (boxes[:, 0]+boxes[:, 2])*0.5
    center_y = (boxes[:, 1]+boxes[:, 3])*0.5
    cond1 = np.intersect1d(np.where(center_y[:] >= roi[1])[0], np.where(center_x[:] >= roi[0])[0])
    cond2 = np.intersect1d(np.where(center_y[:] < roi[3])[0], np.where(center_x[:] < roi[2])[0])
    indices = np.intersect1d(cond1, cond2)
    return indices


def clip_images(seq, tile_shape, overlap_shape):
    aux = copy.deepcopy(seq)
    seq.clear()
    for satellite_img in aux.images:
        # Sliding window
        for hh in range(0, satellite_img.tile[3]-satellite_img.tile[1], tile_shape[0]-overlap_shape[0]):
            for ww in range(0, satellite_img.tile[2]-satellite_img.tile[0], tile_shape[1]-overlap_shape[1]):
                img_pred = copy.deepcopy(satellite_img)
                img_pred.tile = np.array([ww, hh, ww+tile_shape[1], hh+tile_shape[0]])
                img_pred.objects.clear()
                for satellite_obj in satellite_img.objects:
                    obj_pred = copy.deepcopy(satellite_obj)
                    # We save coordinates referred to the tile position for all objects
                    if obj_pred.bb != (-1, -1, -1, -1):
                        obj_pred.bb = list(np.array(obj_pred.bb) - np.array([ww, hh, ww, hh]))
                    if obj_pred.obb != (-1, -1, -1, -1, -1, -1, -1, -1):
                        obj_pred.obb = list(np.array(obj_pred.obb) - np.array([ww, hh, ww, hh, ww, hh, ww, hh]))
                    if obj_pred.multipolygon is not [np.array([[[-1, -1]], [[-1, -1]], [[-1, -1]]])]:
                        for polygon in obj_pred.multipolygon:
                            polygon -= np.array([ww, hh])
                    img_pred.add_object(obj_pred)
                seq.add_image(img_pred)


def geometry2numpy(geom):
    # Convert geometry from shapely to numpy
    from shapely import geometry
    coords = np.array(geometry.mapping(geom)['coordinates'], dtype=object)
    if geom.geom_type is 'Point':
        contours = [np.array([[[coords[0], coords[1]]]], dtype=int)]
    elif geom.geom_type is 'LineString':
        contours = [np.array([[[pt[0], pt[1]]] for pt in coords], dtype=int)]
    elif geom.geom_type is 'Polygon':
        contours = [np.array([[[pt[0], pt[1]]] for pt in coords[0]], dtype=int)]
    else:
        contours = np.empty((len(coords),), dtype=object)
        for idx in range(len(coords)):
            contours[idx] = np.array([[[pt[0], pt[1]]] for pt in coords[idx][0]], dtype=int)
    return contours


def numpy2geometry(contour):
    # Convert geometry from numpy to shapely
    from shapely.geometry import Point, LineString, Polygon
    coords = np.array([[pt[0][0], pt[0][1]] for pt in contour], dtype=int)
    if len(coords) == 1:
        geom = Point(coords[0]).wkt
    elif len(coords) == 2:
        geom = LineString(coords).wkt
    else:
        geom = Polygon(coords).wkt
    return geom


def mask2contours(img):
    # Generate several numpy contours from a mask image
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    contours = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
    return contours


def contours2mask(height, width, contours, labels, mapping):
    # Generate a mask image from several numpy contours
    mask = np.ones((height, width)) * 255
    if contours:
        assert len(contours) == len(labels)
        np_contours = np.empty((len(contours),), dtype=object)
        for idx in range(len(np_contours)):
            np_contours[idx] = contours[idx]
        np_labels = np.array(labels)
        for key, val in mapping.items():
            cv2.drawContours(mask, np_contours[np_labels == key], -1, val, cv2.FILLED)
    mask = mask.astype(np.uint8)
    # # Draw contours using random colors
    # from PIL import ImageColor
    # import matplotlib.colors as mcolors
    # colors = [color for name, color in mcolors.XKCD_COLORS.items()]
    # color_mask = np.ones((height, width, 3)) * 255
    # for idx, val in enumerate(list(mapping.values())):
    #     color_mask[mask == val] = ImageColor.getcolor(colors[idx], 'RGB')
    # cv2.imshow('color_mask', color_mask.astype(np.uint8))
    # cv2.waitKey(0)
    return mask

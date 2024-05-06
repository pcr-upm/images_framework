#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import os
import cv2
import math
import numpy as np
from .utils import load_geoimage, save_geoimage, DepthMode, ChannelsMode


def keygen(image):
    """
    It generates a unique key to store different tiles from the same image.
    """
    filename, ext = os.path.splitext(image.filename)
    return filename + '_' + '{:04d}'.format(image.tile[0]) + '_' + '{:04d}'.format(image.tile[1]) + ext


class Viewer:
    """
    Class viewer interface implementation.
    """
    class GeoImage:
        def __init__(self, img, profile):
            self.img = img
            self.profile = profile

    def __init__(self, window_title):
        self._geoimages = {}
        self._window_title = window_title

    def set_image(self, image):
        img, profile = load_geoimage(image.filename, DepthMode.UBYTE, ChannelsMode.THREE)
        # Add black border
        if img.shape[0] < image.tile[3] or img.shape[1] < image.tile[2]:
            tile_shape = (image.tile[3]-image.tile[1], image.tile[2]-image.tile[0])
            input_image_pad = np.zeros([img.shape[0]+tile_shape[0], img.shape[1]+tile_shape[1], 3], np.uint8)
            input_image_pad[0:img.shape[0], 0:img.shape[1], :] = img
            img = input_image_pad
        img = img[image.tile[1]:image.tile[3], image.tile[0]:image.tile[2]]
        self._geoimages.update({keygen(image): self.GeoImage(img, profile)})

    def get_image(self, image):
        return self._geoimages[keygen(image)].img

    def circle(self, image, pt, radius, color, thickness=-1):
        img = self.get_image(image)
        if thickness > 0:
            cv2.circle(img, pt, radius, color, thickness)
        else:
            mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
            cv2.circle(mask, pt, radius, 255, thickness)
            idx = (mask == 255)
            self._geoimages[keygen(image)].img[idx] = np.array(0.75*img[idx]+0.25*np.array(color), np.uint8)

    def line(self, image, pt1, pt2, color, thickness=-1):
        img = self.get_image(image)
        if thickness > 0:
            cv2.line(img, pt1, pt2, color, thickness)
        else:
            mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
            cv2.line(mask, pt1, pt2, 255, thickness)
            idx = (mask == 255)
            self._geoimages[keygen(image)].img[idx] = np.array(0.75*img[idx]+0.25*np.array(color), np.uint8)

    def rectangle(self, image, pt1, pt2, color, thickness=-1):
        img = self.get_image(image)
        if thickness > 0:
            cv2.rectangle(img, pt1, pt2, color, thickness)
        else:
            mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
            cv2.rectangle(mask, pt1, pt2, 255, thickness)
            idx = (mask == 255)
            self._geoimages[keygen(image)].img[idx] = np.array(0.75*img[idx]+0.25*np.array(color), np.uint8)

    def ellipse(self, image, major_axis_length, minor_axis_length, angle, center, color, thickness=-1):
        pts = list([])
        ellipse_number_of_points = 100
        for i in range(ellipse_number_of_points):
            fi = 2 * math.pi * (i / ellipse_number_of_points)
            major_cos = major_axis_length * math.cos(fi)
            minor_sin = minor_axis_length * math.sin(fi)
            cos_angle = math.cos(angle)
            sin_angle = math.sin(angle)
            pts.append([center[0]+(cos_angle*major_cos)-(sin_angle*minor_sin), center[1]+(sin_angle*major_cos)+(cos_angle*minor_sin)])
        pts = [np.array([[pt] for pt in pts], dtype=np.int32)]
        img = self.get_image(image)
        if thickness > 0:
            cv2.polylines(img, pts, True, color, thickness)
        else:
            mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
            cv2.fillPoly(mask, pts, 255)
            idx = (mask == 255)
            self._geoimages[keygen(image)].img[idx] = np.array(0.75*img[idx]+0.25*np.array(color), np.uint8)

    def polygon(self, image, pts, color, thickness=-1):
        img = self.get_image(image)
        if thickness > 0:
            cv2.polylines(img, pts, True, color, thickness)
        else:
            mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
            cv2.fillPoly(mask, pts, 255)
            idx = (mask == 255)
            self._geoimages[keygen(image)].img[idx] = np.array(0.75*img[idx]+0.25*np.array(color), np.uint8)

    def text(self, image, text, pt, scale, color):
        cv2.putText(self.get_image(image), text, pt, cv2.FONT_HERSHEY_SIMPLEX, scale, color)

    def show(self, delay):
        canvas = None
        for key in self._geoimages:
            frame = cv2.cvtColor(self._geoimages[key].img, cv2.COLOR_RGB2BGR)
            canvas = frame if canvas is None else np.hstack((canvas, frame))  # stack images in sequence horizontally
        cv2.namedWindow(self._window_title, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        cv2.moveWindow(self._window_title, 0, 0)
        cv2.imshow(self._window_title, canvas)
        cv2.waitKey(delay)
        # cv2.destroyWindow(self._window_title)
        self._geoimages.clear()

    def save(self, path):
        import uuid
        filename = str(uuid.uuid4())
        canvas = None
        # h, w, c = next(iter(self._geoimages.values())).img.shape
        # video = cv2.VideoWriter(path+filename+'.avi', fourcc=cv2.VideoWriter_fourcc(*'XVID'), fps=30, frameSize=(w, h))
        for key in self._geoimages:
            frame = cv2.cvtColor(self._geoimages[key].img, cv2.COLOR_RGB2BGR)
            canvas = frame if canvas is None else np.hstack((canvas, frame))  # stack images in sequence horizontally
        #     video.write(frame)
        # video.release()
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        save_geoimage(path+filename+'.tif', canvas, self._geoimages[key].profile)
        self._geoimages.clear()

#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@geoaitech.com'

import os
import sys
sys.path.append(os.getcwd())
import cv2
import numpy as np
from satellite_framework.src.composite import SatelliteComposite
from satellite_framework.src.annotations import SatelliteSequence, SatelliteImage
from satellite_framework.src.viewer import Viewer


def main():
    """
    Satellite framework test script.
    """
    print('OpenCV ' + cv2.__version__)
    composite = SatelliteComposite()

    # Process frame and show results
    ann = SatelliteSequence()
    pred = SatelliteSequence()
    for roi in [[0, 0, 600, 600], [0, 600, 600, 1200], [0, 1200, 600, 1800]]:
        img_pred = SatelliteImage('satellite_framework/test/example.png')
        img_pred.tile = np.array(roi)
        pred.add_image(img_pred)
    composite.process(ann, pred)
    viewer = Viewer('satellite_framework_test')
    for satellite_img in pred.images:
        viewer.set_image(satellite_img)
    composite.show(viewer, ann, pred)
    viewer.show(0)
    print('End of satellite_framework_test')


if __name__ == '__main__':
    main()

#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import os
import sys
sys.path.insert(0, os.getcwd())
import cv2
import numpy as np
from pathlib import Path
from pcr_framework.src.composite import Composite
from pcr_framework.src.annotations import GenericVideo, AerialImage
from pcr_framework.src.viewer import Viewer


def main():
    """
    PCR framework test script.
    """
    print('OpenCV ' + cv2.__version__)
    print('NumPy ' + np.__version__)
    # Process frame and show result
    ann, pred = GenericVideo(), GenericVideo()
    for roi in [[0, 0, 600, 600], [0, 600, 600, 1200], [0, 1200, 600, 1800]]:
        img_pred = AerialImage('pcr_framework/test/example.tif')
        img_pred.tile = np.array(roi)
        pred.add_image(img_pred)
    viewer = Viewer('pcr_framework_test')
    for img_pred in pred.images:
        viewer.set_image(img_pred)
    composite = Composite()
    composite.process(ann, pred)
    composite.show(viewer, ann, pred)
    # Save result
    dirname = 'pcr_framework/output/images/'
    Path(dirname).mkdir(parents=True, exist_ok=True)
    viewer.save(dirname, as_video=False, format='tif')
    print('End of pcr_framework_test')


if __name__ == '__main__':
    main()

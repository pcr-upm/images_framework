<h1 align="center">
<img src="https://github.com/user-attachments/assets/b283f327-679b-4e1d-88bf-d57cf0745a2f" width="300">
</h1><br>

[![PyPI version](https://badge.fury.io/py/images-framework.svg)](https://badge.fury.io/py/images-framework)

### Installation
```bash
pip install images-framework
```

This will automatically install most dependencies (OpenCV, SciPy, Rasterio, Pillow, Pascal-voc-writer).

**Important:** GDAL is not installed automatically via `pip`.
```bash
conda install -c conda-forge gdal
```

### Usage
You can run the provided test script to check the library:
```bash
python images_framework/test/images_framework_test.py
```

Or you can use the library directly in your own Python code:
```python
import numpy as np
from images_framework.src.annotations import GenericVideo, GenericImage
from images_framework.src.composite import Composite
from images_framework.src.viewer import Viewer

# Prepare annotations and predictions.
# GenericVideo represents a set of images or a sequence of frames from a video
# for solving a generic computer vision problem.
ann, pred = GenericVideo(), GenericVideo()

# For example, you can split an image into smaller tiles (regions of interest)
# and process each part separately.
for roi in [[0, 0, 600, 600], [0, 600, 600, 1200], [0, 1200, 600, 1800]]:
    img_pred = GenericImage('images_framework/test/example.tif')
    img_pred.tile = np.array(roi)
    pred.add_image(img_pred)

# Viewer class is used to visualize the results of the processing.
viewer = Viewer('images_framework_test')
for img_pred in pred.images:
    viewer.set_image(img_pred)

# Composite class acts as a container where you can plug in different computer 
# vision modules such as detectors, segmentators, regressors, classifiers, etc.
# Once the modules are incorporated, you can call process() to run them over 
# the frames and show() allows you to visualize the results.
composite = Composite()
composite.process(ann, pred)
composite.show(viewer, ann, pred)
```

### Contributions
The [PCR-UPM](https://pcr-upm.github.io/) group welcomes your expertise and enthusiasm!

Small improvements or fixes are always appreciated. If you are considering larger contributions to the source code, please contact bobetocalo@gmail.com first.

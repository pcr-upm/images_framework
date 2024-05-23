#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import os
import sys
sys.path.append(os.getcwd())
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from eval_tools import draw_confusion_matrix, metric_accuracy
from images_framework.src.utils import load_geoimage, geometry2numpy, contours2mask


def parse_file(input_file):
    """
    Parse file to extract results.
    """
    from shapely import wkt
    from natsort import natsorted
    from collections import namedtuple
    Result = namedtuple('Result', 'annotation prediction')
    results, categories = dict(), set()
    with open(input_file) as ifs:
        lines = ifs.readlines()
        for i in tqdm(range(len(lines)), file=sys.stdout):
            line = lines[i].strip()  # remove last newline
            parts = line.split(';')
            if int(parts.pop(0)) != 2:
                continue
            filename = parts.pop(0)
            tile = np.array(parts.pop(0)[1:-1].split(',')).astype(int)
            num_annotations = int(parts.pop(0))
            num_predictions = int(parts.pop(0))
            ann_result, pred_result = [], []
            for _ in range(num_annotations):
                identifier = parts.pop(0)
                num_contours = int(parts.pop(0))
                ann_geometries, ann_labels = [], []
                for _ in range(int(num_contours)):
                    geom = wkt.loads(parts.pop(0))
                    ann_geometries.append(geom)
                num_labels = int(parts.pop(0))
                for _ in range(int(num_labels)):
                    category = parts.pop(0)
                    categories.add(category)
                    ann_labels.append(category)
                ann_result.append([ann_geometries, ann_labels])
            for _ in range(num_predictions):
                identifier = parts.pop(0)
                num_contours = int(parts.pop(0))
                pred_geometries, pred_labels = [], []
                for _ in range(int(num_contours)):
                    geom = wkt.loads(parts.pop(0))
                    pred_geometries.append(geom)
                num_labels = int(parts.pop(0))
                for _ in range(int(num_labels)):
                    category = parts.pop(0)
                    categories.add(category)
                    pred_labels.append(category)
                pred_result.append([pred_geometries, pred_labels])
            results[filename] = Result(ann_result, pred_result)
    ifs.close()
    categories = natsorted(list(categories))
    return results, categories


def main():
    print('Program started ...')
    input_file = 'images_framework/output/results.txt'
    parser = argparse.ArgumentParser(prog='eval_segmentation', add_help=False)
    parser.add_argument('--background', dest='background', action='store_true', help='Evaluate background label.')
    parser.set_defaults(background=False)
    args, unknown = parser.parse_known_args()
    print(parser.format_usage())
    if os.path.exists(input_file):
        results, categories = parse_file(input_file)
        mapping = dict(zip(categories, range(len(categories))))
        if args.background:
            mapping['Background'] = 255
        cm = np.zeros((len(mapping), len(mapping)), dtype=np.int64)
        for filename, result in tqdm(results.items(), file=sys.stdout):
            img, _ = load_geoimage(filename)
            height, width, _ = np.shape(img)
            ann_contours = [geometry2numpy(ann[0][0])[0] for ann in result.annotation]
            ann_labels = [ann[1][0] for ann in result.annotation]
            annotation_mask = contours2mask(height, width, ann_contours, ann_labels, mapping)
            pred_contours = [geometry2numpy(pred[0][0])[0] for pred in result.prediction]
            pred_labels = [pred[1][0] for pred in result.prediction]
            prediction_mask = contours2mask(height, width, pred_contours, pred_labels, mapping)
            cm += confusion_matrix(annotation_mask.flatten(), prediction_mask.flatten(), labels=list(mapping.values()))
        if not np.all((cm == 0)):
            # Compute the confusion matrix
            print('Confusion matrix:')
            print(cm)
            draw_confusion_matrix(cm, list(mapping.keys()), True)
            metric_accuracy(cm, sorted(categories))
            print('='*50)
            # Jaccard index is defined as the intersection of two sets divided by their union.
            # Jaccard = |A∩B| / |A∪B| = TP / (TP + FP + FN)
            correct_samples_class = np.diag(cm).astype(float)
            total_predicts_class = np.sum(cm, axis=0).astype(float)
            total_samples_class = np.sum(cm, axis=1).astype(float)
            iou = correct_samples_class / np.maximum(1.0, total_samples_class+total_predicts_class-correct_samples_class)
            print('Mean IoU: %.3f%%' % (iou.mean() * 100))
            for idx, val in enumerate(mapping):
                print('> %s: Iou: %.3f%%' % (val, iou[idx] * 100))
        else:
            print('Empty object segmentation results')
    else:
        print('Usage: python eval_segmentation.py')


if __name__ == '__main__':
    main()

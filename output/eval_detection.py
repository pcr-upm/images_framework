#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import os
import sys
sys.path.append(os.getcwd())
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from eval_tools import draw_precision_recall, draw_confusion_matrix


def area_intersection(boxes, box):
    """
    Computes the intersection between a predicted bounding box and each annotated bounding box.
    """
    xmin = np.maximum(np.min(boxes[:, 0::2], axis=1), np.min(box[0::2]))
    ymin = np.maximum(np.min(boxes[:, 1::2], axis=1), np.min(box[1::2]))
    xmax = np.minimum(np.max(boxes[:, 0::2], axis=1), np.max(box[0::2]))
    ymax = np.minimum(np.max(boxes[:, 1::2], axis=1), np.max(box[1::2]))
    w = np.maximum(xmax - xmin + 1.0, 0.0)
    h = np.maximum(ymax - ymin + 1.0, 0.0)
    return w * h


def area_union(boxes, box):
    """
    Computes the union between a predicted bounding box and each annotated bounding box.
    """
    area_anns = (np.max(box[0::2])-np.min(box[0::2])+1.0) * (np.max(box[1::2])-np.min(box[1::2])+1.0)
    area_pred = (np.max(boxes[:, 0::2], axis=1)-np.min(boxes[:, 0::2], axis=1)+1.0) * (np.max(boxes[:, 1::2], axis=1)-np.min(boxes[:, 1::2], axis=1)+1.0)
    return area_anns + area_pred - area_intersection(boxes, box)


def calc_iou(is_obb_mode, boxes, box):
    """
    We estimate the ground truth annotation that best matches with the prediction according to IoU metric.
    """
    if is_obb_mode:
        from output.obb_devkit import polyiou as p
        iou = [p.iou_poly(p.VectorDouble(b), p.VectorDouble(box)) for b in boxes]
    else:
        iou = area_intersection(boxes, box) / area_union(boxes, box)
    max_value = np.max(iou)
    max_index = np.argmax(iou)
    return max_value, max_index


def calc_ap(rec, prec, use_07_metric=False):
    """
    Computes AP given precision and recall. If use_07_metric is True, uses the VOC 2007 11 point method.
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # First append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))
        # Compute the precision envelope
        for i in range(mpre.size-1, 0, -1):
            mpre[i-1] = np.maximum(mpre[i-1], mpre[i])
        # To calculate area under PR curve, look for points where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]
        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i+1] - mrec[i]) * mpre[i+1])
    return ap


def parse_file(input_file):
    """
    Parse file to extract results.
    """
    from natsort import natsorted
    annotations, predictions = {}, {}
    categories = set()
    with open(input_file) as ifs:
        lines = ifs.readlines()
        for i in tqdm(range(len(lines)), file=sys.stdout):
            line = lines[i].strip()  # remove last newline
            parts = line.split(';')
            if int(parts.pop(0)) != 1:
                continue
            filename, ext = os.path.splitext(os.path.basename(parts.pop(0)))
            tile = np.array(parts.pop(0)[1:-1].split(',')).astype(int)
            keyname = filename + '_' + '_'.join([str(val) for val in tile]) + ext
            annotations.setdefault(keyname, {})
            predictions.setdefault(keyname, {})
            num_annotations = int(parts.pop(0))
            num_predictions = int(parts.pop(0))
            # print(keyname, num_annotations, num_predictions)
            for idx in range(num_annotations):
                identifier = parts.pop(0)
                bb = np.array(parts.pop(0)[1:-1].split(',')).astype(float)
                obb = np.array(parts.pop(0)[1:-1].split(',')).astype(float)
                label = parts.pop(0)
                categories.add(label)
                annotations[keyname].setdefault(label, {'bbox': [], 'obb': []})
                annotations[keyname][label]['bbox'].append(bb)
                annotations[keyname][label]['obb'].append(obb)
            for idx in range(num_predictions):
                identifier = parts.pop(0)
                bb = np.array(parts.pop(0)[1:-1].split(',')).astype(float)
                obb = np.array(parts.pop(0)[1:-1].split(',')).astype(float)
                label = parts.pop(0)
                confidence = float(parts.pop(0))
                categories.add(label)
                predictions[keyname].setdefault(label, {'bbox': [], 'obb': [], 'confidence': []})
                predictions[keyname][label]['bbox'].append(bb)
                predictions[keyname][label]['obb'].append(obb)
                predictions[keyname][label]['confidence'].append(confidence)  # sort detections by confidence
    ifs.close()
    categories = natsorted(list(categories))
    return annotations, predictions, categories


def main():
    print('Program started ...')
    input_file = 'images_framework/output/results.txt'

    if len(sys.argv) == 1 and os.path.exists(input_file):
        annotations, predictions, categories = parse_file(input_file)
        # Define horizontal or oriented bounding boxes
        is_obb_mode = False
        if bool(annotations):
            aps = []
            thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            default_cls = 'Background'
            for threshold in thresholds:
                y_true, y_pred = [], []  # confusion matrix
                tps, confidences = dict(), dict()  # draw precision-recall curves for each category
                for cls in categories:
                    # Compute TP, FP and FN for each image
                    tps[cls], confidences[cls] = [], []
                    for keyname in predictions:
                        # Sort 'cls' predictions by confidence for each file
                        pred_hbbs, pred_obbs, pred_confidences = [], [], []
                        if cls in predictions[keyname].keys():
                            for idx in range(len(predictions[keyname][cls]['bbox'])):
                                pred_hbbs.append(predictions[keyname][cls]['bbox'][idx])
                                pred_obbs.append(predictions[keyname][cls]['obb'][idx])
                                pred_confidences.append(predictions[keyname][cls]['confidence'][idx])
                            sorted_ind = np.argsort(-np.array(pred_confidences))
                            pred_hbbs = np.array(pred_hbbs)[sorted_ind, :]
                            pred_obbs = np.array(pred_obbs)[sorted_ind, :]
                        pred_hbbs, pred_obbs = np.array(pred_hbbs).astype(float), np.array(pred_obbs).astype(float)
                        # Define 'cls' annotations for each file
                        anno_hbbs, anno_obbs = [], []
                        if cls in annotations[keyname].keys():
                            anno_hbbs = annotations[keyname][cls]['bbox']
                            anno_obbs = annotations[keyname][cls]['obb']
                        anno_hbbs, anno_obbs = np.array(anno_hbbs).astype(float), np.array(anno_obbs).astype(float)
                        pred_boxes = pred_obbs if is_obb_mode else pred_hbbs
                        anno_boxes = anno_obbs if is_obb_mode else anno_hbbs
                        anno_indices = list(range(len(anno_boxes)))
                        # Compare a single prediction 'pred_box' with all annotations 'anno_boxes'
                        for pred_idx, pred_box in enumerate(pred_boxes):
                            # A prediction is correct if its IoU with the ground truth is above the threshold
                            iou_value, ann_index = calc_iou(is_obb_mode, anno_boxes, pred_box) if len(anno_boxes) > 0 else (-1, -1)
                            if iou_value > threshold and ann_index in anno_indices:
                                # TP
                                anno_indices.remove(int(ann_index))
                                tps[cls] += [1.0]
                                y_true += [cls]
                            else:
                                # FP
                                tps[cls] += [0.0]
                                y_true += [default_cls]
                            y_pred += [cls]
                            confidences[cls] += [pred_confidences[pred_idx]]
                        # FN
                        y_true += [cls] * len(anno_indices)
                        y_pred += [default_cls] * len(anno_indices)
                # Compute AP metric
                print('IoU threshold:', threshold)
                y_true, y_pred = np.array(y_true), np.array(y_pred)
                precision_list, recall_list, ap_list = [], [], []
                for cls in categories:
                    sorted_ind = np.argsort(-np.array(confidences[cls]))
                    tp = np.cumsum(np.array(tps[cls])[sorted_ind], dtype=float)
                    recall = np.array([0.0]) if len(tp) == 0 else tp / np.maximum(np.sum(y_true == cls), np.finfo(np.float64).eps)
                    precision = np.array([0.0]) if len(tp) == 0 else tp / np.maximum(list(range(1, np.sum(y_pred == cls)+1)), np.finfo(np.float64).eps)
                    ap = calc_ap(recall, precision)
                    print('> %s: Recall: %.3f%% Precision: %.3f%% AP: %.3f%%' % (cls, recall[-1]*100, precision[-1]*100, ap*100))
                    precision_list.append(precision)
                    recall_list.append(recall)
                    ap_list.append(ap)
                mean_ap = np.mean(ap_list)
                print('mAccuracy: %.3f%%' % (accuracy_score(y_true, y_pred)*100))
                print('mRecall: %.3f%%' % (recall_score(y_true, y_pred, average='macro', zero_division=1)*100))
                print('mPrecision: %.3f%%' % (precision_score(y_true, y_pred, average='macro', zero_division=1)*100))
                print('mAP: %.3f%%' % (mean_ap*100))
                aps.append(mean_ap)
                if threshold == 0.5:
                    names = categories.copy()
                    names.insert(0, default_cls)
                    cm = confusion_matrix(y_true, y_pred, labels=names)
                    print('Confusion matrix:')
                    print(cm)
                    draw_confusion_matrix(cm, names, True)
                    draw_precision_recall(precision_list, recall_list, categories)
                print('=====' * 10)
            print('mmAP: %.3f%%' % (sum(aps)/len(aps)*100))
        else:
            print('Empty object detection results')
    else:
        print('Usage: python eval_detection.py')


if __name__ == '__main__':
    main()

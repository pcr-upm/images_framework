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
from eval_tools import draw_confusion_matrix


def parse_file(input_file):
    """
    Parse file to extract results.
    """
    from natsort import natsorted
    results = {}
    categories = set()
    with open(input_file) as ifs:
        lines = ifs.readlines()
        for i in tqdm(range(len(lines)), file=sys.stdout):
            line = lines[i].strip()  # remove last newline
            parts = line.split(';')
            if int(parts.pop(0)) != 4:
                continue
            filename, ext = os.path.splitext(os.path.basename(parts.pop(0)))
            tile = np.array(parts.pop(0)[1:-1].split(',')).astype(int)
            keyname = filename + '_' + '_'.join([str(val) for val in tile]) + ext
            results.setdefault(keyname, {})
            num_annotations = int(parts.pop(0))
            num_predictions = int(parts.pop(0))
            # print(keyname, num_annotations, num_predictions)
            for idx in range(num_annotations):
                identifier = parts.pop(0)
                label = parts.pop(0)
                results[keyname].setdefault(identifier, {})
                results[keyname][identifier]['annotation'] = label
                categories.add(label)
            for idx in range(num_predictions):
                identifier = parts.pop(0)
                label = parts.pop(0)
                score = float(parts.pop(0))
                results[keyname].setdefault(identifier, {})
                results[keyname][identifier]['prediction'] = label
                results[keyname][identifier]['score'] = score
                categories.add(label)
    ifs.close()
    categories = natsorted(list(categories))
    return results, categories


def main():
    print('Program started ...')
    input_file = 'images_framework/output/results.txt'

    if len(sys.argv) == 1 and os.path.exists(input_file):
        results, categories = parse_file(input_file)
        if bool(results):
            # Compute the confusion matrix
            y_true, y_pred = [], []
            for keyname in results.keys():
                for identifier in results[keyname].keys():
                    y_true.append(results[keyname][identifier]['annotation'])
                    y_pred.append(results[keyname][identifier]['prediction'])
            cm = confusion_matrix(y_true, y_pred, labels=categories)
            print('Confusion matrix:')
            print(cm)
            draw_confusion_matrix(cm, categories, True)
            print('mAccuracy: %.3f%%' % (accuracy_score(y_true, y_pred)*100))
            print('mRecall: %.3f%%' % (recall_score(y_true, y_pred, average='macro', zero_division=1)*100))
            print('mPrecision: %.3f%%' % (precision_score(y_true, y_pred, average='macro', zero_division=1)*100))
            # Draw metrics for each category
            for idx, val in enumerate(categories):
                # True/False Positives (TP/FP) refer to the number of predicted positives that were correct/incorrect.
                # True/False Negatives (TN/FN) refer to the number of predicted negatives that were correct/incorrect.
                tp = cm[idx, idx]
                fp = sum(cm[:, idx]) - tp
                fn = sum(cm[idx, :]) - tp
                tn = sum(np.delete(sum(cm) - cm[idx, :], idx))
                # True Positive Rate: proportion of real positive cases that were correctly predicted as positive.
                recall = tp / np.maximum(tp+fn, np.finfo(np.float64).eps)
                # Precision: proportion of predicted positive cases that were truly real positives.
                precision = tp / np.maximum(tp+fp, np.finfo(np.float64).eps)
                # True Negative Rate: proportion of real negative cases that were correctly predicted as negative.
                specificity = tn / np.maximum(tn+fp, np.finfo(np.float64).eps)
                # Dice coefficient refers to two times the intersection of two sets divided by the sum of their areas.
                # Dice = 2 |A∩B| / (|A|+|B|) = 2 TP / (2 TP + FP + FN)
                f1_score = 2 * ((precision * recall) / np.maximum(precision+recall, np.finfo(np.float64).eps))
                print('> %s: Recall: %.3f%% Precision: %.3f%% Specificity: %.3f%% Dice: %.3f%%' % (val, recall*100, precision*100, specificity*100, f1_score*100))
        else:
            print('Empty object recognition results')
    else:
        print('Usage: python eval_recognition.py')


if __name__ == '__main__':
    main()

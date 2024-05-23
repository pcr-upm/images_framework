#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import os
import sys
sys.path.append(os.getcwd())
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from eval_tools import draw_confusion_matrix, metric_accuracy


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
    input_file = 'output/results.txt'

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
            metric_accuracy(cm, categories)
        else:
            print('Empty object recognition results')
    else:
        print('Usage: python eval_recognition.py')


if __name__ == '__main__':
    main()

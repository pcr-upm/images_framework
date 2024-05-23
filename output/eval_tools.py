#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import numpy as np
import matplotlib as mpl
from pathlib import Path
mpl.rcParams['figure.figsize'] = [25.6, 19.2]  # figure size in inches
mpl.rcParams['figure.dpi'] = 100  # figure resolution in dots per inch
mpl.rcParams['savefig.dpi'] = 100
mpl.rcParams['lines.linewidth'] = 15
mpl.rcParams['font.size'] = 20
mpl.rcParams['axes.labelsize'] = 40
mpl.rcParams['legend.fontsize'] = 25


def draw_histogram(errors, categories):
    # Draw histogram of errors
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlim(0, len(categories))
    plt.bar(np.arange(len(categories)), errors, 0.7, color='blue')
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels, loc='lower right')
    plt.xlabel('Landmark index')
    plt.ylabel('NME')
    ax.set_xticks(np.arange(len(categories))+0.35)
    plt.setp(ax.set_xticklabels(categories), rotation=90)
    plt.grid('on', linestyle='--', linewidth=1.5)
    plt.savefig('output/histogram_errors.eps', format='eps')
    plt.close(fig)


def draw_cumulative_curve(errors, categories, threshold=15, database=None):
    # Draw cumulative curves for each category
    import matplotlib.pyplot as plt
    import matplotlib.colors as col
    from eval_alignment import area_under_curve, failure_rate
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.axis([0, threshold, 0, 1])
    c_dark = list(filter(lambda x: x.startswith('dark'), col.cnames.keys()))
    aucs, frs = [], []
    for idx in range(errors.shape[1]):
        values, base = np.histogram(errors[:, idx], bins=10000)
        cumulative = np.cumsum(values/float(errors.shape[0]))
        base = [x for x in base if (x < threshold)]
        cumulative = cumulative[0:len(base)]
        np.savetxt('output/cum/cumulative_' + str(idx) + '.txt', np.column_stack([base, cumulative]), fmt='%1.6f')
        aucs.append(area_under_curve(base, cumulative, threshold))
        frs.append(failure_rate(base, cumulative, threshold))
        plt.plot(base, cumulative, color=c_dark[idx], zorder=20, label=categories[idx])
    literature = {'burgos13': ('RCPR', '#96f97b'), 'kazemi14': ('ERT', '#929591'), 'lee15': ('cGPRT', '#ffff14'), 'honari16': ('RCN', '#f97306'), 'marek17': ('DAN', '#c20078'), 'yang17': ('SHN', '#78c930'), 'deng17': ('MHM', '#944f00'), 'wu18': ('LAB', '#15b01a'), 'valle18': ('DCFE', '#e50000'), 'feng18': ('PRN', '#7e1e9c'), 'dong18': ('SAN', '#e6daa6'), 'valle19a': ('3DDE''#653700', '#653700'), 'valle19b': ('CHR2C', '#00e1a2'), 'valle20': ('MNN+OERT', '#0096d1'), 'dad22': ('DAD-3DHeads', '#69d100')}
    if database is not None:
        import os
        from pathlib import Path
        dirname = 'output/cum/landmarks/' + database
        for i in os.listdir(dirname):
            if i.endswith('.txt'):
                tmp = np.loadtxt(os.path.join(dirname, i))
                base = tmp[:, 0]
                cumulative = tmp[:, 1]
                base = [x for x in base if (x < threshold)]
                cumulative = cumulative[0:len(base)]
                aucs.append(area_under_curve(base, cumulative, threshold))
                frs.append(failure_rate(base, cumulative, threshold))
                plt.plot(base, cumulative, color=literature[Path(i).stem][1], zorder=list(literature.keys()).index(Path(i).stem), label=literature[Path(i).stem][0])
    handles, labels = ax.get_legend_handles_labels()
    handles = [h for (a, h) in sorted(zip(aucs, handles), reverse=True)]
    labels = [l for (a, l) in sorted(zip(aucs, labels), reverse=True)]
    frs = [f for (a, f) in sorted(zip(aucs, frs), reverse=True)]
    plt.legend(handles, labels, loc='lower right')
    plt.xlabel('Cumulative error')
    plt.ylabel('Images proportion')
    plt.grid('on', linestyle='--', linewidth=1.5)
    auc = ['{:.2f}'.format(x) for x in sorted(aucs, reverse=True)]
    fr = ['{:.2f}'.format(x) for x in frs]
    labels = [str(l + ' [' + a + '] (' + f + ')') for (l, a, f) in zip(labels, auc, fr)]
    plt.legend(handles, labels, loc='upper left')
    plt.savefig('output/cumulative_curve.eps', format='eps')
    plt.close(fig)


def draw_precision_recall(precisions, recalls, categories):
    # Draw precision-recall curves for each category
    import matplotlib.pyplot as plt
    import matplotlib.colors as col
    from eval_detection import calc_ap
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.axis([0, 1, 0, 1])
    c_dark = list(filter(lambda x: x.startswith('dark'), col.cnames.keys()))
    aps = []
    dirname = 'output/pr/'
    Path(dirname).mkdir(parents=True, exist_ok=True)
    # Compare algorithms for a specific category
    # for idx, filename in enumerate(['SCRDet19', 'RetinaNet17']):
    #     precision, recall = np.loadtxt('output/pr/cowc_' + filename.lower() + '.txt', unpack=True)
    #     plt.plot(recall, precision, color=c_dark[idx], label=filename)
    #     aps.append(calc_ap(recall, precision))
    # Compare categories for a specific algorithm
    for idx in range(len(categories)):
        np.savetxt('output/pr/precision_recall_' + str(idx) + '.txt', np.column_stack([precisions[idx], recalls[idx]]))
        plt.plot(recalls[idx], precisions[idx], color=c_dark[idx], label=categories[idx])
        aps.append(calc_ap(recalls[idx], precisions[idx]))
    handles, labels = ax.get_legend_handles_labels()
    labels = [str(val + ' [' + '{:.3f}'.format(aps[idx]) + ']') for idx, val in enumerate(labels)]
    handles = [h for (ap, h) in sorted(zip(aps, handles), key=lambda x: x[0], reverse=True)]
    labels = [l for (ap, l) in sorted(zip(aps, labels), key=lambda x: x[0], reverse=True)]
    leg = plt.legend(handles, labels, loc='lower left')
    leg.set_zorder(100)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid('on', linestyle='--', linewidth=1.5)
    plt.savefig('output/precision_recall.eps', format='eps')
    plt.close(fig)


def draw_confusion_matrix(cm, categories, normalize=False):
    # Draw confusion matrix
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=[6.4*pow(len(categories), 0.5), 4.8*pow(len(categories), 0.5)])
    ax = fig.add_subplot(111)
    if normalize:
        cm = cm.astype('float') / np.maximum(cm.sum(axis=1)[:, np.newaxis], np.finfo(np.float64).eps)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.get_cmap('Blues'))
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=categories, yticklabels=categories, ylabel='Annotation', xlabel='Prediction')
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    # Loop over data dimensions and create text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha='center', va='center', color='white' if cm[i, j] > thresh else 'black', fontsize=int(20-pow(len(categories), 0.5)))
    fig.tight_layout()
    plt.savefig('output/confusion_matrix.eps', format='eps')
    plt.close(fig)


def metric_accuracy(cm, categories):
    # Compute the accuracy
    correct_samples_class = np.diag(cm).astype(float)
    total_samples_class = np.sum(cm, axis=1).astype(float)
    print('Pixel Acc: %.3f%%' % (np.sum(correct_samples_class) / np.sum(total_samples_class) * 100))
    acc = correct_samples_class / np.maximum(total_samples_class, np.finfo(np.float64).eps)
    print('Mean Acc: %.3f%%' % (acc.mean() * 100))
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
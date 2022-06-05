"""
Functions for detecting error labels using a label order.
"""

import numpy as np
from synthetic_generation import RankingDataset
from sklearn.linear_model import LogisticRegression
from typing import Union, Optional, List, Tuple
import cleanlab.filter as flt
from collections.abc import Callable

def transform_labels(noisy_labels: np.array,
                     true_labels: np.array,
                     order: np.array = None) -> Tuple[list, list, dict]:
    """
    Transform integer labels into their multi-label representation using the given order.
    :param noisy_labels: noisy labels
    :param true_labels: true labels
    :param order: order of the labels to be used for the transformation
    :return: transformed noisy labels, transformed true labels, mapping from integer labels to their multi-label
    representation
    """
    num_classes = len(np.unique(noisy_labels))
    if order is None:
        order = np.arange(len(noisy_labels))
    label_map = {}
    for i in range(num_classes):
        label_map[order[i]] = np.arange(i)
    transformed_noisy = []
    transformed_true = []
    for i in range(noisy_labels.shape[0]):
        transformed_noisy.append(label_map[noisy_labels[i]])
        transformed_true.append(label_map[true_labels[i]])
    return transformed_noisy, transformed_true, label_map


def predict_proba(data: np.array,
                  transformed_labels: list,
                  cv: int = 5,
                  model: Optional[Callable] = LogisticRegression) -> np.array:
    """
    Calculate probabilities of each label belonging to each class for multi-label classification using K-fold
    crossvalidation. The classes are predicted in one-vs-all fashion, so the rows of the resulting matrix do not need to
    sum to 1.
    :param data: array of data points
    :param transformed_labels: list of labels transformed from integer to multi-label representation
    :param cv: number of CV folds
    :param model: model to predict the probabilities
    :return: nxm array A, where A[i, j] denotes the probability that sample i belongs to the class j.
    """
    num_classes = len(np.unique(np.concatenate(transformed_labels)))
    num_samples = len(transformed_labels)
    num_crosses = cv
    cv_sets = np.random.choice(np.arange(num_crosses), size=num_samples, replace=True)
    probas = np.empty(shape=(num_samples, num_classes))
    for cross in np.arange(num_crosses):
        train_ix = np.where(cv_sets != cross)[0]
        test_ix = np.where(cv_sets == cross)[0]
        data_train = data[train_ix, :]
        data_test = data[test_ix, :]

        for classn in np.arange(num_classes):
            class_map = [1 if classn in transformed_labels[i] else 0 for i in range(num_samples)]
            train_class = np.array(class_map)[train_ix]
            clf = model()
            clf.fit(data_train, train_class)
            preds = clf.predict_proba(data_test)
            probas[test_ix, classn] = preds[:, 1]
    return probas


def detect_noisy_labels(data: np.array,
                        noisy_labels: np.array,
                        true_labels: np.array,
                        order: np.array = None,
                        model: Optional[Callable] = LogisticRegression,
                        filter_by: str = "prune_by_noise_rate") -> Tuple[np.array, float, float, float]:
    """
    Detect errors in the labels using the passed order and evaluate the detection against true labels.
    :param data: array of data points
    :param noisy_labels: list of noisy labels
    :param true_labels: list of true labels
    :param order: order to be used for error detection
    :param model: model to predict class probabilities
    :return: Array of error detections, precision, recall and accuracy of error detection
    """
    noisy, true, mapper = transform_labels(noisy_labels, true_labels, order=order)
    probas = predict_proba(data, noisy, model=model)
    wrong_labels = flt.find_label_issues(noisy, probas, multi_label=True, filter_by=filter_by)
    wrong_labels = np.array(wrong_labels, dtype=int)
    noise_mask = np.array([0 if noisy_labels[i] == true_labels[i] else 1 for i in range(len(true_labels))])
    precision = np.sum(np.logical_and(wrong_labels == 1, noise_mask == 1))/len(np.where(wrong_labels == 1)[0])
    recall = np.sum(np.logical_and(wrong_labels == 1, noise_mask == 1))/len(np.where(noise_mask == 1)[0])
    acc = np.sum(wrong_labels == noise_mask)/len(noise_mask)
    return wrong_labels, precision, recall, acc

# def main():
#     rd = RankingDataset()
#     data, true_labels, noisy_labels = rd.generate_clusters()
#     noisy_labels = np.array(noisy_labels)
#     true_labels = np.array(true_labels)
#
#     order = np.arange(len(np.unique(noisy_labels)))
#     model = LogisticRegression
#     errors, precision, recall, acc = detect_noisy_labels(data, noisy_labels, true_labels, order, model)
#
# if __name__ == '__main__':
#     main()

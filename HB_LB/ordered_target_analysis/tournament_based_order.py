"""
Functions for generating label order from the data.
"""
from order_based_detection import transform_labels
from synthetic_generation import RankingDataset
from cleanlab.count import compute_confident_joint, calibrate_confident_joint
import numpy as np
import networkx as nx
from networkx.algorithms import tournament
from networkx import DiGraph
from typing import Tuple
from collections.abc import Callable
from sklearn.linear_model import LogisticRegression
from typing import Callable, Optional

def predict_proba(data: np.array,
                  noisy_labels: list,
                  cv: int = 5,
                  model: Optional[Callable] = LogisticRegression) -> np.array:
    """
    Calculate probabilities of each label belonging to each class for single-label classification using K-fold
    crossvalidation.
    :param data: array of data points
    :param noisy_labels: list of labels
    :param cv: number of CV folds
    :param model: model to predict the probabilities
    :return: nxm array A, where A[i, j] denotes the probability that sample i belongs to the class j.
    """
    num_classes = len(np.unique(noisy_labels))
    num_samples = len(noisy_labels)
    num_crosses = cv
    cv_sets = np.random.choice(np.arange(num_crosses), size=num_samples, replace=True)
    probas = np.empty(shape=(num_samples, num_classes))
    for cross in np.arange(num_crosses):
        train_ix = np.where(cv_sets != cross)[0]
        test_ix = np.where(cv_sets == cross)[0]
        data_train = data[train_ix, :]
        data_test = data[test_ix, :]
        train_class = np.array(noisy_labels)[train_ix]
        clf = model()
        clf.fit(data_train, train_class)
        preds = clf.predict_proba(data_test)
        probas[test_ix, :] = preds

    return probas

def find_tournament_ranking(labels: np.array,
                            probs: np.array,
                            calibrate: bool = False) -> Tuple[DiGraph, list]:
    """
    Find label ranking based on the Hamiltonian path of the label tournament. The tournament is generated such that
    there exists an edge u->v iff. p(u, v) > p(v, u), where p is the joint probability in the confident joint matrix.
    :param labels: noisy labels
    :param probs: label class probabilities
    :param calibrate: whether to calibrate the confident joint matrix
    :return: tournament graph and its Hamiltonian path.
    """
    confident_joint = compute_confident_joint(labels=labels, pred_probs=probs, multi_label=False,
                                              return_indices_of_off_diagonals=False)
    if calibrate:
        confident_joint = calibrate_confident_joint(confident_joint, labels, multi_label=False)
    absolute = np.max(np.abs(confident_joint))
    # Add noise to eliminate draws
    noise = np.random.uniform(low=-absolute / 100, high=absolute / 100,
                              size=(confident_joint.shape[0], confident_joint.shape[1]))

    joint_with_noise = confident_joint + noise
    adjacency_matrix = np.array(joint_with_noise - np.transpose(joint_with_noise) > 0, dtype=int)
    trmnt = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
    hamiltonian_path = tournament.hamiltonian_path(trmnt)
    return trmnt, hamiltonian_path

# def main():
#     rd = RankingDataset(num_dimensions=10)
#     data, true, noisy = rd.generate_clusters()
#     true = np.array(true)
#     noisy = np.array(noisy)
#     probas = predict_proba(data, noisy)
#     tournament, path = find_tournament_ranking(noisy, probas)
#     order = np.arange(len(np.unique(true)))
#     transformed_noisy, transformed_true, _ = transform_labels(noisy, true, order=order)

# if __name__ == '__main__':
#     main()

import numpy as np
from typing import Union, Optional, List, Tuple
from collections.abc import Callable
import numbers
import matplotlib.pyplot as plt

class RankingDataset:
    """
    Synthetic dataset for testing label error detection.
    """

    def __init__(self,
                 num_clusters: numbers.Integral = 5,
                 sizes: Union[list, numbers.Integral] = 50,
                 shape: Optional[Callable] = None,
                 num_dimensions: numbers.Integral = 2,
                 center_spread: numbers.Real = 2,
                 cluster_spreads: Union[list, numbers.Number] = 0.1,
                 label_distortions: Union[list, numbers.Number] = 0.1,
                 label_bias: bool = True):
        """

        :param num_clusters: Number of clusters
        :param sizes: Either a list of sizes of each cluster, or a single number for the same-sized clusters.
        :param shape: If passed, the centers of the clusters are aligned in the space according to the passed transformation.
        :param num_dimensions: The dimension of the space.
        :param center_spread: The standard deviation of the normal distribution for the centers of the clusters.
        :param cluster_spreads: The standard deviation(s) of the normal distribution(s) for the clusters.
        :param label_distortions: The fraction(s) of the labels to be distorted for each cluster.
        :param label_bias: If True, distortion will be biased, so that distortion depends on the cluster distance.
        """

        self.num_clusters = int(num_clusters)
        try:
            self.sizes = [s for s in sizes]
        except TypeError:
            self.sizes = list(np.repeat(sizes, num_clusters))
        assert len(self.sizes) == num_clusters

        try:
            self.spreads = [s for s in cluster_spreads]
        except TypeError:
            self.spreads = list(np.repeat(cluster_spreads, num_clusters))
        assert len(self.spreads) == num_clusters

        try:
            self.distortions = [d for d in label_distortions]
        except TypeError:
            self.distortions = list(np.repeat(label_distortions, num_clusters))
        assert len(self.distortions) == num_clusters

        transformation = shape if shape is not None else lambda x: x
        self.num_dimensions = num_dimensions

        if isinstance(transformation(0), numbers.Number):
            multidimensional = False
        elif len(transformation(0)) == self.num_dimensions:
            multidimensional = True
        else:
            raise TypeError("Wrong dimension of the output of 'shape'. Must be either 1 or 'num_dimensions'")

        if not multidimensional:
            self.transformation = lambda x: list(np.repeat(transformation(x), self.num_dimensions))
        else:
            self.transformation = transformation
        self.current_cluster_ix = 0
        self.center_spread = center_spread
        self.true_labels = None
        self.distorted_labels = None
        self.bias = label_bias

    def generate_clusters(self):
        """
        The main function generating the clusters.
        :return: Dataset with point position, labels without distortion, noisy labels
        """
        self.current_cluster_ix = 0
        centers = self.__find_centers(self.center_spread)
        centers = sorted(centers, key=lambda x: x[0])
        total_size = np.sum(self.sizes)
        dataset = np.empty((total_size, self.num_dimensions))
        i = 0
        true_labels = []
        all_distorted_labels = []
        for center, size, spread in zip(centers, self.sizes, self.spreads):
            cluster, distorted_labels, labels = self.__generate_cluster(center, size, spread)
            dataset[i:(i + cluster.shape[0]), :] = cluster
            true_labels += labels
            all_distorted_labels += distorted_labels
            i += cluster.shape[0]
        perm = np.random.permutation(np.arange(dataset.shape[0]))
        dataset = dataset[perm, :]
        true_labels = list(np.array(true_labels)[perm])
        all_distorted_labels = list(np.array(all_distorted_labels)[perm])
        return dataset, true_labels, all_distorted_labels

    def __find_centers(self,
                       center_spread: numbers.Number) -> List[list]:
        """
        Find centers for the clusters.
        :param center_spread: scale for drawing the centers of the clusters
        :return: Cluster centers with the transformation applied
        """
        x_centers = list(np.random.normal(loc=0, scale=center_spread, size=self.num_clusters))
        transformed_centers = [self.transformation(x) for x in x_centers]
        return transformed_centers

    def __generate_cluster(self,
                           center: list,
                           size: numbers.Integral,
                           spread: numbers.Real) -> Tuple[np.array, list, list]:
        """
        Generate a single cluster with its labels
        :param center: cluster center
        :param size: number of points in the cluster
        :param spread: variance of each cluster. Multivariate normal distribution with cluster independence is applied.
        :return: Cluster points, noisy labels, true labels
        """
        points = np.random.multivariate_normal(mean=center,
                                               cov=spread * np.eye(int(self.num_dimensions)),
                                               size=size)
        labels = list(np.repeat(self.current_cluster_ix, size))
        if not self.bias:
            distorted_labels = self.__distort_cluster_labels(labels, self.distortions[self.current_cluster_ix])
        else:
            distorted_labels = self.__distort_cluster_labels_biased(labels, self.distortions[self.current_cluster_ix])
        self.current_cluster_ix += 1

        return points, distorted_labels, labels

    def __distort_cluster_labels(self,
                                 labels: list,
                                 label_distortion: numbers.Real) -> list:
        """
        Distort the labels of the clusters without bias.
        :param labels: list of true labels
        :param label_distortion: proportion of the labels to be distorted.
        :return: distorted labels
        """
        all_labels = np.arange(self.num_clusters)
        labels_to_choose = np.setdiff1d(all_labels, labels[0])
        len_l = len(labels)
        ixs_to_distort = np.random.choice(np.arange(len_l), int(label_distortion * len_l), replace=False)
        distorted_labels_ixs = np.random.choice(labels_to_choose, len(ixs_to_distort), replace=True)
        labels = np.array(labels)
        labels[ixs_to_distort] = distorted_labels_ixs
        return list(labels)

    def __distort_cluster_labels_biased(self,
                                 labels: list,
                                 label_distortion: numbers.Real) -> list:
        """
        Distort the labels of the clusters with bias
        :param labels: list of true labels
        :param label_distortion: proportion of the labels to be distorted.
        :return: distorted labels
        """
        all_labels = np.arange(self.num_clusters)
        labels_to_choose = np.setdiff1d(all_labels, labels[0])
        len_l = len(labels)
        ixs_to_distort = np.random.choice(np.arange(len_l), int(label_distortion * len_l), replace=False)
        # Distortion represented with drawing probability from a random sample with replacement.
        p = np.abs(labels[0] - labels_to_choose)
        p = np.exp(-p)**2
        p = p/np.sum(p)
        distorted_labels_ixs = np.random.choice(labels_to_choose, len(ixs_to_distort), replace=True, p=p)
        labels = np.array(labels)
        labels[ixs_to_distort] = distorted_labels_ixs
        return list(labels)


def main():
    rd = RankingDataset(num_dimensions=2, sizes=50)
    x1, true, noisy = rd.generate_clusters()
    print(x1)
    plt.scatter(x1[:, 0], x1[:, 1], c=noisy)
    for i in range(x1.shape[0]):
        plt.annotate(noisy[i], (x1[i, 0], x1[i, 1]))
    plt.show()


if __name__ == '__main__':
    main()

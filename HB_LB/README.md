# Errors in text datasets

## Goal

The main of the project was to reproduce the results of error detection methods and to modify them, so that they are applicable to a classification task with an ordered target.

## Description

The results are divided into two parts. The `ordered_target_analysis` directory contains the results corresponding to an implementation and analysis of our custom solution. Specifically:

- `synthetic_generation.py` contains the implementation of synthetic dataset generation, in which the spatial structure of the sample corresponds to the class ordering,
- `ordered_based_detection.py` contains the implementation of our method, i.e., a modification of the CleanLab solution, in which we map the classification target to a multiclass target to include the order information.
- `tournament_based_order.py` contains the implementation of a designed heuristic which aims to infer the class order from the data,
- `compare_orders.py` contains the comparison of various orderings of the target and their impact on error detection accuracy,
- `ordered_target_comparison.ipynb` contains the comparison of our solution with the standard CleanLab error detection pipeline.

The `article_analysis` directory contains results and observations coming from analyzing the two reference articles:

- `reproduce_confident_learning_jair.ipynb` provides code to train a fastText model on the Amazon Reviews dataset, and use it to clean label errors,
- `reproduce_label_errors_neurips.ipynb` provides code to benchmark all the confident learning algorithms on multiple datasets, and compare the results to the ground truth coming from MTurk.

Both notebooks are also available in an `html` version for more accessible reading.

## References

Curtis G. Northcutt, Lu Jiang, Isaac L. Chuang. **Confident Learning: Estimating Uncertainty in Dataset Labels.** *Journal of Artificial Intelligence Research*, 2021. https://arxiv.org/abs/1911.00068

Curtis G. Northcutt, Anish Athalye, Jonas Mueller. **Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks.** *NeurIPS 2021 Datasets and Benchmarks Track*, 2021. https://openreview.net/forum?id=XccDXrDNLek

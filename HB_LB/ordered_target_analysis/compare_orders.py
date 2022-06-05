"""
This file contains preliminary analysis of our hamilton path-based methods for inferring order from the data.
"""

import numpy as np
from synthetic_generation import RankingDataset
from tqdm import tqdm
import pandas as pd
import warnings
from sklearn.ensemble import GradientBoostingClassifier
from tournament_based_order import predict_proba, find_tournament_ranking
from order_based_detection import detect_noisy_labels

def compare_orders(rds, nclust=5):
    data, true, pred = rds
    lr = GradientBoostingClassifier
    probas = predict_proba(data, pred, model=lr)
    pred = np.array(pred)
    random_perm = np.random.permutation(np.arange(nclust))
    my_perm = find_tournament_ranking(pred, probas)[1]
    my_perm = np.array(my_perm)
    inv_perm = my_perm[::-1]
    wl, prec, rec, acc = detect_noisy_labels(data, pred, true, order=random_perm)
    average_results = pd.DataFrame({"perm": [random_perm], "acc": [acc], "rec": [rec], "prec": [prec]})
    print("Random: " + str(acc))
    wl, prec, rec, acc = detect_noisy_labels(data, pred, true, order=my_perm)
    hamiltonian_results = pd.DataFrame({"perm": [my_perm], "acc": [acc], "rec": [rec], "prec": [prec]})
    print("Hamilton: " + str(acc))
    wl, prec, rec, acc = detect_noisy_labels(data, pred, true, order=inv_perm)
    inverse_results = pd.DataFrame({"perm": [inv_perm], "acc": [acc], "rec": [rec], "prec": [prec]})
    print("Inverse: " + str(acc))
    wl, prec, rec, acc = detect_noisy_labels(data, pred, true, order=np.arange(nclust))
    true_results = pd.DataFrame({"perm": [np.arange(nclust)], "acc": [acc], "rec": [rec], "prec": [prec]})
    print("True: " + str(acc))
    return average_results, hamiltonian_results, inverse_results, true_results


def main():
    warnings.filterwarnings("ignore")
    nclust = 10
    rds = [RankingDataset(nclust, sizes=100, cluster_spreads=0.01).generate_clusters() for _ in range(30)]
    avg_res = []
    ham_res = []
    inv_res = []
    tru_res = []
    for i, rd in tqdm(enumerate(rds)):
        print(i)
        av, hm, inv, tru = compare_orders(rd, nclust=nclust)
        avg_res.append(av)
        ham_res.append(hm)
        inv_res.append(inv)
        tru_res.append(tru)
    avg_df = pd.concat(avg_res, ignore_index=True)
    ham_df = pd.concat(ham_res, ignore_index=True)
    inv_df = pd.concat(inv_res, ignore_index=True)
    tru_df = pd.concat(tru_res, ignore_index=True)
    print(avg_df.acc.mean())
    print(ham_df.acc.mean())
    print(inv_df.acc.mean())
    print(tru_df.acc.mean())
    # Aside from the high accuracy increase when using the ground-truth order,
    # we observe no increase in performance when using our hamilton path-based methods
if __name__ == '__main__':
    main()

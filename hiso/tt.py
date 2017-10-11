import numpy as np


def _construct_thresholds(probs, targets, top_k=None):
    assert probs.shape == targets.shape, \
        "The shape of predictions should match the shape of targets."
    nb_samples, nb_labels = targets.shape
    top_k = top_k or nb_labels

    # Sort predicted probabilities in descending order
    idx = np.argsort(probs, axis=1)[:, :-(top_k + 1):-1]
    p_sorted = np.vstack([probs[i, idx[i]] for i in range(len(idx))])
    t_sorted = np.vstack([targets[i, idx[i]] for i in range(len(idx))])
    print(probs)
    print(idx)
    print(p_sorted)
    print(t_sorted)
    # Compute F-1 measures for every possible threshold position
    F1 = []
    TP = np.zeros(nb_samples)
    FN = t_sorted.sum(axis=1)
    FP = np.zeros(nb_samples)
    for i in range(top_k):
        TP += t_sorted[:, i]
        FN -= t_sorted[:, i]
        FP += 1 - t_sorted[:, i]
        F1.append(2 * TP / (2 * TP + FN + FP))
    print(F1)
    F1 = np.vstack(F1).T
    print(F1)
    # Find the thresholds
    row = np.arange(nb_samples)
    col = F1.argmax(axis=1)
    print(col)
    p_sorted = np.hstack([p_sorted, np.zeros(nb_samples)[:, None]])
    print(p_sorted[row, col][:, None])
    ratio = 0.382
    T = (ratio * p_sorted[row, col] +
         (1 - ratio) * p_sorted[row, col + 1])[:, None]

    return T


if __name__ == '__main__':
    a = [1, 2, 3]
    for aa in a:
        aa += 1
    print(a)
    exit()
    probs = np.array([[0.2, 0.1, 0.1, 0.23,
                       0.37], [0.18, 0.12, 0.13, 0.26, 0.37],
                      [0.1, 0.2, 0.05, 0.55, 0.1]])
    gt = np.array([[0, 0, 0, 1, 1], [1, 0, 0, 0, 1], [0, 0, 0, 1, 0]])

    print(_construct_thresholds(probs, gt, top_k=4))

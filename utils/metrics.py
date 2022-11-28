import numpy as np


def dice_score(gt: np.ndarray, pred: np.ndarray):
    """Compute dice across classes. The corresponding labels should be
    previously matched.
    Args:
        gt (np.ndarray): Grounth truth
        pred (np.ndarray): Labels
    Returns:
        list: Dice scores per tissue [CSF, GM, WM]
    """
    gt[gt == 66] == 0
    classes = np.unique(gt[gt != 0]).astype(int)
    dice = np.zeros((len(classes)))
    for i in classes:
        bin_pred = np.where(pred == i, 1, 0)
        bin_gt = np.where(gt == i, 1, 0)
        dice[i-1] = np.sum(bin_pred[bin_gt == 1]) * 2.0 / (np.sum(bin_pred) + np.sum(bin_gt))
    return dice.tolist()

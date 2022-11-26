import numpy as np


def reconstruct_volume_from_tabular(
    preds_categorical: np.ndarray, brain_mask: np.ndarray, shape: tuple
) -> np.ndarray:
    brain_mask = brain_mask.flatten()
    predict_volume = brain_mask.copy()
    predict_volume[brain_mask == 255] = preds_categorical + 1
    predict_volume = predict_volume.reshape(shape)
    return predict_volume


def match_labels(seg: np.ndarray, gt: np.ndarray, prob_array: np.ndarray = None):
    """
    Matches the labels numbers based on the counts of voxels inside the masks
        deifned by gt labels
    Args:
        seg (np.ndarray): segmentation results from em
        gt (np.ndarray): gt labels
        prob_array (np.ndarray, optional): posterior probs array used inside em.
            Defaults to None.
    """
    shape = seg.shape
    seg = seg.flatten()
    gt = gt.flatten()
    order = {}
    for label in [0, 2, 3]:
        labels, counts = np.unique(seg[gt == label], return_counts=True)
        order[label] = labels[np.argmax(counts)]
    order[1] = [i for i in [0, 1, 2, 3] if i not in list(order.values())][0]
    seg_ = seg.copy()
    prob_array_ = prob_array.copy() if prob_array is not None else None
    for des_val, seg_val in order.items():
        seg[seg_ == seg_val] = des_val
        if des_val in [1, 2, 3]:
            if prob_array_ is not None:
                prob_array[:, des_val-1] = prob_array_[:, seg_val-1]
    if prob_array is not None:
        return seg.reshape(shape), prob_array
    else:
        return seg.reshape(shape)

from typing import List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import SimpleITK as sitk
import pandas as pd


def plot_dice(df: pd.DataFrame):
    plt.figure(figsize=(10, 4))
    plt.title('Dice score across patients grouped by tissue and experient', fontsize=14)
    sns.boxplot(data=df, x="tissue", y="dice", hue="experiment_name")
    sns.despine()
    plt.grid(axis='y')
    plt.ylim([0, 1.1])
    plt.xlabel('Tissues', fontsize=12)
    plt.ylabel('Dice Score', fontsize=12)
    plt.legend(loc='lower right', title='Models', fontsize=10)
    plt.show(block=False)
    plt.pause(0.01)


def brains_figure(
    cases: List[str], data_path: Path, segs_path: Path, exp_names: List[str],
    out_path: Path = None, slice_n: int = 125
):
    # TODO: Fix this docstring
    """
    Plot all segementations in plots of 7xn_cases. First row is t1 and second
    ground truth
    Args:
        img_path (Path): t1 path
        segs_path (Path): directory containing all segementations
        cases (List[str]): list of case names to use in the plot
        exp_names (List[str]): list of exp_names to include
        slice_n (int, optional): Axial slice to plot. Defaults to 25.
    """
    n_figures = int(np.ceil(len(exp_names) / 5))
    for n in range(n_figures):
        sublist_exp = exp_names[n*5:(n+1)*5]
        sublist_seg_p = segs_path[n*5:(n+1)*5]
        n_rows, n_cols = 2+len(sublist_exp), len(cases)
        _, ax = plt.subplots(n_rows, n_cols, figsize=(13, 3*(len(sublist_exp)+2)))
        for i, name in enumerate(['T1', 'GT']):
            for j, case in enumerate(cases):
                cmap = 'gray' if (i == 0) else 'viridis'
                img_name = f'{case}.nii.gz' if i == 0 else f'{case}_3C.nii.gz'
                img = sitk.ReadImage(str(data_path / case / img_name))
                img_array = sitk.GetArrayFromImage(img)
                if i == 0:
                    ax[i][j].set_title(case, fontsize=14)
                if (j == 0):
                    y_label = name
                    ax[i][j].set_ylabel(y_label, fontsize=14)
                ax[i][j].imshow(img_array[slice_n, :, :], cmap=cmap)
                ax[i][j].set_xticks([])
                ax[i][j].set_yticks([])
        for i, (name, seg_path) in enumerate(zip(sublist_exp, sublist_seg_p), 2):
            for j, case in enumerate(cases):
                img_name = f'{case}.nii.gz'
                img = sitk.ReadImage(str(seg_path / img_name))
                img_array = sitk.GetArrayFromImage(img)
                if (j == 0):
                    y_label = name
                    ax[i][j].set_ylabel(y_label, fontsize=14)
                ax[i][j].imshow(img_array[slice_n, :, :], cmap='viridis')
                ax[i][j].set_xticks([])
                ax[i][j].set_yticks([])
                if i == n_rows:
                    ax[i][j].set_xlabel(case, fontsize=14)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.01)
        if out_path is not None:
            plt.savefig(out_path/f'brains_{n}.svg', bbox_inches='tight', format='svg')

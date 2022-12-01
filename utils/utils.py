import numpy as np
import SimpleITK as sitk
from pathlib import Path
import pandas as pd


def extract_metadata(img: sitk.Image) -> dict:
    header = {
        'direction': img.GetDirection(),
        'origin': img.GetOrigin(),
        'spacing': img.GetSpacing(),
        'metadata': {}
    }
    for key in img.GetMetaDataKeys():
        header['metadata'][key] = img.GetMetaData(key)
    return header


def save_img_from_array_using_referece(
    volume: np.ndarray, reference: sitk.Image, filepath: Path
) -> None:
    """Stores the volume in nifty format using the spatial parameters coming
        from a reference image
    Args:
        volume (np.ndarray): Volume to store as in Nifty format
        reference (sitk.Image): Reference image to get the spatial parameters from.
        filepath (Path): Where to save the volume.
    """
    # Save image
    if (type(volume) == list) or (len(volume.shape) > 3):
        if type(volume[0]) == sitk.Image:
            vol_list = [vol for vol in volume]
        else:
            vol_list = [sitk.GetImageFromArray(vol) for vol in volume]
        joiner = sitk.JoinSeriesImageFilter()
        img = joiner.Execute(*vol_list)
    else:
        img = sitk.GetImageFromArray(volume)
    img.SetDirection(reference.GetDirection())
    img.SetOrigin(reference.GetOrigin())
    img.SetSpacing(reference.GetSpacing())
    for key in reference.GetMetaDataKeys():
        img.SetMetaData(key, reference.GetMetaData(key))
    sitk.WriteImage(img, str(filepath))


def save_segmentations(
    volume: np.ndarray, metadata: dict, filepath: Path
) -> None:
    """Stores the volume in nifty format using the spatial parameters coming
        from a reference image
    Args:
        volume (np.ndarray): Volume to store as in Nifty format
        metadata (dict): Metadata from the reference image to store the volumetric image.
        filepath (Path): Where to save the volume.
    """
    # Save image
    if (type(volume) == list) or (len(volume.shape) > 3):
        if type(volume[0]) == sitk.Image:
            vol_list = [vol for vol in volume]
        else:
            vol_list = [sitk.GetImageFromArray(vol) for vol in volume]
        joiner = sitk.JoinSeriesImageFilter()
        img = joiner.Execute(*vol_list)
    else:
        img = sitk.GetImageFromArray(volume)
    img.SetDirection(metadata['direction'])
    img.SetOrigin(metadata['origin'])
    img.SetSpacing(metadata['spacing'])
    for key, val in metadata['metadata'].items():
        img.SetMetaData(key, val)
    sitk.WriteImage(img, filepath)


def translate_results_dict_to_df(res_json: dict) -> pd.DataFrame:
    """Translates the dictionary of results prepared to store in json format
    to a pandas df ready to store as csv table.
    Args:
        res_json (dict): Dictionary with experiment results.
            example: {
                'experiment_name': 'bla',
                'algorithm': 'bla',
                'initialization': 'bla',
                'case_0': {'n_iters': 0, 'time': 0,
                           'dice': {'csf': 0, 'wm': 0, 'gm':0}}}
            }
    Returns:
        pd.DataFrame
    """
    macro_columns = ['experiment_name', 'algorithm', 'initialization']
    micro_columns = ['n_iters', 'time']
    dice_columns = ['csf', 'gm', 'wm']
    results_csv = []
    base_row = [res_json[key] for key in macro_columns]
    for case, val in res_json.items():
        if case in macro_columns:
            continue
        row = base_row + [case] + [val[key] for key in micro_columns]
        row = row + [val['dice'][key] for key in dice_columns]
        results_csv.append(row)
    return pd.DataFrame(results_csv, columns=macro_columns + ['id'] + micro_columns + dice_columns)

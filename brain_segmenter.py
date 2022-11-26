from pathlib import Path
import numpy as np
import time
import logging
import pickle
import argparse

import SimpleITK as sitk

from models.em import ExpectationMaximization
from utils import utils
from utils.metrics import dice_score
from postprocessing.postprocessing import match_labels, reconstruct_volume_from_tabular
from preprocessing.preprocessing import Preprocessor

logging.basicConfig(level=logging.INFO, format="%message")


def main():
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ip", dest="input_path", help="Path to the nii file to process", required=True)
    parser.add_argument(
        "--chkpt", dest="checkpt_path", help="Path to the checkpt file to use", required=True)
    parser.add_argument(
        "--op", dest="ouput_path", help="Directory where to store the result", required=True)
    # parser.add_argument(
    #     '--v', dest='verbose', action='store_true', help='Wheter to print process info or not')
    args = parser.parse_args()

    # Load checkpoint file and get configurations
    chkpt_path = Path(args.checkpt_path)
    with open(chkpt_path, "rb") as pklfile:
        chkpt = pickle.load(pklfile)
    tissue_models = chkpt['tissue_models']
    cfg = chkpt['cfg']

    # Define the preprocessors
    t1_preprocessor = Preprocessor(
        normalization_cfg=cfg['data']['normalization_cfg'],
        skull_stripping=cfg['data']['skull_stripping'],
        resize_cfg=cfg['data']['resize_cfg'],
        register_atlases=False,
        mni_atlas=None,
        mv_atlas=None,
        tissue_models=tissue_models
    )
    if cfg['data']['resize_cfg'] is not None:
        resize_cfg_mask = cfg['data']['resize_cfg'].copy()
        resize_cfg_mask['interpolation'] = 'NearestNeighbour'
    else:
        resize_cfg_mask = None

    label_preprocessor = Preprocessor(
        normalization_cfg=None,
        skull_stripping=None,
        resize_cfg=resize_cfg_mask,
        register_atlases=False,
        mni_atlas=None,
        mv_atlas=None,
        tissue_models=None
    )

    logging.info('Loading image')
    # Read the images and preprocess them
    img_path = Path(args.input_path)
    img_name = img_path.name.rstrip('.nii.gz')
    img_path = img_path.parent

    # Load image, labels and brain mask
    t1 = sitk.ReadImage(str(img_path / f'{img_name}.nii.gz'))
    ref_metadata = utils.extract_metadata(t1)
    t1 = sitk.GetArrayFromImage(t1)

    if (img_path / f'{img_name}_3C.nii.gz').exists():
        ground_truth = sitk.GetArrayFromImage(
                sitk.ReadImage(str(img_path / f'{img_name}_3C.nii.gz'))
            ).astype('int')
        ground_truth, _, _, _ = label_preprocessor.preprocess(ground_truth)
    else:
        ground_truth = None

    brain_mask = sitk.GetArrayFromImage(
        sitk.ReadImage(str(img_path / f'{img_name}_1C.nii.gz'))
    )
    brain_mask = np.where(brain_mask != 0, 255, 0).astype('uint8')

    # Preprocess
    logging.info('Preprocessing')
    t1, _, _, tissue_models_labels = t1_preprocessor.preprocess(t1, brain_mask)
    brain_mask, _, _, _ = label_preprocessor.preprocess(brain_mask)

    # Load atlases
    load_atlases = cfg['model']['initialization'] in ['mni_atlas', 'mv_atlas']
    load_atlases = load_atlases or (cfg['model']['use_atlas_in_em'] is not None)
    if load_atlases:
        tpm_mni = sitk.GetArrayFromImage(sitk.ReadImage(
            str(img_path / f'{img_name}_mni_atlas.nii.gz')
        ))
        tpm_mni = np.clip(tpm_mni, a_min=0, a_max=1)
        tpm_mv = sitk.GetArrayFromImage(sitk.ReadImage(
            str(img_path / f'{img_name}_mv_atlas.nii.gz')
        ))
        tpm_mv = np.clip(tpm_mv, 0, 1)

    # Define output path
    ouput_path = Path(args.ouput_path)
    ouput_path.mkdir(exist_ok=True, parents=True)
    logging.info(f'Results will be saved in: {ouput_path}')

    # Run model over the complete dataset
    t1_vector = t1[brain_mask == 255].flatten()
    data = np.array(t1_vector)[:, np.newaxis]

    # Reformat the atlas probability maps to tabular form
    atlas_in_em = cfg['model']['which_atlas_in_em']
    if ('atlas' in cfg['model']['initialization']) or (atlas_in_em is not None):
        if (cfg['model']['initialization'] == 'mv_atlas') or ('mv' in atlas_in_em):
            atlas_map_vector = tpm_mv[:, brain_mask == 255]
        else:
            atlas_map_vector = tpm_mni[:, brain_mask == 255]
        atlas_map_vector = atlas_map_vector.reshape(atlas_map_vector.shape[0], -1)
        # Discard background classs values
        atlas_map_vector = atlas_map_vector[1:, :]
    else:
        atlas_map_vector = None

    # Define model
    logging.info('Loading checkpoint model')
    model = ExpectationMaximization(
        n_components=cfg['model']['n_components'],
        mean_init=cfg['model']['initialization'],
        max_iter=cfg['model']['n_iterations'],
        change_tol=cfg['model']['tolerance'],
        verbose=cfg['model']['verbose'],
        plot_rate=cfg['model']['plot_rate'],
        tissue_models=tissue_models,
        atlas_use=cfg['model']['use_atlas_in_em'],
        atlas_map=atlas_map_vector
    )

    model.means = chkpt[img_name]['means']
    model.sigmas = chkpt[img_name]['cov_mat']
    model.priors = chkpt[img_name]['priors']
    model.training = False
    model.fitted = True

    # Run model
    logging.info('Segmenting')
    start_s = time.time()
    _, preds_categorical = model.predict(data)
    pred_time = time.time() - start_s

    # Posptrocess
    logging.info('Postprocessing')
    # Reshape results
    predict_volume = reconstruct_volume_from_tabular(
        preds_categorical, brain_mask, t1.shape)
    # Match labels if initialization was kmeans
    if ground_truth is not None:
        if (cfg['model']['initialization'] == 'kmeans'):
            predict_volume = match_labels(predict_volume, ground_truth)
        # Compute metrics
        dice = dice_score(ground_truth, predict_volume)

    logging.info('Saving segmentation')
    # Save resulting segmentation
    utils.save_segmentations(
        predict_volume, ref_metadata, str(ouput_path / f"{img_name}_seg.nii.gz")
    )

    total_time = time.time() - start
    logging.info(f'Segmentation finished - Seg-Time: {pred_time} - Total time {total_time}')
    if ground_truth is not None:
        logging.info(f'Achieved Dice - CSF:{dice[0]} - WM: {dice[1]} - GM: {dice[2]}')


if __name__ == '__main__':
    main()

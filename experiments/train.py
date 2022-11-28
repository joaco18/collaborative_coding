from dataset.dataset import MedvisionDataset
from pathlib import Path
from tqdm import tqdm
import numpy as np
import yaml
import time
import logging
import json
import pickle

from models.em import ExpectationMaximization
from utils import utils
from utils.metrics import dice_score
from postprocessing.postprocessing import match_labels, reconstruct_volume_from_tabular

logging.basicConfig(level=logging.INFO, format="%message")


def main():
    # Set path relative to this file
    experiments_path = Path('__file__').resolve().parent / 'experiments'

    # Load configurations file
    cfg_path = experiments_path / 'train_config.yaml'
    with open(cfg_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    load_atlases = cfg['model']['initialization'] in ['mni_atlas', 'mv_atlas']
    load_atlases = load_atlases or (cfg['model']['use_atlas_in_em'] is not None)
    # Generate train dataset
    train_dataset = MedvisionDataset(
        datapath=Path(cfg['data']['datapath']),
        tissue_models_filepath=Path(cfg['data']['tissue_models_filepath']),
        modalities=cfg['data']['modalities'],
        pathologies=cfg['data']['pathologies'],
        partitions=['train'],
        case_selection=cfg['data']['case_selection'],
        load_atlases=load_atlases,
        normalization_cfg=cfg['data']['normalization_cfg'].copy(),
        skull_stripping=cfg['data']['skull_stripping'],
        resize_cfg=cfg['data']['resize_cfg'],
    )

    # Define container for results and checkpoint
    experiment_name = cfg['results']['experiment_name']
    results = {
        'experiment_name': experiment_name,
        'algorithm': 'Expectation Maximization',
        'initialization': cfg['model']['initialization'],
    }
    checkp = results.copy()
    # Define output path
    results_path = Path(cfg['results']['results_path']) / experiment_name
    if cfg['results']['use_time_in_log']:
        results_path = results_path / int(time.time())
    results_path.mkdir(exist_ok=True, parents=True)
    logging.info(f'Results will be saved in: {results_path}')

    # Run model over the complete dataset
    logging.info('Starting runs on training set...')
    for case_idx in tqdm(range(len(train_dataset))):
        case = train_dataset[case_idx]
        # Only classify brain region pixels
        brain_mask = case['brain_mask']

        # Reformat the image data as tabular one:
        #    [samples (voxels), features(intensities)]
        t1_vector = case['t1'][brain_mask == 255].flatten()
        data = np.array(t1_vector)[:, np.newaxis]

        # Reformat the atlas probability maps to tabular form
        atlas_in_em = cfg['model']['which_atlas_in_em']
        condition = (atlas_in_em is not None) and (cfg['model']['use_atlas_in_em'] is not None)
        if ('atlas' in cfg['model']['initialization']) or condition:
            if (cfg['model']['initialization'] == 'mv_atlas') or ('mv' in atlas_in_em):
                atlas_map_vector = case['tpm_mv'][:, brain_mask == 255]
            else:
                atlas_map_vector = case['tpm_mni'][:, brain_mask == 255]
            atlas_map_vector = atlas_map_vector.reshape(atlas_map_vector.shape[0], -1)
            # Discard background classs values
            atlas_map_vector = atlas_map_vector[1:, :]
        else:
            atlas_map_vector = None

        # Define model
        model = ExpectationMaximization(
            n_components=cfg['model']['n_components'],
            mean_init=cfg['model']['initialization'],
            max_iter=cfg['model']['n_iterations'],
            change_tol=cfg['model']['tolerance'],
            verbose=cfg['model']['verbose'],
            plot_rate=cfg['model']['plot_rate'],
            tissue_models=train_dataset.tissue_models,
            atlas_use=cfg['model']['use_atlas_in_em'],
            atlas_map=atlas_map_vector
        )

        # Run model
        start = time.time()
        _, preds_categorical = model.fit_predict(data)
        pred_time = time.time() - start

        # Posptrocess
        # Reshape results
        predict_volume = reconstruct_volume_from_tabular(
            preds_categorical, brain_mask, case['t1'].shape)
        # Match labels if initialization was kmeans
        if cfg['model']['initialization'] == 'kmeans':
            predict_volume = match_labels(predict_volume, case['ground_truth'])

        # Compute metrics
        dice = dice_score(case['ground_truth'], predict_volume)

        # Store results
        results[str(case['id'])] = {
            'n_iters': model.n_iter_,
            'time': pred_time,
            'dice': {'csf': dice[0], 'wm': dice[1], 'gm': dice[2]},
            'means': model.means,
            'cov_mat': model.sigmas,
            'priors': model.priors
        }

        # Save resulting segmentation
        if cfg['results']['save_segmentations']:
            segmentations_path = results_path / 'segmentations'
            segmentations_path.mkdir(exist_ok=True, parents=True)
            utils.save_segmentations(
                predict_volume, case['ref_metadata'],
                str(segmentations_path / f"{case['id']}.nii.gz")
            )

        # Save checkpoints (contains means, cov, and priors) and save readable resutls.
        # Save step by step for safety in case the experiment goes down
        with open(str(results_path / 'checkpoint.pkl'), 'wb') as pkl_file:
            checkp.update({str(case['id']): results[str(case['id'])].copy()})
            pickle.dump(checkp, pkl_file)
        with open(str(results_path / 'results.json'), 'w') as json_file:
            for key in ['means', 'cov_mat', 'priors']:
                del results[str(case['id'])][key]
            json.dump(results, json_file, sort_keys=True, indent=4, separators=(',', ': '))

    # In case a csv with dice results is required translate json file into a csv file
    results_csv = utils.translate_results_dict_to_df(results)
    if cfg['results']['save_results_csv']:
        results_csv.to_csv(results_path/'results.csv')

    # Store the configuration info
    # For readability
    with open(results_path / 'test_config.yaml', "w") as ymlfile:
        yaml.safe_dump(cfg, ymlfile)

    # For fast pipeline configuration
    with open(str(results_path / 'checkpoint.pkl'), 'wb') as pkl_file:
        update = {'cfg': cfg, 'tissue_models': train_dataset.tissue_models}
        checkp.update(update)
        pickle.dump(checkp, pkl_file)

    logging.info('Experiment finished!')
    logging.info(f'    Mean dice CSF: {np.around(results_csv.csf.mean(), 3)}')
    logging.info(f'    Mean dice WM: {np.around(results_csv.wm.mean(), 3)}')
    logging.info(f'    Mean dice GM: {np.around(results_csv.gm.mean(), 3)}')


if __name__ == '__main__':
    main()

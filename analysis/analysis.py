import argparse
from pathlib import Path
import pandas as pd
from utils import plots
import matplotlib.pyplot as plt


def main():
    data_path = Path('__file__').resolve().parent / 'data'

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rf", dest="results_folder", help="Directory where to experiments are stored")
    parser.add_argument(
        "--dp", dest="data_path", help="Path to data directory", default=data_path)
    parser.add_argument(
        "--op", dest="ouput_path", help="Directory where to store the result")
    parser.add_argument(
        '--exp', dest='experiments', nargs='*', help='List of experiments to analyze')
    parser.add_argument(
        '--cases', dest='cases', nargs='*', help='List of 5 cases to plot the brains')
    args = parser.parse_args()

    savepath = Path(args.ouput_path)
    savepath.mkdir(exist_ok=True, parents=True)
    data_path = Path(args.data_path)
    experiments_path = Path(args.results_folder)
    data_path = data_path / ('test_set' if 'test' in str(experiments_path) else 'train_set')
    exp_names = args.experiments
    cases = args.cases

    segs_path = [experiments_path / exp / 'segmentations' for exp in exp_names]
    any_exp = False

    # Get results from experiments
    dices = []
    for experiment_path in experiments_path.iterdir():
        if experiment_path.name not in exp_names:
            continue
        any_exp = True
        dices.append(pd.read_csv(experiment_path / 'results.csv', index_col=0))

        dice_scores = pd.concat(dices, ignore_index=True)

    if any_exp:
        # plot dice scores
        df = pd.melt(
            dice_scores,
            id_vars=['experiment_name', 'algorithm', 'initialization', 'n_iters', 'time'],
            value_vars=['csf', 'gm', 'wm'], ignore_index=False
        )
        df.columns = [
            'experiment_name', 'algorithm', 'initialization', 'n_iters', 'time', 'tissue', 'dice'
        ]
        plots.plot_dice(df)
        plt.savefig(savepath/'dices.svg', bbox_inches='tight', format='svg')

        # plot brain segementations figure
        plots.brains_figure(cases, data_path, segs_path, exp_names, savepath)
        plt.show()


if __name__ == '__main__':
    main()

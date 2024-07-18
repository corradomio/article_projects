from collections import defaultdict
import numpy as np
from stdlib import jsonx
from sklearnx.metrics import weighted_absolute_percentage_error


# Model statistics


def main():
    n_timeseries = 0
    algo_stats = defaultdict(lambda: 0)
    nfh_stats = defaultdict(lambda: 0)
    name_stats = defaultdict(lambda: 0)
    best_wape = 10
    worst_wape = 0
    best_algo = None
    worst_algo = None

    best_models = jsonx.load("best_models.json")

    y_dataset_true = np.zeros(12)
    y_dataset_pred = np.zeros(12)

    for area_item in best_models:
        # print(area_item)
        best_model_info = best_models[area_item]

        # 'y_true'
        # 'y_pred'
        # 'algo'
        # 'wape'
        # 'worst'
        # 'nfh'
        y_true = np.array(best_model_info['y_true'])
        y_pred = np.array(best_model_info['y_pred'])
        if len(y_pred) == 0:
            print(f"ERROR: {area_item} with NO predictions")
            y_pred = np.zeros(12)

        y_dataset_true += y_true
        y_dataset_pred += y_pred

        algo = best_model_info['algo']
        wape = best_model_info['wape']
        nfh = best_model_info['nfh']
        name = best_model_info['name']

        if wape < best_wape:
            best_wape = wape
            best_algo = algo
        if wape > worst_wape:
            worst_wape = wape
            worst_algo = algo

        n_timeseries += 1
        algo_stats[algo] += 1
        name_stats[name] += 1
        nfh_stats[nfh] += 1
        pass

    dataset_wape = weighted_absolute_percentage_error(y_dataset_true, y_dataset_pred)

    jsonx.dump({
        'n_timeseries': n_timeseries,
        'algo_stats': algo_stats,
        'name_stats': name_stats,
        'nfh_stats': nfh_stats,
        'best_algo': {
            'algo': best_algo,
            'wape': best_wape
        },
        'worst_algo': {
            'algo': worst_algo,
            'wape': worst_wape
        },
        'dataset_wape': dataset_wape,
        'prediction_quality': 1 - dataset_wape
    }, "models_stats.json")
    pass


if __name__ == "__main__":
    main()


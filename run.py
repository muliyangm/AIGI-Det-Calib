# @2025 APR Lab, Your Deepfake Detector Can Secretly Achieve SOTA Accuracy, If Calibrated.
# Muli Yang, Gabriel James Goenawan, Henan Wang, Huaiyuan Qin, Chenghao Xu, Yanhua Yang, Fen Fang, Ying Sun, Joo Hwee Lim, Hongyuan Zhu 

import json
import torch
import argparse
import numpy as np
from typing import Callable, Dict, List, Optional
from src.calibration import find_best_threshold, supervised_calibration, unsupervised_calibration
from src.metric import calc_acc, avg

bench = {'chameleon': ['chameleon'],
         'aigcdetectionbenchmark': ['progan', 'stylegan', 'biggan', 'cyclegan', 'stargan', 'gaugan',
                                    'stylegan2', 'whichfaceisreal', 'ADM', 'Glide', 'Midjourney',
                                    'stable_diffusion_v_1_4', 'stable_diffusion_v_1_5', 'VQDM',
                                    'wukong', 'DALLE2'],
         'genimage': ['Midjourney', 'stable_diffusion_v_1_4', 'stable_diffusion_v_1_5', 'ADM', 'Glide',
                      'wukong', 'VQDM', 'biggan']}

def balanced_sample(label: np.ndarray,
                    n_shots: int = 20):
    # balanced real-fake sampler.
    fake_sample, real_sample = n_shots // 2, n_shots // 2

    one_index = np.where(label == 1)[0]
    np.random.shuffle(one_index)
    one_index = one_index[:fake_sample]

    zero_index = np.where(label == 0)[0]
    np.random.shuffle(zero_index)
    zero_index = zero_index[:real_sample]

    sample_indices = torch.cat([torch.from_numpy(one_index), torch.from_numpy(zero_index)])
    return sample_indices


def run_n_experiment(label : torch.Tensor, 
                     logits : torch.Tensor, 
                     methods : List[Callable], 
                     confs: Optional[List[Dict]],
                     n_shot: int = 50,
                     n_exp: int = 20,
                     ):
    assert len(label) == len(logits), 'Label and Logits need to have the same length.'
    assert len(label) >= n_shot, 'Label and Logits length must be greater than n_shot'
    if confs is not None:
        assert len(methods) == len(confs), 'confs should be a list of the same length as methods or None.'
    else:
        confs = [{} for _ in methods]

    all_thresholds = [[] for _i in methods]
    all_accuracies = [[] for _i in methods]

    for _i in range(n_exp):
        # balance sampler
        sampled_indices = balanced_sample(label.numpy())
        sampled_label = label[sampled_indices]
        sampled_logits = logits[sampled_indices]

        for n, (m, c) in enumerate(zip(methods, confs)):
            threshold = m(label=sampled_label,
                          logit=sampled_logits,
                          **c)
            all_thresholds[n].append(threshold)
            all_accuracies[n].append(calc_acc(label, logits + threshold))
    
    res = []
    for n in range(len(methods)):
        res.append({'accuracies' : all_accuracies[n],
                    'thresholds' : all_thresholds[n],
                    'threshold_avg' : avg(all_thresholds[n]),
                    'accuracy_avg' : calc_acc(label, logits + avg(all_thresholds[n])),})
    return res
        
def parse_args():
    parser = argparse.ArgumentParser(description="Script for running experiments with logits.")

    parser.add_argument(
        "--n_shots",
        type=int,
        default=100,
        help="Number of shots."
    )

    parser.add_argument(
        "--n_exp",
        type=int,
        default=20,
        help="Number of experimental runs to perform."
    )

    parser.add_argument(
        "--logits_json",
        type=str,
        default='./logits/CNNSpot_progan_trained.json',
        help="Path to the JSON file containing logits."
    )

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print(f'''Configuration: 
N-shot   : {args.n_shots}
N-exp.   : {args.n_exp}
====================''')
    
    methods = [supervised_calibration, unsupervised_calibration]
    confs = None

    with open(args.logits_json, 'r') as f:
        d = json.load(f)
    
    res_per_ds = {}
    original_acc_per_ds = {}

    cnt = 0
    for ds in d:
        y_true = torch.tensor(d[ds]['y_true'])
        y_pred = torch.tensor(d[ds]['y_pred'])
        calib_res = run_n_experiment(label=y_true,
                               logits=y_pred,
                               methods=methods,
                               confs=confs,
                               n_shot=args.n_shots,
                               n_exp=args.n_exp,)

        res_per_ds[ds] = calib_res
        original_acc_per_ds[ds] = calc_acc(y_true, y_pred) * 100
        
        print(f'''Dataset      : {ds}
Original     : {original_acc_per_ds[ds]:.2f}    
Supervised   : {calib_res[0]['accuracy_avg'] * 100 :.2f} (±{(np.array(calib_res[0]['accuracies']) * 100).std().item():.2f})
Unsupervised : {calib_res[1]['accuracy_avg'] * 100 :.2f} (±{(np.array(calib_res[1]['accuracies']) * 100).std().item():.2f})
====================''')
        
    print('''========================================
Benchmarks
========================================''')

    for b, dss in bench.items():

        original_acc = avg([original_acc_per_ds[_d] for _d in dss])

        average_acc_supervised = avg([res_per_ds[_d][0]['accuracy_avg'] for _d in dss]) * 100
        accuracies_per_exp_supervised = np.array([res_per_ds[_d][0]['accuracies'] for _d in dss]) * 100 # n_ds, n_exp
        average_std_supervised = accuracies_per_exp_supervised.mean(1).std(0)

        average_acc_unsupervised = avg([res_per_ds[_d][1]['accuracy_avg'] for _d in dss]) * 100
        accuracies_per_exp_unsupervised = np.array([res_per_ds[_d][1]['accuracies'] for _d in dss]) * 100 # n_ds, n_exp
        average_std_unsupervised = accuracies_per_exp_unsupervised.mean(1).std(0)

        print(f'''Benchmark    : {b}
Original     : {original_acc:.2f}
Supervised   : {average_acc_supervised:.2f} (±{average_std_supervised:.2f})
Unsupervised : {average_acc_unsupervised:.2f} (±{average_std_unsupervised:.2f})
========================================''')
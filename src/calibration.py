# @2025 APR Lab, Your Deepfake Detector Can Secretly Achieve SOTA Accuracy, If Calibrated.
# Muli Yang, Gabriel James Goenawan, Henan Wang, Huaiyuan Qin, Chenghao Xu, Yanhua Yang, Fen Fang, Ying Sun, Joo Hwee Lim, Hongyuan Zhu 

import torch
import numpy as np
from scipy.stats import gaussian_kde
from scipy.optimize import minimize_scalar

from .metric import calc_acc

def find_best_threshold(label : torch.Tensor, 
                       logit : torch.Tensor, 
                       eps : float = 5e-5, 
                       init_limits =[-100, 100],
                       max_step=5,
                       allow_invert = True,):
    # iterative algorithm to find best theoritical accuracy threshold.

    def _find_best_accuracy(label, logit, eps, init_limits, max_step):
        ll = init_limits[0]
        rl =  init_limits[1]

        cnt = 0
        while rl-ll > eps:
            l_acc = calc_acc(label, logit + ll)
            r_acc = calc_acc(label, logit + rl)
    
            if l_acc == 0.5 or r_acc == 0.5: # just in case real the distributions isn't discovered yet
                if l_acc == 0.5:
                    ll += max_step
                if r_acc == 0.5:
                    rl -= max_step
            
            elif l_acc >= r_acc:
                rl -= min(((rl - (rl + ll) / 2))/2 , max_step)
            else:
                ll += min((((rl + ll) / 2) - ll)/2, max_step)
            cnt += 1

        return calc_acc(label, logit + ((rl+ll) / 2)), {'offset' : (rl+ll) / 2}
    
    if isinstance(label, np.ndarray):
        label = torch.from_numpy(label)
    if isinstance(logit, np.ndarray):
        logit = torch.from_numpy(logit)

    accuracy_normal = _find_best_accuracy(label, logit, eps, init_limits, max_step)
    if allow_invert:
        accuracy_inverse = _find_best_accuracy(label, logit * -1, eps, init_limits, max_step)
        
        if accuracy_normal[0] < accuracy_inverse[0]:
            result = accuracy_inverse[1]
            result['scale'] = -1
        else:
            result = accuracy_normal[1]
            result['scale'] = 1
    else:
        result = accuracy_normal[1]
        result['scale'] = 1
    return result

def unsupervised_calibration(label: torch.Tensor, 
                            logit: torch.Tensor, 
                            real_ratio: float= 0.5, 
                            **kwargs):
    # Convert to numpy
    label = label.numpy()
    logit = logit.numpy()

    # Define KDE
    x_vals = np.linspace(min(logit), max(logit), int((max(logit) - min(logit)) * 10))
    kde = gaussian_kde(logit)
    kde_vals = kde(x_vals)
    prob_dist = kde_vals / kde_vals.sum()

    def objective(threshold):
        weight = x_vals - threshold

        weight[weight < 0] *= (1-real_ratio)
        weight[weight > 0] *= real_ratio

        return abs((weight * prob_dist).sum())

    # Optimize threshold to minimize the imbalance
    result = minimize_scalar(objective, 
                             bounds=(min(logit), max(logit)), 
                             method='bounded',
                             options={'xatol':1e-12},)
    return -result.x

def supervised_calibration(label: torch.Tensor, 
                           logit: torch.Tensor, 
                           **kwargs):
    # accuracy-maximization, scipy-opt method
    # PD Estimation -> iterative.
    label = label.numpy()
    logit = logit.numpy()
    
    kde_0 = gaussian_kde(logit[label==0])
    kde_1 = gaussian_kde(logit[label==1])
    
    # define F(T) (we’ll maximize F by minimizing -F)
    def negF(T):
        cdf0 = kde_0.integrate_box_1d(-np.inf, T)      # CDF₀(T)
        cdf1 = kde_1.integrate_box_1d(-np.inf, T)      # CDF₁(T)
        return -( cdf0 + (1 - cdf1) )                  # -(F(T))

    # bounded scalar optimization
    res = minimize_scalar(
        negF,
        bounds=(logit.min(), logit.max()),
        method='bounded',
        options={'xatol':1e-12,
                 'disp':False},
    )
    T_opt = res.x
    return -float(T_opt)
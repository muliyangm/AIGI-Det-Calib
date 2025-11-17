# (AAAI 2026) Your Deepfake Detector Can Secretly Achieve SOTA Accuracy, If Calibrated
Muli Yang, Gabriel James Goenawan, Henan Wang, Huaiyuan Qin, Chenghao Xu, Yanhua Yang, Fen Fang, Ying Sun, Joo Hwee Lim, Hongyuan Zhu <br>

# Getting Started
If you have Miniconda or Anaconda installed, you can set up this repository with the following commands:
```
git clone https://github.com/muliyangm/AIGI-Det-Calib
conda create --name aigi_det_calib python=3.10 -y
conda activate aigi_det_calib
pip install -r requirements.txt
```

## Data Preparation
Our method operates at the logits level, so only the logits and labels for each dataset are required. A pre-generated logits file is available under [`./logits`](logits) or testing purposes. The JSON format is as follows:
```
{'dataset_1': {'y_true': [0,0, ...],
               'y_pred': [0.1, 0.3, ...]},
 'dataset_2': {'y_true': [0,0, ...],
               'y_pred': [0.2, -0.2, ...]},}
```

Please refer into [`run.py`](run.py) to see how these logits are used in the evaluation.

## Executing the code
To evaluate using a provided JSON logits file, run:
```
python run.py --logits_json ./logits/CNNSpot_progan_trained.json \
              --n_shots N_SHOTS \
              --n_exp N_EXP 
```
By default, we use `N_SHOTS` = 50 and `N_EXP` = 20. Due to inherent randomness, minor differences between the reported performance and reproduced results are expected.


# Citations
If you find our work helpful, please consider citing our paper:
```
@inproceedings{yang2026deepfake,
  title     = {Your Deepfake Detector Can Secretly Achieve SOTA Accuracy, If Calibrated},
  author    = author = {Yang, Muli and Goenawan, Gabriel James and Wang, Henan and Qin, Huaiyuan and Xu, Chenghao and Yang, Yanhua and Fang, Fen and Sun, Ying and Lim, Joo Hwee and Zhu, Hongyuan},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year      = {2026},
}
```

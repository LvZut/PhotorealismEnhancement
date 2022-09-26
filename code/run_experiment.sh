#!/usr/bin/env bash

python epe/EPEExperiment.py evaluate_model ./config/train_c2r_baseline.yaml --log=info
python epe/EPEExperiment.py evaluate_model ./config/train_c2r_CELoss_ft.yaml --log=info
python epe/EPEExperiment.py evaluate_model ./config/train_c2r_CELoss.yaml --log=info
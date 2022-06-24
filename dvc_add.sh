#!/bin/bash

declare -a experiment_names=("catboost" "blending" "gbt" "logreg" "mlp" "rf" "stacking" "svm")

for experiment_name in ${experiment_names[@]}; 
    do
        dvc add submissions/$experiment_name/baseline.csv
        dvc add oof_pred/$experiment_name/baseline.pkl
        dvc add test_pred/$experiment_name/baseline.pkl
        dvc add checkpoints/$experiment_name/baseline
done

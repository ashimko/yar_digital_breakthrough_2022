#!/usr/bin/bash
#!/usr/local/bin/bash

declare -A model_names=(
  ['catboost']="baseline only_real_cols only_cat_cols cat_encoders baseline_optuna lossguide_optuna depthwise lossguide selected_features"
  ['blending']="baseline"
  ['gbt']="baseline"
  ['logreg']="baseline"
  ['mlp']="baseline"
  ['rf']="baseline"
  ['svm']="baseline"
  ['stacking']="baseline"
  ['lgb']="baseline"
  ['xgb']="baseline"
  ['keras']="baseline"
)

for model_name in ${!model_names[@]}; do
  for experiment_name in ${model_names[$model_name]}; do
    dvc add submissions/$model_name/$experiment_name.csv
    dvc add oof_pred/$model_name/$experiment_name.pkl
    dvc add test_pred/$model_name/$experiment_name.pkl
    dvc add checkpoints/$model_name/$experiment_name
  done
done


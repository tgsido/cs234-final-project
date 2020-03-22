#!/bin/bash

# BASELINES
python main.py --model_name=fixed_dose_baseline --feature_vector_type=baseline2 --warfarin_data_filename='additional/data/warfarin.csv' --output_dir=. --reward_type=standard --alpha_type=standard
python main.py --model_name=warfarin_clinical_dosing_baseline --feature_vector_type=baseline2 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=standard --alpha_type=standard

# Reward: Standard -- Alpha: Standard
python main.py --model_name=lin_ucb_disjoint --feature_vector_type=baseline2 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=standard --alpha_type=standard
python main.py --model_name=lin_ucb_hybrid --feature_vector_type=baseline2 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=standard --alpha_type=standard
python main.py --model_name=lin_ucb_disjoint --feature_vector_type=fv1 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=standard --alpha_type=standard
python main.py --model_name=lin_ucb_hybrid --feature_vector_type=fv1 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=standard --alpha_type=standard

# Reward: Standard -- Alpha: Exponential
python main.py --model_name=lin_ucb_disjoint --feature_vector_type=baseline2 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=standard --alpha_type=exponential
python main.py --model_name=lin_ucb_hybrid --feature_vector_type=baseline2 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=standard --alpha_type=exponential
python main.py --model_name=lin_ucb_disjoint --feature_vector_type=fv1 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=standard --alpha_type=exponential
python main.py --model_name=lin_ucb_hybrid --feature_vector_type=fv1 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=standard --alpha_type=exponential

# Reward: logistic -- Alpha: Standard
# baseline2 feature vector
python main.py --model_name=lin_ucb_disjoint --feature_vector_type=baseline2 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=-3500 --reward_k=10e-4 --alpha_type=standard
python main.py --model_name=lin_ucb_disjoint --feature_vector_type=baseline2 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=-100 --reward_k=10e-4 --alpha_type=standard
python main.py --model_name=lin_ucb_disjoint --feature_vector_type=baseline2 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=500 --reward_k=10e-4 --alpha_type=standard
python main.py --model_name=lin_ucb_disjoint --feature_vector_type=baseline2 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=5000 --reward_k=10e-4 --alpha_type=standard
python main.py --model_name=lin_ucb_disjoint --feature_vector_type=baseline2 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=-3500 --reward_k=5e-4 --alpha_type=standard
python main.py --model_name=lin_ucb_disjoint --feature_vector_type=baseline2 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=-100 --reward_k=5e-4 --alpha_type=standard
python main.py --model_name=lin_ucb_disjoint --feature_vector_type=baseline2 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=500 --reward_k=5e-4 --alpha_type=standard
python main.py --model_name=lin_ucb_disjoint --feature_vector_type=baseline2 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=5000 --reward_k=5e-4 --alpha_type=standard

python main.py --model_name=lin_ucb_hybrid --feature_vector_type=baseline2 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=-3500 --reward_k=10e-4 --alpha_type=standard
python main.py --model_name=lin_ucb_hybrid --feature_vector_type=baseline2 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=-100 --reward_k=10e-4 --alpha_type=standard
python main.py --model_name=lin_ucb_hybrid --feature_vector_type=baseline2 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=500 --reward_k=10e-4 --alpha_type=standard
python main.py --model_name=lin_ucb_hybrid --feature_vector_type=baseline2 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=5000 --reward_k=10e-4 --alpha_type=standard
python main.py --model_name=lin_ucb_hybrid --feature_vector_type=baseline2 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=-3500 --reward_k=5e-4 --alpha_type=standard
python main.py --model_name=lin_ucb_hybrid --feature_vector_type=baseline2 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=-100 --reward_k=5e-4 --alpha_type=standard
python main.py --model_name=lin_ucb_hybrid --feature_vector_type=baseline2 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=500 --reward_k=5e-4 --alpha_type=standard
python main.py --model_name=lin_ucb_hybrid --feature_vector_type=baseline2 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=5000 --reward_k=5e-4 --alpha_type=standard

# fv1 feature vector
python main.py --model_name=lin_ucb_disjoint --feature_vector_type=fv1 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=-3500 --reward_k=10e-4 --alpha_type=standard
python main.py --model_name=lin_ucb_disjoint --feature_vector_type=fv1 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=-100 --reward_k=10e-4 --alpha_type=standard
python main.py --model_name=lin_ucb_disjoint --feature_vector_type=fv1 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=500 --reward_k=10e-4 --alpha_type=standard
python main.py --model_name=lin_ucb_disjoint --feature_vector_type=fv1 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=5000 --reward_k=10e-4 --alpha_type=standard
python main.py --model_name=lin_ucb_disjoint --feature_vector_type=fv1 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=-3500 --reward_k=5e-4 --alpha_type=standard
python main.py --model_name=lin_ucb_disjoint --feature_vector_type=fv1 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=-100 --reward_k=5e-4 --alpha_type=standard
python main.py --model_name=lin_ucb_disjoint --feature_vector_type=fv1 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=500 --reward_k=5e-4 --alpha_type=standard
python main.py --model_name=lin_ucb_disjoint --feature_vector_type=fv1 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=5000 --reward_k=5e-4 --alpha_type=standard

python main.py --model_name=lin_ucb_hybrid --feature_vector_type=fv1 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=-3500 --reward_k=10e-4 --alpha_type=standard
python main.py --model_name=lin_ucb_hybrid --feature_vector_type=fv1 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=-100 --reward_k=10e-4 --alpha_type=standard
python main.py --model_name=lin_ucb_hybrid --feature_vector_type=fv1 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=500 --reward_k=10e-4 --alpha_type=standard
python main.py --model_name=lin_ucb_hybrid --feature_vector_type=fv1 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=5000 --reward_k=10e-4 --alpha_type=standard
python main.py --model_name=lin_ucb_hybrid --feature_vector_type=fv1 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=-3500 --reward_k=5e-4 --alpha_type=standard
python main.py --model_name=lin_ucb_hybrid --feature_vector_type=fv1 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=-100 --reward_k=5e-4 --alpha_type=standard
python main.py --model_name=lin_ucb_hybrid --feature_vector_type=fv1 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=500 --reward_k=5e-4 --alpha_type=standard
python main.py --model_name=lin_ucb_hybrid --feature_vector_type=fv1 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=5000 --reward_k=5e-4 --alpha_type=standard

# Reward: logistic -- Alpha: Exponential
# baseline2 feature vector
python main.py --model_name=lin_ucb_disjoint --feature_vector_type=baseline2 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=-3500 --reward_k=10e-4 --alpha_type=exponential
python main.py --model_name=lin_ucb_disjoint --feature_vector_type=baseline2 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=-100 --reward_k=10e-4 --alpha_type=exponential
python main.py --model_name=lin_ucb_disjoint --feature_vector_type=baseline2 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=500 --reward_k=10e-4 --alpha_type=exponential
python main.py --model_name=lin_ucb_disjoint --feature_vector_type=baseline2 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=5000 --reward_k=10e-4 --alpha_type=exponential
python main.py --model_name=lin_ucb_disjoint --feature_vector_type=baseline2 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=-3500 --reward_k=5e-4 --alpha_type=exponential
python main.py --model_name=lin_ucb_disjoint --feature_vector_type=baseline2 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=-100 --reward_k=5e-4 --alpha_type=exponential
python main.py --model_name=lin_ucb_disjoint --feature_vector_type=baseline2 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=500 --reward_k=5e-4 --alpha_type=exponential
python main.py --model_name=lin_ucb_disjoint --feature_vector_type=baseline2 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=5000 --reward_k=5e-4 --alpha_type=exponential

python main.py --model_name=lin_ucb_hybrid --feature_vector_type=baseline2 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=-3500 --reward_k=10e-4 --alpha_type=exponential
python main.py --model_name=lin_ucb_hybrid --feature_vector_type=baseline2 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=-100 --reward_k=10e-4 --alpha_type=exponential
python main.py --model_name=lin_ucb_hybrid --feature_vector_type=baseline2 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=500 --reward_k=10e-4 --alpha_type=exponential
python main.py --model_name=lin_ucb_hybrid --feature_vector_type=baseline2 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=5000 --reward_k=10e-4 --alpha_type=exponential
python main.py --model_name=lin_ucb_hybrid --feature_vector_type=baseline2 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=-3500 --reward_k=5e-4 --alpha_type=exponential
python main.py --model_name=lin_ucb_hybrid --feature_vector_type=baseline2 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=-100 --reward_k=5e-4 --alpha_type=exponential
python main.py --model_name=lin_ucb_hybrid --feature_vector_type=baseline2 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=500 --reward_k=5e-4 --alpha_type=exponential
python main.py --model_name=lin_ucb_hybrid --feature_vector_type=baseline2 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=5000 --reward_k=5e-4 --alpha_type=exponential

# fv1 feature vector
python main.py --model_name=lin_ucb_disjoint --feature_vector_type=fv1 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=-3500 --reward_k=10e-4 --alpha_type=exponential
python main.py --model_name=lin_ucb_disjoint --feature_vector_type=fv1 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=-100 --reward_k=10e-4 --alpha_type=exponential
python main.py --model_name=lin_ucb_disjoint --feature_vector_type=fv1 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=500 --reward_k=10e-4 --alpha_type=exponential
python main.py --model_name=lin_ucb_disjoint --feature_vector_type=fv1 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=5000 --reward_k=10e-4 --alpha_type=exponential
python main.py --model_name=lin_ucb_disjoint --feature_vector_type=fv1 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=-3500 --reward_k=5e-4 --alpha_type=exponential
python main.py --model_name=lin_ucb_disjoint --feature_vector_type=fv1 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=-100 --reward_k=5e-4 --alpha_type=exponential
python main.py --model_name=lin_ucb_disjoint --feature_vector_type=fv1 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=500 --reward_k=5e-4 --alpha_type=exponential
python main.py --model_name=lin_ucb_disjoint --feature_vector_type=fv1 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=5000 --reward_k=5e-4 --alpha_type=exponential

python main.py --model_name=lin_ucb_hybrid --feature_vector_type=fv1 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=-3500 --reward_k=10e-4 --alpha_type=exponential
python main.py --model_name=lin_ucb_hybrid --feature_vector_type=fv1 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=-100 --reward_k=10e-4 --alpha_type=exponential
python main.py --model_name=lin_ucb_hybrid --feature_vector_type=fv1 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=500 --reward_k=10e-4 --alpha_type=exponential
python main.py --model_name=lin_ucb_hybrid --feature_vector_type=fv1 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=5000 --reward_k=10e-4 --alpha_type=exponential
python main.py --model_name=lin_ucb_hybrid --feature_vector_type=fv1 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=-3500 --reward_k=5e-4 --alpha_type=exponential
python main.py --model_name=lin_ucb_hybrid --feature_vector_type=fv1 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=-100 --reward_k=5e-4 --alpha_type=exponential
python main.py --model_name=lin_ucb_hybrid --feature_vector_type=fv1 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=500 --reward_k=5e-4 --alpha_type=exponential
python main.py --model_name=lin_ucb_hybrid --feature_vector_type=fv1 --warfarin_data_filename=additional/data/warfarin.csv --output_dir=. --reward_type=logistic --reward_l=1 --reward_x0=5000 --reward_k=5e-4 --alpha_type=exponential

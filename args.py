### args.py ###
import argparse

def get_main_args():
    """Get arguments needed in main.py."""
    parser = argparse.ArgumentParser('Entry point for WarfarinRL')

    #parser.add_argument("--do_linear_ucb_disjoint", action='store_true',help="Whether to run training.")
    parser.add_argument("--model_name", default='lin_ucb_disjoint', type=str, required=True,
                        help="Model Name (lin_ucb_disjoint, lin_ucb_hybrid,fixed_dose_baseline, warfarin_clinical_dosing_baseline)")
    parser.add_argument("--feature_vector_type", default='baseline2', type=str, required=False,
                        help="Type of feature_vector_type, Includes (baseline2,fv1)")
    parser.add_argument("--warfarin_data_filename", default=None, type=str, required=True,
                        help="Path to warfarin data file")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--num_actions", default=3, type=int, required=False,
                        help="num_actions = K = 3 in this problem")
    parser.add_argument("--decay_rate", default=1, type=float, required=False,
                        help="Decay rate, which affects alpha")
    parser.add_argument("--reward_type", default='standard', type=str, required=False,
                        help="Reward type, Includes (standard, logistic)")
    parser.add_argument("--alpha_type", default='standard', type=str, required=False,
                        help="Reward type, Includes (standard, exponential)")
    parser.add_argument("--reward_l", default=-1.0, type=float, required=False,
                        help="l")
    parser.add_argument("--reward_k", default=-1.0, type=float, required=False,
                        help="k")
    parser.add_argument("--reward_x0", default=-1.0, type=float, required=False,
                        help="x0")




    args = parser.parse_args()

    return args

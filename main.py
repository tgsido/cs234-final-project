"""
main.py

Main entry point for models
"""

from __future__ import absolute_import, division, print_function, unicode_literals
from models import LinUCBDisjoint, LinUCBHybrid, WarfarinClinicalDosingAlgorithmBaseline, FixedDosageBaseline
from utils import get_context_and_labels, get_label_from_action_index, get_optimal_betas
from tensorboardX import SummaryWriter
from args import get_main_args
import matplotlib.pyplot as plt

import os
import io
import json
import sys
import logging
import tensorflow as tf
import pathlib
import random
import numpy as np

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(MAIN_DIR, "data")

NUM_TRIALS = 20

def get_general_descriptor_str(args):
    if args.model_name == 'fixed_dose_baseline' or args.model_name == 'warfarin_clinical_dosing_baseline':
        descriptor_str = "model_{}".format(args.model_name)
    else:
        if args.reward_type == 'standard':
            descriptor_str = "model_{}_feature_{}_reward_{}_alpha_type_{}".format(args.model_name, args.feature_vector_type, args.reward_type, args.alpha_type)
        else:
            descriptor_str = "model_{}_feature_{}_reward_{}_reward_l_{}_reward_k_{}_reward_x0_{}_alpha_type_{}".format(args.model_name, args.feature_vector_type, args.reward_type, \
                args.reward_l, args.reward_k, args.reward_x0, args.alpha_type)
    return descriptor_str

def get_descriptor_str(iter, args):
    if args.model_name == 'fixed_dose_baseline' or args.model_name == 'warfarin_clinical_dosing_baseline':
        descriptor_str = "model_{}_iter_{}".format(args.model_name, iter)
    else:
        if args.reward_type == 'standard':
            descriptor_str = "model_{}_iter_{}_feature_{}_reward_{}_alpha_type_{}".format(args.model_name, iter, args.feature_vector_type, args.reward_type, args.alpha_type)
        else:
            descriptor_str = "model_{}_iter_{}_feature_{}_reward_{}_reward_l_{}_reward_k_{}_reward_x0_{}_alpha_type_{}".format(args.model_name, iter, args.feature_vector_type, args.reward_type, \
                args.reward_l, args.reward_k, args.reward_x0, args.alpha_type)
    return descriptor_str

def get_confidence_interval_title_str(args, quantity):
    title_str = None
    if args.model_name == 'fixed_dose_baseline' or args.model_name == 'warfarin_clinical_dosing_baseline':
        title_str = 'Confidence Interval for {} (95%) - Model:{} - feature:{} - reward_type:{}'.format(quantity, args.model_name,args.feature_vector_type, args.reward_type)
    else:
        if args.reward_type == 'standard':
            title_str = 'Confidence Interval for {} (95%) - Model:{} - feature:{} - reward_type:{} - alpha_type:{}'.format(quantity, args.model_name,args.feature_vector_type, args.reward_type, args.alpha_type)
        else:
            title_str = 'Confidence Interval for {} (95%) - Model:{} - feature:{} - reward_type:{} - reward_l:{} - reward_k:{} - reward_x0:{} - alpha_type:{}'.format(quantity, args.model_name,args.feature_vector_type, args.reward_type, \
                args.reward_l, args.reward_k,  args.reward_x0, args.alpha_type)
    return title_str

def main(args):

    print("Running test...")
    print("Args:{}".format(args))

    ### GET DATA ###
    context_vectors, labels = get_context_and_labels(args.warfarin_data_filename, args.feature_vector_type)
    num_examples, vector_dimension = context_vectors.shape

    iter_to_prediction_dict = {}
    iter_to_incorrect_percentage_dict = {}
    iter_to_regret_lst_dict = {}


    reward_dict = {'reward_l':args.reward_l, 'reward_k':args.reward_k,'reward_x0':args.reward_x0}



    regret_lst = []

    ### RUN 20 TRIALS ###
    for iter in range(1, NUM_TRIALS + 1):
        ### SETUP TENSORBOARD ###
        descriptor_str = get_descriptor_str(iter, args)
        """
        if args.model_name == 'fixed_dose_baseline' or args.model_name == 'warfarin_clinical_dosing_baseline':
            iter_dir = os.path.join(args.output_dir,'tb','model_{}_iter_{}_feature_type_{}_reward_type_{}'.format(args.model_name,iter, args.feature_vector_type,args.reward_type))
        else:
            iter_dir = os.path.join(args.output_dir,'tb','model_name_{}_iter_{}_feature_type_{}_reward_type_{}_max_pos_reward_{}_max_neg_reward_{}_reward_a_{}_reward_b_{}'.format(args.model_name,iter, args.feature_vector_type,args.reward_type, \
                args.positive_reward_max, args.negative_reward_max, args.reward_a, args.reward_b))
        """
        pic_dir = os.path.join(args.output_dir,'pics')
        if not os.path.exists(pic_dir):
            os.makedirs(pic_dir)

        iter_dir = os.path.join(args.output_dir,'tb',descriptor_str)
        if not os.path.exists(iter_dir):
            os.makedirs(iter_dir)

        writer = SummaryWriter(iter_dir)

        ## shuffle vectors and labels ##
        perm = np.arange(num_examples,dtype=np.int32)
        np.random.shuffle(perm)
        #print(perm[:10])
        context_vectors = context_vectors[perm]
        labels = labels[perm]

        ## set alpha to decrease as time goes on ##
        initial_alpha = alpha = 1

        regret_time_lst = []

        ## instantiate model ##
        if args.model_name == 'lin_ucb_disjoint':
            model = LinUCBDisjoint(iter, args.num_actions, vector_dimension, alpha, args.alpha_type, args.reward_type, reward_dict)
            confidence_time_step = 1
        elif args.model_name == 'lin_ucb_hybrid':
            model = LinUCBHybrid(iter, args.num_actions, vector_dimension, alpha, args.alpha_type, args.reward_type, reward_dict)
            confidence_time_step = 2
        elif args.model_name == 'fixed_dose_baseline':
            model = FixedDosageBaseline(iter, args.num_actions, vector_dimension, alpha, args.alpha_type, args.reward_type, reward_dict)
            confidence_time_step = 3
        elif args.model_name == 'warfarin_clinical_dosing_baseline':
            model = WarfarinClinicalDosingAlgorithmBaseline(iter, args.num_actions, vector_dimension, alpha, args.alpha_type, args.reward_type, reward_dict)
            confidence_time_step = 4
        else:
            print('invalid model_name used!!')

        correct_count, total_count = 0,0

        true_idx_predicted_action_lst = [] # [(idx=2, selected_action_idx=0), ...]
        incorrect_percentage_lst = [] # [.23,.25, ....]

        ## iterate through each vector, running algorithm ##
        for t in range(num_examples):
            context_vector = context_vectors[t]
            true_label = labels[t]
            selected_action_idx = model.handle_single_example(context_vector, true_label, t)

            if get_label_from_action_index(selected_action_idx) == true_label:
                correct_count += 1
            total_count += 1

            true_idx_predicted_action_lst.append((int(perm[t]), selected_action_idx))
            incorrect_percentage_lst.append(1 - (correct_count/total_count))

            ## adjust alpha ##
            if args.alpha_type == 'standard':
                alpha = initial_alpha
            elif args.alpha_type == 'exponential':
                alpha = initial_alpha * (1. / (1. + args.decay_rate * (t/50))) # time-based decay for alpha
            model.alpha = alpha

            writer.add_scalar('incorrect_fraction', 1 - (correct_count/total_count) , t)
            writer.add_scalar('correct_fraction', (correct_count/total_count) , t)
            writer.add_scalar('alpha', alpha , t)


        #print('incorrect_percentage_lst[-3:]', incorrect_percentage_lst[-3:])
        iter_to_prediction_dict[iter] = true_idx_predicted_action_lst
        iter_to_incorrect_percentage_dict[iter] = incorrect_percentage_lst
        print("iter{}/{}".format(iter,NUM_TRIALS))

        ### Compute Expected Regret ###
        regret_sum = 0
        beta_list = get_optimal_betas(context_vectors, labels, args.num_actions)
        true_idx_predicted_action_lst = iter_to_prediction_dict[iter]
        for t in range(num_examples):
            context_vector = context_vectors[t]
            true_label = labels[t]
            predicted_action_idx = true_idx_predicted_action_lst[t][1]
            beta_idx_val_tuple_lst = []
            for i in range(args.num_actions):
                val = np.dot(beta_list[i], context_vector)
                beta_idx_val_tuple_lst.append((i, val))
            max_val = max(beta_idx_val_tuple_lst, key=lambda tup: tup[1])[1]
            pred_beta_val = np.dot(beta_list[predicted_action_idx], context_vector)
            regret_sum += max_val - pred_beta_val
            regret_time_lst.append(max_val - pred_beta_val)
        regret_lst.append(regret_sum)

        iter_to_regret_lst_dict[iter] = regret_time_lst

        writer.close()

    ### Calculate Confidence Intervals ###
    T_DISTR_VAL = 2.093 # degrees of freedom = 20-1; .05

    ## plot incorrect confidence_intervals by time step ##
    plt.figure(figsize=(12,5))
    incorrect_fraction_means = []
    incorrect_interval_margins = []
    for t in range(num_examples):
        incorrect_fraction_t_lst = []
        for iter in range(1, NUM_TRIALS + 1):
            incorrect_fraction = iter_to_incorrect_percentage_dict[iter][t]
            incorrect_fraction_t_lst.append(incorrect_fraction)

        incorrect_fraction_std_dev = np.std(incorrect_fraction_t_lst)
        incorrect_fraction_margin = T_DISTR_VAL * (incorrect_fraction_std_dev/np.sqrt(NUM_TRIALS))

        incorrect_fraction_means.append(np.mean(incorrect_fraction_t_lst))
        incorrect_interval_margins.append(incorrect_fraction_margin)

    plt.errorbar(np.arange(num_examples), incorrect_fraction_means, yerr=incorrect_interval_margins, barsabove=True, errorevery=20)
    plt.xlabel('patient index')
    plt.ylabel('incorrect_fraction')

    confidence_interval_title = get_confidence_interval_title_str(args, 'Incorrect Dosage')
    #plt.title(confidence_interval_title)
    plt.title('Incorrect Dosage')
    plt.tight_layout()
    plt.savefig(os.path.join(pic_dir, "incorrect_fraction__"+get_general_descriptor_str(args)+".png"))

    ## plot regret intervlas by time step ##
    plt.figure(figsize=(12,5))
    regret_means = []
    regret_interval_margins = []
    for t in range(num_examples):
        regret_t_lst = []
        for iter in range(1, NUM_TRIALS + 1):
            regret_val = iter_to_regret_lst_dict[iter][t]
            regret_t_lst.append(regret_val)

        regret_std_dev = np.std(regret_t_lst)
        regret_margin = T_DISTR_VAL * (regret_std_dev/np.sqrt(NUM_TRIALS))

        regret_means.append(np.mean(regret_t_lst))
        regret_interval_margins.append(regret_margin)

    plt.errorbar(np.arange(num_examples), regret_means, yerr=regret_interval_margins, barsabove=True, errorevery=100)
    plt.xlabel('patient index')
    plt.ylabel('regret')
    confidence_interval_title = get_confidence_interval_title_str(args, 'Regret')
    #plt.title(confidence_interval_title)
    plt.title('Regret')
    plt.tight_layout()
    plt.savefig(os.path.join(pic_dir, "regret__"+get_general_descriptor_str(args)+".png"))

    final_incorrect_percentages = np.array([iter_to_incorrect_percentage_dict[iter][-1] for iter in range(1, NUM_TRIALS + 1)])
    #print("final_incorrect_percentages", final_incorrect_percentages)
    incorrect_sample_std_dev = np.std(final_incorrect_percentages)
    incorrect_sample_mean = np.mean(final_incorrect_percentages)
    incorrect_interval_margin = T_DISTR_VAL * (incorrect_sample_std_dev/np.sqrt(NUM_TRIALS))
    incorrect_lower_bound = incorrect_sample_mean - incorrect_interval_margin
    incorrect_upper_bound = incorrect_sample_mean + incorrect_interval_margin

    regret_lst = np.array(regret_lst)
    regret_sample_std_dev = np.std(regret_lst)
    regret_sample_mean = np.mean(regret_lst)
    regret_interval_margin =  T_DISTR_VAL * (regret_sample_std_dev/np.sqrt(NUM_TRIALS))
    regret_lower_bound = regret_sample_mean - regret_interval_margin
    regret_upper_bound = regret_sample_mean + regret_interval_margin

    f = open(os.path.join(pic_dir,"{}_stats.txt".format(get_general_descriptor_str(args))),"w+")

    f.write(get_general_descriptor_str(args)+'\n\n')
    f.write("incorrect_interval_margin:{} -- regret_interval_margin:{}\n".format(incorrect_interval_margin, regret_interval_margin))
    f.write('incorrect_sample_mean:{} -- incorrect_lower_bound:{} -- incorrect_upper_bound:{}\n'.format(incorrect_sample_mean, incorrect_lower_bound, incorrect_upper_bound))
    f.write('correct_sample_mean:{} -- correct_lower_bound:{} -- correct_upper_bound:{}\n'.format(1 - incorrect_sample_mean, 1 - incorrect_lower_bound, 1 - incorrect_upper_bound))
    f.write("regret_sample_mean:{} -- regret_lower_bound:{} -- regret_upper_bound:{}\n".format(regret_sample_mean, regret_lower_bound, regret_upper_bound))

    tb_dir = os.path.join(args.output_dir,'confidence_intervals', get_general_descriptor_str(args))
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)
    tb_writer = SummaryWriter(tb_dir)
    tb_writer.add_scalar('regret_interval', regret_lower_bound, confidence_time_step)
    tb_writer.add_scalar('regret_interval', regret_upper_bound, confidence_time_step)
    tb_writer.add_scalar('incorrect_fraction_interval', incorrect_lower_bound, confidence_time_step)
    tb_writer.add_scalar('incorrect_fraction_interval', incorrect_upper_bound, confidence_time_step)
    tb_writer.add_scalar('correct_fraction_interval', 1 - incorrect_lower_bound, confidence_time_step)
    tb_writer.add_scalar('correct_fraction_interval', 1 - incorrect_upper_bound, confidence_time_step)
    tb_writer.close()


if __name__ == '__main__':
    main(get_main_args())

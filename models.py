import numpy as np
import random

from utils import get_label_from_action_index, get_dosage_label_from_mg, get_label_index_from_label
EPSILON = 5e-2

def get_reward(reward_type, selected_action_idx, true_label, reward_dict=None, t=None):
    if reward_type == 'standard':
        reward = 0 if get_label_from_action_index(selected_action_idx) == true_label else -1
    elif reward_type == 'logistic':
        if get_label_from_action_index(selected_action_idx) != true_label: # wrong
            reward = -reward_dict['reward_l'] / (1 + np.exp(-reward_dict['reward_k']*(t + reward_dict['reward_x0'])))
        else: # right
            reward = reward_dict['reward_l'] / (1 + np.exp(-reward_dict['reward_k']*(t + reward_dict['reward_x0'])))
    return reward

class FixedDosageBaseline:
    """
    Init:

    baseline1_obj = Baseline1(0, 3, 8)

    """
    def __init__(self, id, num_actions, feature_size, alpha, alpha_type, reward_type, reward_dict):
        pass

    """
    Given context vector:

    Finds action with highest UCB, updates params for corresponding action
    Returns index of selected action
    """
    def handle_single_example(self, context_vector, true_label, t):
        selected_action_idx = 1
        return selected_action_idx

class WarfarinClinicalDosingAlgorithmBaseline:
    """
    Init:

    baseline1_obj = Baseline2(0, 3, 8)

    """
    def __init__(self, id, num_actions, feature_size, alpha, alpha_type, reward_type, reward_dict):
        pass

    """
    Given context vector:

    Finds action with highest UCB, updates params for corresponding action
    Returns index of selected action
    """
    def handle_single_example(self, context_vector, true_label, t):
        selected_action_idx = None

        age_in_dec = context_vector[0]
        height_in_cm = context_vector[1]
        weight_in_kg = context_vector[2]
        asian_race = context_vector[3]
        black_or_african_american = context_vector[4]
        missing_or_mixed_race = context_vector[5]
        enzyme_inducer_status = context_vector[6]
        amiodarone_status = context_vector[7]

        calc_mg = 4.0376 - .2546*age_in_dec + .0118*height_in_cm
        calc_mg += .0134*weight_in_kg - .6752*asian_race + .4060*black_or_african_american
        calc_mg += .0443*missing_or_mixed_race + 1.2799*enzyme_inducer_status - .5695*amiodarone_status

        calc_mg = calc_mg**2

        label = get_dosage_label_from_mg(calc_mg)
        selected_action_idx = get_label_index_from_label(label)

        return selected_action_idx

class LinUCBHybrid:
    """
    Init:

    lin_ucb_obj = LinUCBHybrid(0, 3, 8)

    """
    def __init__(self, id, num_actions, feature_size, alpha, alpha_type, reward_type, reward_dict):
        self.d = 8
        self.k = feature_size - 8
        self.id = id
        self.K = num_actions
        self.alpha = alpha

        self.A_0 = np.identity(self.k)
        self.b_0 = np.zeros(self.k,)

        self.A_matrix_lst = []
        self.b_matrix_lst = []
        self.B_matrix_lst = []

        self.reward_type = reward_type
        self.reward_dict = reward_dict

        for _ in range(num_actions):
            self.A_matrix_lst.append(np.identity(self.d))
            self.B_matrix_lst.append(np.zeros((self.d,self.k)))
            self.b_matrix_lst.append(np.zeros(self.d,))

    """
    Given context vector:

    Finds action with highest UCB, updates params for corresponding action
    Returns index of selected action
    """
    def handle_single_example(self, context_vector, true_label, t):
        #print('context_vector', context_vector)
        x_ta = context_vector[:self.d]
        z_ta = context_vector[self.d:]

        beta_hat = np.matmul(np.linalg.inv(self.A_0), self.b_0) # (k,)


        theta_p_lst = []
        for i in range(self.K):

            A_inv_i = np.linalg.inv(self.A_matrix_lst[i])
            A_0_inv = np.linalg.inv(self.A_0)
            theta_i = np.matmul(A_inv_i, (self.b_matrix_lst[i] - np.matmul(self.B_matrix_lst[i], beta_hat)))
            s_ta = np.dot(z_ta, np.matmul(A_0_inv,z_ta)) - 2*np.dot(z_ta, np.matmul(A_0_inv, np.matmul(np.transpose(self.B_matrix_lst[i]), np.matmul(A_inv_i, x_ta))))
            s_ta += np.dot(x_ta, np.matmul(A_inv_i, x_ta)) + np.dot(x_ta, np.matmul(A_inv_i, np.matmul(self.B_matrix_lst[i], np.matmul(A_0_inv, np.matmul(np.transpose(self.B_matrix_lst[i]), np.matmul(A_inv_i, x_ta))))))
            p_ti = np.dot(z_ta, beta_hat) + np.dot(x_ta,theta_i) + self.alpha*np.sqrt(s_ta) + np.random.normal(0, EPSILON)
            theta_p_lst.append((theta_i, p_ti))

        max_p_ti = max(theta_p_lst, key = lambda tup: tup[1])[1]
        max_action_indices = []
        for i in range(self.K):
            if theta_p_lst[i][1] == max_p_ti:
                max_action_indices.append(i)

        #print("theta_p_lst",theta_p_lst)
        #print("max_action_indices",max_action_indices)

        ## shuffle top actions then pick 1st one ##
        random.shuffle(max_action_indices)
        selected_action_idx = max_action_indices[0]

        ## set reward to 0 if correct action, -1 otherwise ##
        #reward = None
        reward = get_reward(self.reward_type, selected_action_idx, true_label, reward_dict=self.reward_dict, t=t)
        """
        if self.reward_type == 'standard':
            reward = 0 if get_label_from_action_index(selected_action_idx) == true_label else -1
        elif self.reward_type == 'logistic':
            if get_label_from_action_index(selected_action_idx) != true_label: # wrong
                reward = self.reward_dict['max_negative_reward'] / (1 + self.reward_dict['reward_a'] * np.exp(-self.reward_dict['reward_b']*t))
            else: # right
                reward = self.reward_dict['max_positive_reward'] / (1 + self.reward_dict['reward_a'] * np.exp(-self.reward_dict['reward_b']*t))
        """

        ### Updates to selected action's parameters ###
        self.A_0 = self.A_0 + np.matmul(np.transpose(self.B_matrix_lst[selected_action_idx]), np.matmul(np.linalg.inv(self.A_matrix_lst[selected_action_idx]), self.B_matrix_lst[selected_action_idx]))
        self.b_0 = self.b_0 + np.matmul(np.transpose(self.B_matrix_lst[selected_action_idx]), np.matmul(np.linalg.inv(self.A_matrix_lst[selected_action_idx]), self.b_matrix_lst[selected_action_idx]))

        A_selected_idx, B_selected_idx, b_selected_idx = self.A_matrix_lst[selected_action_idx], self.B_matrix_lst[selected_action_idx], self.b_matrix_lst[selected_action_idx]
        self.A_matrix_lst[selected_action_idx] = A_selected_idx + np.matmul(np.expand_dims(x_ta, axis=1), np.expand_dims(x_ta, axis=0))
        self.B_matrix_lst[selected_action_idx] = B_selected_idx + np.matmul(np.expand_dims(x_ta, axis=1), np.expand_dims(z_ta, axis=0))
        self.b_matrix_lst[selected_action_idx] = b_selected_idx + reward * x_ta

        self.A_0 = self.A_0 + np.matmul(np.expand_dims(z_ta, axis=1), np.expand_dims(z_ta, axis=0)) - np.matmul(np.transpose(self.B_matrix_lst[selected_action_idx]), np.matmul(np.linalg.inv(self.A_matrix_lst[selected_action_idx]), self.B_matrix_lst[selected_action_idx]))
        self.b_0 = self.b_0 + reward*z_ta - np.matmul(np.transpose(self.B_matrix_lst[selected_action_idx]), np.matmul(np.linalg.inv(self.A_matrix_lst[selected_action_idx]), self.b_matrix_lst[selected_action_idx]))


        return selected_action_idx


class LinUCBDisjoint:
    """
    Init:

    lin_ucb_obj = LinUCBDisjoint(0, 3, 8)

    """
    def __init__(self, id, num_actions, feature_size, alpha, alpha_type, reward_type, reward_dict):
        self.A_matrix_lst = []
        self.b_matrix_lst = []
        self.id = id
        self.K = num_actions
        self.alpha = alpha
        self.reward_type = reward_type
        self.reward_dict = reward_dict
        for _ in range(num_actions):
            self.A_matrix_lst.append(np.identity(feature_size))
            self.b_matrix_lst.append(np.zeros(feature_size,))

    """
    Given context vector:

    Finds action with highest UCB, updates params for corresponding action
    Returns index of selected action
    """
    def handle_single_example(self, context_vector, true_label, t):
        #print('context_vector', context_vector)
        theta_p_lst = []
        for i in range(self.K):
            A_inv = np.linalg.inv(self.A_matrix_lst[i])
            theta_i = np.matmul(A_inv, self.b_matrix_lst[i])
            p_ti = np.dot(theta_i, context_vector) + self.alpha*np.sqrt(np.dot(context_vector, np.matmul(A_inv, context_vector))) + np.random.normal(0, EPSILON)
            theta_p_lst.append((theta_i, p_ti))

        max_p_ti = max(theta_p_lst, key = lambda tup: tup[1])[1]
        max_action_indices = []
        for i in range(self.K):
            if theta_p_lst[i][1] == max_p_ti:
                max_action_indices.append(i)

        #print("theta_p_lst",theta_p_lst)
        #print("max_action_indices",max_action_indices)

        ## shuffle top actions then pick 1st one ##
        random.shuffle(max_action_indices)
        selected_action_idx = max_action_indices[0]

        ## set reward to 0 if correct action, -1 otherwise ##
        reward = get_reward(self.reward_type, selected_action_idx, true_label, reward_dict=self.reward_dict, t=t)
        """
        reward = None
        if self.reward_type == 'standard':
            reward = 0 if get_label_from_action_index(selected_action_idx) == true_label else -1
        elif self.reward_type == 'logistic':
            if get_label_from_action_index(selected_action_idx) != true_label: # wrong
                reward = self.reward_dict['max_negative_reward'] / (1 + self.reward_dict['reward_a'] * np.exp(-self.reward_dict['reward_b']*t))
            else: # right
                reward = self.reward_dict['max_positive_reward'] / (1 + self.reward_dict['reward_a'] * np.exp(-self.reward_dict['reward_b']*t))
        """

        ### Updates to selected action's parameters ###
        A_selected_idx, b_selected_idx = self.A_matrix_lst[selected_action_idx], self.b_matrix_lst[selected_action_idx]
        A_selected_idx = A_selected_idx + np.matmul(np.expand_dims(context_vector, axis=1), np.expand_dims(context_vector, axis=0))
        b_selected_idx = b_selected_idx + reward * context_vector

        self.A_matrix_lst[selected_action_idx] = A_selected_idx
        self.b_matrix_lst[selected_action_idx] = b_selected_idx

        return selected_action_idx

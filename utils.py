"""
utils.py

Several useful functions
"""

import pandas as pd
import numpy as np

print('pandas version',pd.__version__)

TRUE_DOSAGE_COLUMN_NAME = 'Therapeutic Dose of Warfarin'
TRUE_DOSAGE_LABEL_COLUMN_NAME = 'true_dosage_label'

def get_label_index_from_label(label):
    if label == 'low':
        return 0
    elif label == 'med':
        return 1
    elif label == 'high':
        return 2

def get_label_from_action_index(idx):
    if idx == 0:
        return 'low'
    elif idx == 1:
        return 'med'
    elif idx == 2:
        return 'high'
    else:
        assert(idx in list(range(2))), 'action_index:{} is not valid'.format(idx)

def get_dosage_label_from_mg(mg):
    label = None
    if mg < 21:
        label = 'low'
    elif 21 <= mg <= 49:
        label = 'med'
    elif mg > 49:
        label = 'high'
    return label

def get_dosage_label(row):
    dosage_mg = row['Therapeutic Dose of Warfarin']
    label = None
    if dosage_mg < 21:
        label = 'low'
    elif 21 <= dosage_mg <= 49:
        label = 'med'
    elif dosage_mg > 49:
        label = 'high'
    return label


def load_data(warfarin_filename):
    warfarin_df = pd.read_csv(warfarin_filename)
    #df = warfarin_df
    #print("df.columns[df.isnull().any()].tolist() ", df.columns[df.isnull().any()].tolist() )
    warfarin_df = warfarin_df.dropna(subset=[TRUE_DOSAGE_COLUMN_NAME])
    warfarin_df = warfarin_df.groupby(warfarin_df.columns, axis = 1).transform(lambda x: x.fillna(x.mean()))
    warfarin_df[TRUE_DOSAGE_LABEL_COLUMN_NAME] = warfarin_df.apply (lambda row: get_dosage_label(row), axis=1)
    return warfarin_df

def get_dosage_baseline2(row):
    first_digit = 0
    if isinstance(row['Age'],str):
        first_digit = row['Age'][0]
    age_in_dec = int(first_digit)
    height_in_cm = row['Height (cm)']
    weight_in_kg = row['Weight (kg)']
    asian_race = 1 if row['Race'] == 'Asian' else 0
    black_or_african_american = 1 if row['Race'] == 'Black or African American' else 0
    missing_or_mixed_race = 1 if row['Race'] == 'Unknown' else 0
    enzyme_vals = [row['Carbamazepine (Tegretol)'], row['Phenytoin (Dilantin)'], row['Rifampin or Rifampicin']]
    enzyme_inducer_status = 1 if 1 in enzyme_vals else 0
    amiodarone_status = 1 if row['Amiodarone (Cordarone)'] else 0

    calc_mg = 4.0376 - .2546*age_in_dec + .0118*height_in_cm
    calc_mg += .0134*weight_in_kg - .6752*asian_race + .4060*black_or_african_american
    calc_mg += .0443*missing_or_mixed_race + 1.2799*enzyme_inducer_status - .5695*amiodarone_status

    calc_mg = calc_mg**2
    return calc_mg

def get_context_vectors(df, feature_vector_type):
    print("getting context_vectors...")
    if feature_vector_type == 'baseline2':
        needed_cols = [TRUE_DOSAGE_LABEL_COLUMN_NAME, TRUE_DOSAGE_LABEL_COLUMN_NAME, 'Age', 'Height (cm)', 'Weight (kg)', 'Race', 'Carbamazepine (Tegretol)', 'Phenytoin (Dilantin)', 'Rifampin or Rifampicin', 'Amiodarone (Cordarone)']
        cols_to_drop = cols = [col for col in df.columns if col not in needed_cols]
        df = df.drop(cols_to_drop, axis=1)
        #df = df.groupby(df.columns, axis = 1).transform(lambda x:  x.fillna(x.mode()) if isinstance(x,str) else)
        df['fv1'] = df.apply(lambda row: int(row['Age'][0]) if isinstance(row['Age'],str) else 0, axis=1)
        df['fv2'] = df.apply(lambda row: row['Height (cm)'], axis=1)
        df['fv3'] = df.apply(lambda row: row['Weight (kg)'], axis=1)
        df['fv4'] = df.apply(lambda row: 1 if row['Race'] == 'Asian' else 0, axis=1)
        df['fv5'] = df.apply(lambda row: 1 if row['Race'] == 'Black or African American' else 0, axis=1)
        df['fv6'] = df.apply(lambda row: 1 if row['Race'] == 'Unknown' else 0, axis=1)
        df['fv7'] = df.apply(lambda row: 1 if 1 in [row['Carbamazepine (Tegretol)'], row['Phenytoin (Dilantin)'], row['Rifampin or Rifampicin']] else 0, axis=1)
        df['fv8'] = df.apply(lambda row: 1 if row['Amiodarone (Cordarone)'] else 0, axis=1)
        feature_vector_cols = ['fv{}'.format(i) for i in range(1,9)]
        """
        for i in range(1,9):
            print(df['fv{}'.format(i)].describe())
        """
    elif feature_vector_type == 'fv1':
        needed_cols = [TRUE_DOSAGE_LABEL_COLUMN_NAME, TRUE_DOSAGE_LABEL_COLUMN_NAME, 'Age', 'Height (cm)', 'Weight (kg)', 'Race', \
         'Carbamazepine (Tegretol)', 'Phenytoin (Dilantin)', 'Rifampin or Rifampicin', 'Amiodarone (Cordarone)', \
         'Current Smoker', 'Diabetes', 'Congestive Heart Failure and/or Cardiomyopathy', 'Valve Replacement', \
         'Aspirin']
        cols_to_drop = cols = [col for col in df.columns if col not in needed_cols]
        df = df.drop(cols_to_drop, axis=1)
        #df = df.groupby(df.columns, axis = 1).transform(lambda x:  x.fillna(x.mode()) if isinstance(x,str) else)
        df['fv1'] = df.apply(lambda row: int(row['Age'][0]) if isinstance(row['Age'],str) else 0, axis=1)
        df['fv2'] = df.apply(lambda row: row['Height (cm)'], axis=1)
        df['fv3'] = df.apply(lambda row: row['Weight (kg)'], axis=1)
        df['fv4'] = df.apply(lambda row: 1 if row['Race'] == 'Asian' else 0, axis=1)
        df['fv5'] = df.apply(lambda row: 1 if row['Race'] == 'Black or African American' else 0, axis=1)
        df['fv6'] = df.apply(lambda row: 1 if row['Race'] == 'Unknown' else 0, axis=1)
        df['fv7'] = df.apply(lambda row: 1 if 1 in [row['Carbamazepine (Tegretol)'], row['Phenytoin (Dilantin)'], row['Rifampin or Rifampicin']] else 0, axis=1)
        df['fv8'] = df.apply(lambda row: row['Amiodarone (Cordarone)'], axis=1)
        df['fv9'] = df.apply(lambda row: row['Current Smoker'], axis=1)
        df['fv10'] = df.apply(lambda row: row['Diabetes'], axis=1)
        df['fv11'] = df.apply(lambda row: row['Congestive Heart Failure and/or Cardiomyopathy'], axis=1)
        df['fv12'] = df.apply(lambda row: row['Valve Replacement'], axis=1)
        df['fv12'] = df.apply(lambda row: row['Aspirin'], axis=1)
        feature_vector_cols = ['fv{}'.format(i) for i in range(1,13)]
        """
        for i in range(1,9):
            print(df['fv{}'.format(i)].describe())
        """


    #print("df[feature_vector_cols].head()", df[feature_vector_cols][:15])

    context_vectors = df[feature_vector_cols].fillna(0).to_numpy()
    print("context_vectors shape: {}".format(context_vectors.shape))
    return context_vectors


def get_context_and_labels(warfarin_filename, feature_vector_type):
    warfarin_df = load_data(warfarin_filename)
    labels = warfarin_df[TRUE_DOSAGE_LABEL_COLUMN_NAME].to_numpy()
    context_df = warfarin_df.drop(TRUE_DOSAGE_LABEL_COLUMN_NAME, axis=1)
    context_df = context_df.drop(TRUE_DOSAGE_COLUMN_NAME, axis=1)
    context_vectors = get_context_vectors(context_df, feature_vector_type)
    return context_vectors, labels


def get_optimal_betas(context_vectors, labels, num_actions):
    beta_list = []
    num_vectors = context_vectors.shape[0]
    for i in range(num_actions):
        indices = []
        for j in range(num_vectors):
            if get_label_index_from_label(labels[j]) == i:
                indices.append(j)
        A = context_vectors[indices].astype(float)
        b = np.array(list(map(get_label_index_from_label, labels[indices].tolist()))).astype(float)
        x, _, _, _ = np.linalg.lstsq(A,b, rcond=None)
        beta_list.append(x)
    return beta_list

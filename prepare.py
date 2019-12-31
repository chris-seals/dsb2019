import pandas as pd 
import numpy as np
import re

from typing import Any
#from collections import Counter
from tqdm.notebook import tqdm
from sklearn.preprocessing import OneHotEncoder

def remove_dead_weight(df, train_labels, test_set=False):
    #df = df[df['world'] != 'NONE']
    
    # filtering by ids that took assessments
    ids_w_assessments = df[df['type'] == 'Assessment']['installation_id'].drop_duplicates()
    df = df[df['installation_id'].isin(ids_w_assessments)]
    
    # If training set then make sure the installation ids are in the labels and remove assements not in the labels
    if test_set == False:
        # drop data whose installation does not contain any scored assessments in train_labels
        df = df[df['installation_id'].isin(train_labels['installation_id'].unique())]
        
        assessments = df[df.type == 'Assessment']
        assessments = assessments[~assessments.game_session.isin(train_labels.game_session)]
        df = df[~df.game_session.isin(assessments.game_session)]
        df.reset_index(drop=True, inplace=True)
        
    return df


def add_datepart(df: pd.DataFrame, field_name: str,
                 prefix: str = None, drop: bool = True, time: bool = True, date: bool = True):
    """
    Helper function that adds columns relevant to a date in the column `field_name` of `df`.
    from fastai: https://github.com/fastai/fastai/blob/master/fastai/tabular/transform.py#L55
    """
    field = df[field_name]
    prefix = ifnone(prefix, re.sub('[Dd]ate$', '', field_name))
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Is_month_end', 'Is_month_start']
    if date:
        attr.append('Date')
    if time:
        attr = attr + ['Hour', 'Minute']
    for n in attr:
        df[prefix + n] = getattr(field.dt, n.lower())
    if drop:
        df.drop(field_name, axis=1, inplace=True)
    return df


def ifnone(a: Any, b: Any) -> Any:
    """`a` if `a` is not None, otherwise `b`.
    from fastai: https://github.com/fastai/fastai/blob/master/fastai/core.py#L92"""
    return b if a is None else a


def encode_col(train, test, col):
    ohe = OneHotEncoder()
    unique_list = list(set(train[col].unique()).union(set(test[col].unique())))
    feature_arr = ohe.fit_transform(np.array(unique_list).reshape(-1, 1)).toarray()
    new_train_arr = ohe.transform(train[[col]]).toarray()
    new_test_arr = ohe.transform(test[[col]]).toarray()
    labels = sorted([col + '_' + s for s in unique_list])
    new_train = pd.DataFrame(new_train_arr, columns=labels)
    new_test = pd.DataFrame(new_test_arr, columns=labels)
    
    return pd.concat([train, new_train], axis=1), pd.concat([train, new_test], axis=1)


def flatten_add_features(sample):
    title_cols = [col for col in sample if col.startswith('title_')]
    world_cols = [col for col in sample if col.startswith('world_')]
    sum_cols = title_cols + world_cols
    
    reduced_sample = sample.iloc[-1].copy()
    # sum column counts
    reduced_sample.update(sample[sum_cols].sum())
    # correct events against incorrect events
    reduced_sample['total_event_count'] = sample['event_count'].sum()
    reduced_sample['avg_event_count'] = sample['event_count'].mean()

    reduced_sample['total_game_time'] = sample['game_time'].sum()

    # find how much time was spent
    num_games = sample[sample['game_time'] != 0]['game_session'].nunique()

    if num_games == 0:
        reduced_sample['avg_game_time'] = 0
    else:
        reduced_sample['avg_game_time'] = reduced_sample['total_game_time'] / num_games
        
    return reduced_sample


def process_data(df, test_set=False):
    compiled_data = []

    for ins_id, sample in tqdm(df.groupby('installation_id', sort=False), total=df.installation_id.nunique()):

        sample.reset_index(drop=True, inplace=True)

        assessments_array = sample[sample['type'] == 'Assessment']['game_session'].unique()

        if test_set:
            compiled_data.append(flatten_add_features(sample))
        else:    
            for i, assessment_session_id in enumerate(assessments_array):
                # Grab the row location for that session's event code 2000
                truncate_index = sample.index[(sample['game_session'] == assessment_session_id) & (sample['event_code'] == 2000)][0]

                # Now get a slice of the user_sample from the beginning to the truncated assessment start
                truncated_user_sample = sample.iloc[:truncate_index + 1].copy()
                truncated_user_sample['installation_id_slice'] = ins_id + '_slice_' + str(i)

                compiled_data.append(flatten_add_features(truncated_user_sample))
    
    return compiled_data
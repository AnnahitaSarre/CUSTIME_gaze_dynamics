#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUSTIME fonctions for cleaning
"""

#%% Import modules

import os
import numpy as np
import pandas as pd
from tqdm import tqdm


#%% General cleaning functions

def clean_names(args):
    # Carefull with args.subjects, use this line if needed:
    # sub_name = args.group if isinstance(subject, list) and args.group.split('_', 1)[0] in ('all', '1', '2', '3') else subject
    if args.run == '*':
        run_name = 'all'
    else:
        run_name = args.run
    task_name = '+'.join(args.task)
    return run_name, task_name


#%% Eye-tracking cleaning functions

def keep_common_videos(args, subject, video_files):
    print('Control participant(s) included: keeping only stims common to all')
    if args.stimulus == 'words':
        common_stim = ['word_h0_03_pseudo.mp4', 'word_h0_04_pseudo.mp4', 'word_h0_06.mp4', 'word_h0_09.mp4', 'word_h0_11.mp4', 'word_h0_14.mp4', 'word_h0_17.mp4', 'word_h0_18.mp4', 'word_h0_19.mp4', 'word_h0_23.mp4', 'word_h0_29.mp4', 'word_h0_31.mp4',
                       'word_h1_01_pseudo.mp4', 'word_h1_03.mp4', 'word_h1_04_pseudo.mp4', 'word_h1_09.mp4', 'word_h1_15.mp4', 'word_h1_21.mp4', 'word_h1_23.mp4', 'word_h1_25.mp4', 'word_h1_27.mp4', 'word_h1_28.mp4', 'word_h1_32.mp4', 'word_h1_37.mp4',
                       'word_h2_02.mp4', 'word_h2_02_pseudo.mp4', 'word_h2_04.mp4', 'word_h2_04_pseudo.mp4', 'word_h2_07.mp4', 'word_h2_09.mp4', 'word_h2_10.mp4', 'word_h2_11.mp4', 'word_h2_12.mp4', 'word_h2_18.mp4', 'word_h2_33.mp4', 'word_h2_36.mp4',
                       'word_l0_03_pseudo.mp4', 'word_l0_05_pseudo.mp4', 'word_l0_09.mp4', 'word_l0_12.mp4', 'word_l0_17.mp4', 'word_l0_18.mp4', 'word_l0_24.mp4', 'word_l0_29.mp4', 'word_l0_33.mp4', 'word_l0_36.mp4', 'word_l0_38.mp4', 'word_l0_40.mp4',
                       'word_l1_01.mp4', 'word_l1_02.mp4', 'word_l1_03.mp4', 'word_l1_04.mp4', 'word_l1_04_pseudo.mp4', 'word_l1_05.mp4', 'word_l1_05_pseudo.mp4', 'word_l1_06.mp4', 'word_l1_07.mp4', 'word_l1_08.mp4', 'word_l1_09.mp4', 'word_l1_10.mp4',
                       'word_l2_01.mp4', 'word_l2_01_pseudo.mp4', 'word_l2_02_pseudo.mp4', 'word_l2_07.mp4', 'word_l2_11.mp4', 'word_l2_12.mp4', 'word_l2_16.mp4', 'word_l2_27.mp4',  'word_l2_29.mp4', 'word_l2_35.mp4', 'word_l2_36.mp4', 'word_l2_38.mp4']
        video_files_temp = []
        for video_file in video_files:
            if os.path.basename(video_file) in common_stim:
                video_files_temp.append(video_file)
        video_files = video_files_temp
    if args.stimulus == 'sentences':
        common_stim = [f'sent_{num}.mp4' for num in range(71,101)]
        video_files_temp = []
        for video_file in video_files:
            if os.path.basename(video_file) in common_stim:
                video_files_temp.append(video_file)
        video_files = video_files_temp
    return video_files


def keep_common_stims(args, all_stims_df_list):
    print('Control participant(s) included: keeping only stims common to all')
    if args.stimulus == 'words':
        common_stim = ['word_h0_03_pseudo', 'word_h0_04_pseudo', 'word_h0_06', 'word_h0_09', 'word_h0_11', 'word_h0_14', 'word_h0_17', 'word_h0_18', 'word_h0_19', 'word_h0_23', 'word_h0_29', 'word_h0_31',
                       'word_h1_01_pseudo', 'word_h1_03', 'word_h1_04_pseudo', 'word_h1_09', 'word_h1_15', 'word_h1_21', 'word_h1_23', 'word_h1_25', 'word_h1_27', 'word_h1_28', 'word_h1_32', 'word_h1_37',
                       'word_h2_02', 'word_h2_02_pseudo', 'word_h2_04', 'word_h2_04_pseudo', 'word_h2_07', 'word_h2_09', 'word_h2_10', 'word_h2_11', 'word_h2_12', 'word_h2_18', 'word_h2_33', 'word_h2_36',
                       'word_l0_03_pseudo', 'word_l0_05_pseudo', 'word_l0_09', 'word_l0_12', 'word_l0_17', 'word_l0_18', 'word_l0_24', 'word_l0_29', 'word_l0_33', 'word_l0_36', 'word_l0_38', 'word_l0_40',
                       'word_l1_01', 'word_l1_02', 'word_l1_03', 'word_l1_04', 'word_l1_04_pseudo', 'word_l1_05', 'word_l1_05_pseudo', 'word_l1_06', 'word_l1_07', 'word_l1_08', 'word_l1_09', 'word_l1_10',
                       'word_l2_01', 'word_l2_01_pseudo', 'word_l2_02_pseudo', 'word_l2_07', 'word_l2_11', 'word_l2_12', 'word_l2_16', 'word_l2_27',  'word_l2_29', 'word_l2_35', 'word_l2_36', 'word_l2_38']
    if args.stimulus == 'sentences':
        common_stim = [f'sent_{num}' for num in range(71,101)]
    for i_sub_df, sub_df in tqdm(enumerate(all_stims_df_list), total=len(all_stims_df_list)):
        all_stims_df_list[i_sub_df] = sub_df[sub_df['stim'].isin(common_stim)]
    return all_stims_df_list
    
    
def compute_data_loss_reject(args, stim_epoch_df):
    # Replace values indicating data loss by nan values: 250 and 100 because sure that these values indicate loss of data, and I can't put 0 directly because MNE transforms values
    stim_epoch_df['et_x'] = np.where(stim_epoch_df['et_x'] > 250, stim_epoch_df['et_x'], np.nan)
    stim_epoch_df['et_y'] = np.where(stim_epoch_df['et_y'] > 250, stim_epoch_df['et_y'], np.nan)
    stim_epoch_df['et_pupil'] = np.where(stim_epoch_df['et_pupil'] > 100, stim_epoch_df['et_pupil'], np.nan)
    qc_stim_row = {}
    # Detect data points where the subject was not looking at the video
    for index in stim_epoch_df.index:
        if pd.notna(stim_epoch_df['et_x'].iloc[index]) and int(stim_epoch_df['et_x'].iloc[index]) not in range(0,2560):
            stim_epoch_df.loc[index,'et_x'] = np.nan
        if pd.notna(stim_epoch_df['et_y'].iloc[index]) and int(stim_epoch_df['et_y'].iloc[index]) not in range(0,1600):
            stim_epoch_df.loc[index,'et_y'] = np.nan
    # Compute the loss rate
    loss_rate = []
    for col in ['et_x', 'et_y', 'et_pupil']:
        loss_rate.append(((stim_epoch_df[col].isna()).sum() / len(stim_epoch_df)) * 100)
    loss_rate = max(loss_rate)
    qc_stim_row['loss_rate'] = loss_rate
    qc_stim_row['et_pupil'] = stim_epoch_df['et_pupil'].to_list()
    # Reject epoch if needed
    if loss_rate > args.loss_rate_criteria:
        stim_epoch_df = None
        qc_stim_row['rejected'] = 1
    else:
        qc_stim_row['rejected'] = 0
    return stim_epoch_df, qc_stim_row

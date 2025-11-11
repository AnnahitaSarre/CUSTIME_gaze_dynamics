#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUSTIME fonctions for statistical analyses
"""


#%% Import modules

import os
import mne
import re
import sys
import pickle
from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from math import sqrt
import pingouin as pg
from functools import partial
from itertools import groupby, combinations
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point, Polygon
from utils.decod import load_decod_scores, load_encod_betas
from utils.eyetracking import annot_of_timepoint
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
from mne.stats import spatio_temporal_cluster_1samp_test, spatio_temporal_cluster_test, permutation_cluster_1samp_test, permutation_cluster_test
from utils.cleaning import clean_names, keep_common_videos, keep_common_stims, compute_data_loss_reject
from utils.stats_details import proportion_aoi_hits_detailed_analysis, proportion_correlation


#%% Retrieve individual participant IDs

def retrieve_ids(args, data_type):
    if any('Sentences' in task for task in args.task):
        args.stimulus = 'sentences'
    elif any('Words' in task for task in args.task):
        args.stimulus = 'words'
    else:
        args.stimulus = 'syllables'
    # If processing all participants or one or two group(s), retrieve participant IDs
    # Only include participants who have relevant files for the task
    if str(args.subjects[0]) in ('*','1','2','3'):
        subjects_files = []
        args.group = args.subjects
        if args.group == ['*']:
            args.group = 'all'
            excluded_subjects = '_'.join([sub for sub in args.subjects_to_exclude if sub not in ['phantom', 'pilote']])
            # Create the final output by appending the excluded subjects to the base string
            args.group = f'{args.group}_excl_{excluded_subjects}' if excluded_subjects else args.group
        for subject in args.subjects:
            for task in args.task:
                elif data_type == 'beh':
                    subjects_files.append(glob(os.path.join(args.path2data,f'sub-{subject[0]}*?/ses-{args.session_day}/beh/sub-{subject[0]}*?_task-{args.task}_run-{args.run}_beh.csv')))
                elif data_type == 'et':
                    subjects_files.append(glob(os.path.join(args.path2data,f'sub-{subject[0]}*?/ses-{args.session_day}/beh/sub-{subject[0]}*?_ses-{args.session_day}_task-{task}_run-{args.run}.asc')))
        subjects_files = [file for task_files in subjects_files for file in task_files] # Flatten the list of files
        elif data_type == 'et':
            args.subjects = [subjects_files[f][subjects_files[f].find('sub-')+4:subjects_files[f].find('/ses')] for f in range(len(subjects_files))]
        args.subjects = [sub for sub in set(args.subjects) if not args.subjects_to_exclude.count(sub)] # Keep one occurence of each partcipant ID and remove undesired participants
        args.subjects.sort() # Order list of participants
    print(f'Considering {len(args.subjects)} participant(s) with corresponding preprocessed raws.')
    if data_type == 'et':
        if 'Sentences' in args.task:
            args.subjects = [subject for subject in args.subjects if subject != '320']
            print('Excluding subject 320 because of too much stimuli loss.')
            args.subjects = [subject for subject in args.subjects if subject != '33']
            print('Excluding subject 33 because was not looking at the coder nor at the fixation cross.')
            args.subjects = [subject for subject in args.subjects if subject != '12']
            print('Excluding subject 12 because outlier.')
        elif 'Words' in args.task:
            args.subjects = [subject for subject in args.subjects if subject != '13']
            print('Excluding subject 13 because of file loss in the Words experiment.')
            args.subjects = [subject for subject in args.subjects if subject != '111']
            args.subjects = [subject for subject in args.subjects if subject != '121']
            print('Excluding subjects 111 and 121 because of too much stimuli loss.')
            args.subjects = [subject for subject in args.subjects if subject != '33']
            print('Excluding subject 33 because was not looking at the coder nor at the fixation cross.')
            args.subjects = [subject for subject in args.subjects if subject != '32']
            print('Excluding subject 32 because was not looking at the coder.')
            args.subjects = [subject for subject in args.subjects if subject != '12']
            print('Excluding subject 12 because outlier.')
    if data_type == 'beh':
        print('no exclusion for now')
    print(f'Considering {len(args.subjects)} participant(s) with corresponding preprocessed raws.')

        
#%% Retrieve epochs from multiple participants

def retrieve_epochs(args, data_type):
    run_name, task_name = clean_names(args)
    epochs, et_annotations = [None]*len(args.subjects), [None]*len(args.subjects)
    if "vs" in args.contrast: # For contrasts containing group comparison, fetch the base contrast of each participant
        groups = re.search(r'^(.*)_([0-9](?:\+[0-9])?)(?:vs)([0-9](?:\+[0-9])?)$', args.contrast)
        contrast = groups.group(1)
    else:
        contrast = args.contrast
    for i_subject, subject in enumerate(args.subjects):
        epochs_dir = os.path.join(args.path2deriv,f'epochs/sub-{subject}/ses-{args.session_day}/{task_name}/{data_type}/{contrast}')
        epochs_file = f'contrast-{contrast}_sub-{subject}_task-{task_name}_run-{run_name}_epo.fif'
        epochs[i_subject] = mne.read_epochs(os.path.join(epochs_dir,epochs_file))
        if data_type == 'beh':
            et_annots_file = epochs_file.replace('epo.fif', 'annot.pkl')
            et_annotations[i_subject] = pickle.load(open(os.path.join(epochs_dir,et_annots_file), 'rb'))
    return epochs, et_annotations


#%% Retrieve ET epochs

def select_stim_epoch(args, subject, epochs, et_annotations, stim_name):
    print('Selecting relevant epochs...')
    if stim_name.startswith('word'):
        if 'pseudo' in stim_name:
            stim_pseudo = str(1)
        else:
            stim_pseudo = str(0)
        stim_freq = stim_name.split('_')[1][0]
        stim_mismatch = str(stim_name.split('_')[1][1])
        stim_num_in_cat = str(int(stim_name.split('_')[2]))
        query = f"event_type=='word_onset' and pseudo=={stim_pseudo} and freq=='{stim_freq}' and mismatch=={stim_mismatch} and num_in_cat=={stim_num_in_cat}"
    elif stim_name.startswith('sent'):
        num_sent = int(re.findall(r'\d+', stim_name)[0])
        query = f"event_type=='sentence_onset' and num_sentence=={num_sent}"
    qc_df_stim = pd.DataFrame(columns = ['subject', 'stimulus', 'loss_rate', 'rejected'])
    if type(subject) is not list: # 1st order analysis
        if stim_name.startswith('word'):
            query_metadata_idx = (epochs.metadata['event_type']=='word_onset') & (epochs.metadata['pseudo']==stim_pseudo) & \
                (epochs.metadata['freq']==stim_freq) & (epochs.metadata['mismatch']==stim_mismatch) & \
                (epochs.metadata['num_in_cat']==stim_num_in_cat)
        elif stim_name.startswith('sent'):
            query_metadata_idx = (epochs.metadata['event_type']=='sentence_onset') & (epochs.metadata['num_sentence']==num_sent)
        stim_epoch_df = epochs[query].to_data_frame()
        stim_epoch_df['et_y'] = stim_epoch_df['et_y']*(1600/1440) # Convert the coordinates as calculated by the ET (screen was heigh = 1440) to adapt them to the actual video format (1600)
        if et_annotations is not None: # Some scripts use this fuction without using et_annotations, so 'None' is given as an argument
            epoch_index = query_metadata_idx.idxmax()
            stim_annot = et_annotations[epoch_index]
        else:
            stim_annot = None
        stim_epoch_df, qc_stim_row = compute_data_loss_reject(args, stim_epoch_df)
        qc_stim_row['subject'], qc_stim_row['stimulus'] = subject, stim_name
        qc_df_stim = qc_df_stim.concat(qc_stim_row, ignore_index=True)
    if type(subject) is list: # 2nd order analysis
        stim_epoch_df = []
        stim_annot = []
        with pd.option_context('mode.chained_assignment', None): # To desactivate SettingWithCopyWarning
            for i_subject, sub in enumerate(subject):
                if stim_name.startswith('word'):
                    query_metadata_idx = (epochs[i_subject].metadata['event_type']=='word_onset') & (epochs[i_subject].metadata['pseudo']==stim_pseudo) & \
                        (epochs[i_subject].metadata['freq']==stim_freq) & (epochs[i_subject].metadata['mismatch']==stim_mismatch) & \
                        (epochs[i_subject].metadata['num_in_cat']==stim_num_in_cat)
                elif stim_name.startswith('sent'):
                    query_metadata_idx = (epochs[i_subject].metadata['event_type']=='sentence_onset') & (epochs[i_subject].metadata['num_sentence']==num_sent)
                stim_epoch_df_ind = epochs[i_subject][query].to_data_frame()
                stim_epoch_df_ind['et_y'] = stim_epoch_df_ind['et_y']*(1600/1440)
                if et_annotations is not None: # Some scripts use this fuction without using et_annotations, so 'None' is given as an argument
                    epoch_index = query_metadata_idx.idxmax()
                    stim_annot.append(et_annotations[i_subject][epoch_index])
                else:
                    stim_annot.append(None)
                stim_epoch_df_ind, qc_stim_row = compute_data_loss_reject(args, stim_epoch_df_ind)
                qc_stim_row['subject'], qc_stim_row['stimulus'] = sub, stim_name
                stim_epoch_df.append(stim_epoch_df_ind)
                pd.concat([qc_df_stim, pd.DataFrame([qc_stim_row])], ignore_index=True)
    print('Epochs selected')
    return stim_epoch_df, qc_df_stim, stim_annot


#%% Perform eye-tracking quality check

def compute_precision_accuracy(args, subjects, epochs, et_annotations):
    print('### Computing precision and accuracy at fixation cross')
    print('Only keeping timepoints where the eye is performing a fixation')
    precision_accuracy_df = pd.DataFrame(columns = ['subject', 'precision', 'accuracy'])
    for i_subject, sub in tqdm(enumerate(subjects), total=len(subjects)):
        ## Fetch fixation data and corresponding annotations
        # There was a jitter during display, for all exp: 0.3<fixation_duration<0.9
        fixation_indices = [index for index, el in enumerate(epochs[i_subject].events[:, 2]) if el == 1] # el == 1 <=> fixation cross
        fixation_epochs_df = []
        fixation_epochs = epochs[i_subject][fixation_indices]
        for i_epo in range(len(fixation_epochs)):
            ## Shorten time window to match actual fixation display duration
            fixation_duration = float('{:.3f}'.format(float(fixation_epochs[i_epo].metadata['fixation_duration'])))
            fixation_duration = int(fixation_duration*1000) # *1000 to convert s to nb of samples (1000Hz acquisition)
            fixation_epoch_df = fixation_epochs[i_epo].to_data_frame()
            fixation_epoch_df = fixation_epoch_df.iloc[:fixation_duration, :]
            ## Only keep annotations occuring during fixation cross
            stim_annot = [dic for dic in et_annotations[i_subject] if dic['epoch_idx'] == fixation_indices[i_epo]][0]
            annot_tuples = list(zip(stim_annot['annotations'], stim_annot['onsets'], stim_annot['durations']))
            annot_tuples = [(annot, onset, duration) for (annot, onset, duration) in annot_tuples
                            if onset < fixation_duration/1000-0.001]
            stim_annot['annotations'] = [tup[0] for tup in annot_tuples]
            stim_annot['onsets'] = [tup[1] for tup in annot_tuples]
            stim_annot['durations'] = [tup[2] for tup in annot_tuples]
            ## Only keep timepoints where the eye is performing its longest fixation
            if 'fixation' in stim_annot['annotations']:
                fixation_tuples = [(duration, idx) for idx, (annot, onset, duration)
                                    in enumerate(annot_tuples) if annot == 'fixation']
                longest_fix_index = max(fixation_tuples, key=lambda x: x[0])[1]
                fixation_epoch_df['keep'] = fixation_epoch_df['time'].apply(lambda timepoint: timepoint >= stim_annot['onsets'][longest_fix_index] and timepoint < stim_annot['onsets'][longest_fix_index]+stim_annot['durations'][longest_fix_index])
            else:
                continue
            fixation_epoch_df = fixation_epoch_df[fixation_epoch_df['keep']]
            fixation_epoch_df.drop(columns=['keep'], inplace=True)
            fixation_epochs_df.concat(fixation_epoch_df)
        fixations_df = pd.concat(fixation_epochs_df)
        ## For each sample, compute distance from the fixation cross
        if args.stimulus.startswith('sent'):
            cross_coord = (1280,520)
        elif args.stimulus.startswith('word'):
            cross_coord = (1270,520)
        elif args.stimulus.startswith('syllreading'): # !!! won't work, the pipeline is't adapted to syllable stims
            cross_coord = (1280,720)
        elif args.stimulus.startswith('syllcs'): # !!! same
            cross_coord = (1280,738)
        distances_from_cross_px = (abs(fixations_df['et_x']-cross_coord[0]), abs(fixations_df['et_y']-cross_coord[1]))
        ## Convert from px to mm to visual angles
        px_to_mm = 700/2560 # Screen width in mm / screen width in px (same ratio for height, tested)
        distances_from_cross = (np.degrees(np.arctan(distances_from_cross_px[0]*px_to_mm/600)),
                                np.degrees(np.arctan(distances_from_cross_px[1]*px_to_mm/600)))
        ## Delete samples where either the x or the y coord is too far from the cross
        distances_from_cross = list(distances_from_cross)
        mask = (distances_from_cross[0] > 5) | (distances_from_cross[1] > 5)
        distances_from_cross[0] = distances_from_cross[0][~mask]
        distances_from_cross[1] = distances_from_cross[1][~mask]
        distances_from_cross = tuple(distances_from_cross)
        print(f's{sub}: {(mask).sum()} samples too distant from the cross, dropping them')
        ## Compute precision and accuracy
        precision_tuple = (np.sqrt(np.mean(np.diff(distances_from_cross[0])**2)),
                           np.sqrt(np.mean(np.diff(distances_from_cross[1])**2)))
        precision = sqrt(precision_tuple[0]**2 + precision_tuple[1]**2)
        accuracy_tuple = (distances_from_cross[0].mean(), distances_from_cross[1].mean())
        accuracy = sqrt(accuracy_tuple[0]**2 + accuracy_tuple[1]**2)
        precision_accuracy_df = precision_accuracy_df.concat({'subject': sub,
                                                              'precision': precision,
                                                              'accuracy': accuracy}, ignore_index=True)
    ## Compute group data
    precision_all_subs_mean = precision_accuracy_df['precision'].mean()
    precision_all_subs_max = precision_accuracy_df['precision'].max()
    precision_all_subs_min = precision_accuracy_df['precision'].min()
    accuracy_all_subs_mean = precision_accuracy_df['accuracy'].mean()
    accuracy_all_subs_max = precision_accuracy_df['accuracy'].max()
    accuracy_all_subs_min = precision_accuracy_df['accuracy'].min()
    print('For all participants:')
    print(f'Precision: mean={precision_all_subs_mean:.2f}, min={precision_all_subs_min:.2f}, max={precision_all_subs_max:.2f}')
    print(f'Accuracy: mean={accuracy_all_subs_mean:.2f}, min={accuracy_all_subs_min:.2f}, max={accuracy_all_subs_max:.2f}')
    ## Plot results
    fig, ax = plt.subplots(figsize=(14, 6))
    precision_accuracy_df.set_index('subject')[['precision', 'accuracy']].plot(ax=ax, kind='bar', width=0.8)
    ax.set_title(f'Mean Precision and Accuracy per Participant (stim-{args.stimulus}, sub-{args.group})')
    ax.set_ylabel('Mean Precision and Accuracy (°)')
    ax.set_xlabel('Participant')
    ax.legend(title='Metric')
    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(args.path2figures, f'et_quality_check/precision_accuracy_stim-{args.stimulus}_sub-{args.group}.png'))


def detect_pursuits_in_sub(i_subject, sub, epochs_sub, pursuit_list, args, et_annotations):
    ## Fetch fixation data and corresponding annotation
    durations_df = pd.read_csv(f'utils/durations_{args.stimulus}.csv')
    pursuit_proportions_list = []
    for i_epo in range(len(epochs_sub)):
        epoch_metadata = epochs_sub.metadata.iloc[i_epo]
        if epoch_metadata['event_type'] == 'word_onset' or epoch_metadata['event_type'] == 'sentence_onset':
            ## Shorten time window to match actual stimulus display duration
            if args.stimulus == 'sentences':
                stim_name = f"sent_{'{:02}'.format(int(epoch_metadata['num_sentence']))}"
            else:
                stim_name = f"word_{epoch_metadata['freq']}{int(epoch_metadata['mismatch'])}_{int(epoch_metadata['num_in_cat']):02d}"
                if epoch_metadata['pseudo'] == 1:
                    stim_name += '_pseudo'
            stim_duration = durations_df[durations_df['stim'] == stim_name]['duration'].iloc[0]
            epochs_sub_df = epochs_sub[i_epo].to_data_frame()
            epochs_sub_df = epochs_sub_df.iloc[:stim_duration, :]
            ## Only keep annotations occuring during stimulus
            stim_annot = et_annotations[i_subject][i_epo]
            annot_tuples = list(zip(stim_annot['annotations'], stim_annot['onsets'], stim_annot['durations']))
            annot_tuples = [(annot, onset, duration) for (annot, onset, duration) in annot_tuples
                            if onset < stim_duration/1000-0.001]
            stim_annot['annotations'] = [tup[0] for tup in annot_tuples]
            stim_annot['starts'] = [int(tup[1]) for tup in annot_tuples]
            stim_annot['ends'] = [int(tup[1]+tup[2]) for tup in annot_tuples]
            ## Calculate the distance between first and last coordinate of each fixation of the epoch and detect smooth pursuits
            nb_pursuit_samples = 0
            for i_annot, annot in enumerate(stim_annot['annotations']):
                if annot == 'fixation':
                    start_idx, end_idx = stim_annot['starts'][i_annot], stim_annot['ends'][i_annot]
                    dx = abs(epochs_sub_df['et_x'].iloc[start_idx]-epochs_sub_df['et_x'].iloc[end_idx-1])
                    dy = abs(epochs_sub_df['et_y'].iloc[start_idx]-epochs_sub_df['et_y'].iloc[end_idx-1])
                    # Convert from px to mm to visual angles
                    px_to_mm = 700/2560 # Screen width in mm / screen width in px (same ratio for height, tested)
                    dx, dy = np.degrees(np.arctan(dx*px_to_mm/600)), np.degrees(np.arctan(dy*px_to_mm/600))
                    dist_start_to_end = np.sqrt(dx**2 + dy**2) # Calculate the Euclidean distance
                    if dist_start_to_end > 2:
                        nb_pursuit_samples = nb_pursuit_samples + stim_annot['durations'][i_annot]*1000
            pursuit_proportion_epoch = nb_pursuit_samples/stim_duration*100
            # print(nb_pursuit_samples, stim_duration, pursuit_proportion_epoch)
            pursuit_proportions_list.append(pursuit_proportion_epoch)
    pursuit_proportion_mean = np.mean(pursuit_proportions_list)
    pursuit_list.append({'group': sub[0], 'subject': sub, 'pursuit_proportion': pursuit_proportion_mean})
    
    
def detect_smooth_pursuits(args, subjects, epochs, et_annotations): # Computed on all available data, including rejected eye-tracking trials
    print('### Detecting smooth pursuits')
    compute_func = partial(detect_pursuits_in_sub, args=args, et_annotations=et_annotations)
    pursuit_list = mp.Manager().list()
    with mp.Pool(mp.cpu_count()) as pool:
        list(tqdm(pool.starmap(compute_func, [(i_subject, sub, epochs[i_subject], pursuit_list)
                                              for i_subject, sub in enumerate(subjects)])))
    pool.close()
    pool.join()
    pursuit_df = pd.DataFrame(list(pursuit_list))
    
    ## Display and plot results
    print('## Proportion of smooth pursuits')
    print(f"All participants:\nmean={pursuit_df['pursuit_proportion'].mean():.2f}; std={pursuit_df['pursuit_proportion'].std():.2f}; min={pursuit_df['pursuit_proportion'].min():.2f}; max={pursuit_df['pursuit_proportion'].max():.2f}")
    print('Groups:')
    print(pursuit_df.groupby('group')['pursuit_proportion'].describe())
    print(f"Anova between groups:\n{pg.anova(data=pursuit_df, dv='pursuit_proportion', between='group')}")
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(y='pursuit_proportion', hue='group',
                data=pursuit_df, palette='pastel')
    sns.stripplot(y='pursuit_proportion', hue='group',dodge=True,
                  data=pursuit_df, palette='dark', legend=False)
    ax.set_title(f'Proportion of smooth pursuits (stimuli: {args.stimulus}; criterion: fixations with > 2° drifting)')
    ax.set_ylabel('Proportion of smooth pursuits (% stimulus duration)')
    ax.set_xlabel('Group')
    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(args.path2figures, f'et_quality_check/pursuit_stim-{args.stimulus}.png'))

    
def et_quality_check(args, subjects, epochs):
    print(f'\n## Performing quality check for subject(s) {args.group}')
    if args.stimulus == 'words' or args.stimulus == 'sentences':
        video_files = glob(f'utils/stim_videos/videos/{args.stimulus}/*.mp4')
    else:
        video_files = glob(f'utils/stim_videos/videos/*/{args.stimulus}.mp4')
    # If considering group 3, only consider stimuli that were shown to the 3 groups
    if any([sub.startswith('3') for sub in subjects]):
       video_files = keep_common_videos(args, subjects, video_files)
    if type(subjects) is list:
        nb_subjects = len(subjects)
    else:
        nb_subjects = 1
    qc_df = pd.DataFrame()
    for video_file in tqdm(video_files):
        stim_name = os.path.splitext(os.path.basename(video_file))[0]
        print(f'\nAnalysing stimulus {stim_name}...')
        ## Create query and perform quality check on each stim*participant
        _, qc_df_stim, _ = select_stim_epoch(args, subjects, epochs, None, stim_name)
        qc_df = qc_df.concat(qc_df_stim, ignore_index=True)
    ## Test differences in the nb of rejected trials across groups and types of stim
    qc_df['group'] = qc_df.apply(lambda x: x['subject'][0], axis=1)
    if args.stimulus == 'words':
        qc_df['freq'] = qc_df.apply(lambda x: x['stimulus'][5], axis=1)
        qc_df['mismatch'] = qc_df.apply(lambda x: x['stimulus'][6], axis=1)
        qc_df['pseudo'] = qc_df.apply(lambda x: 1 if 'pseudo' in x['stimulus'] else 0, axis=1)
    if args.stimulus == 'sentences':
        qc_df['predictable'] = qc_df.apply(lambda x: 1 if (int(x['stimulus'].split('_')[1]) % 2) == 0 else 0, axis=1)
    # Chi2 tests
    qc_df_accepted = qc_df[qc_df['rejected']==0]
    if args.stimulus == 'words':
        freq_observed = qc_df_accepted['freq'].value_counts().loc[['h', 'l']]/len(qc_df_accepted)
        freq_stat, freq_p_value = stats.chisquare(freq_observed, [1/2, 1/2])
        mismatch_observed = qc_df_accepted['mismatch'].value_counts().loc[['0', '1', '2']]/len(qc_df_accepted)
        mismatch_stat, mismatch_p_value = stats.chisquare(mismatch_observed, [1/3, 1/3, 1/3])
        pseudo_observed = qc_df_accepted['pseudo'].value_counts().loc[[1, 0]]/len(qc_df_accepted)
        pseudo_stat, pseudo_p_value = stats.chisquare(pseudo_observed, [1/8, 7/8])
        print('No change in the proportion between conditions (p>0.05):')
        print(f'Frequency: stat={freq_stat} and p={freq_p_value}')
        print(f'Mismatch: stat={mismatch_stat} and p={mismatch_p_value}')
        print(f'Pseudo: stat={pseudo_stat} and p={pseudo_p_value}')
        qc_df_contingency = pd.crosstab(qc_df_accepted['group'], [qc_df_accepted['freq'],
                                                                  qc_df_accepted['mismatch'],
                                                                  qc_df_accepted['pseudo']])
    if args.stimulus == 'sentences':
        pred_observed = qc_df_accepted['predictable'].value_counts().loc[[0, 1]]/len(qc_df_accepted)
        pred_stat, pred_p_value = stats.chisquare(pred_observed, [1/2, 1/2])
        print('No change in the proportion between conditions (p>0.05):')
        print(f'Predictability: stat={pred_stat} and p={pred_p_value}')
        qc_df_contingency = pd.crosstab(qc_df_accepted['group'], qc_df_accepted['predictable'])
    chi2, contingency_p_value, dof, exp_freq = stats.chi2_contingency(qc_df_contingency)
    print('\nNo variation in proportion of conditions between groups:')
    print(f'chi2={chi2}, p={contingency_p_value} and dof={dof}')
    ## Control quality in all stim*participant and save resulting tables
    qc_overall = pd.DataFrame({'Overall Metric': [f'Rejection rate (Loss Rate > {args.loss_rate_criteria}% of considered stim)'],
                               'Value': [qc_df['rejected'].sum()*100/(nb_subjects*len(video_files))]})
    qc_participants = qc_df.groupby('subject').agg(count_rejected=('rejected', 'sum')).reset_index()
    qc_participants['rejection_rate'] = qc_participants['count_rejected']/len(video_files)*100
    qc_stim = qc_df.groupby('stimulus').agg(count_rejected=('rejected', 'sum')).reset_index()
    qc_stim['rejection_rate'] = qc_stim['count_rejected']/nb_subjects*100
    qc_overall.to_csv(os.path.join(args.path2deriv,f'eye-tracking/quality_check/qc_overall_stim-{args.stimulus}_sub-{args.group}_loss-{args.loss_rate_criteria}.csv'))
    qc_participants.to_csv(os.path.join(args.path2deriv,f'eye-tracking/quality_check/qc_participants_stim-{args.stimulus}_sub-{args.group}_loss-{args.loss_rate_criteria}.csv'))
    qc_stim.to_csv(os.path.join(args.path2deriv,f'eye-tracking/quality_check/qc_stimuli_stim-{args.stimulus}_sub-{args.group}_loss-{args.loss_rate_criteria}.csv'))
    ## Plot results
    from utils.viz import plot_quality_check
    plot_quality_check(args, qc_overall, qc_participants, qc_stim)


#%% Check if gaze is in AOI

def create_aoi_hit_tables(args, subject, epochs, et_annotations):
    sub_name = args.group if isinstance(subject, list) and args.group.split('_', 1)[0] in ('all', '1', '2', '3') else subject
    print(f'\n## Determining aoi hits for subject(s) {sub_name}')
    if args.stimulus == 'words' or args.stimulus == 'sentences':
        video_files = glob(f'utils/stim_videos/videos/{args.stimulus}/*.mp4')
    else:
        video_files = glob(f'utils/stim_videos/videos/*/{args.stimulus}.mp4')
    if subject.startswith('3'):
       video_files = keep_common_videos(args, subject, video_files)
    df_subject = []
    for video_file in tqdm(video_files):
        stim_name = os.path.splitext(os.path.basename(video_file))[0]
        print(f'\nDetermining aoi hits for stim {stim_name}')
        ## Create query and fetch corresponding epochs
        stim_epoch_df, _, stim_annot = select_stim_epoch(args, subject, epochs, et_annotations, stim_name)
        if stim_epoch_df is None:
            print(f'Rejecting gaze from subject {subject} for stim {stim_name}')  
            continue # Entirely pass the video if dealing with a subject with a rejected epoch
        # AOI points coordinates
        df_aoi_file = glob(os.path.join(args.path2videocoords, f'*/{stim_name}_aoi_coordinates.csv'))[0]
        df_aoi = pd.read_csv(os.path.join(df_aoi_file))
        x_hand_cols = df_aoi.filter(regex='x_r_hand*').columns
        y_hand_cols = df_aoi.filter(regex='y_r_hand*').columns
        x_face_cols = df_aoi.filter(regex='x_face*').columns
        y_face_cols = df_aoi.filter(regex='y_face*').columns
        x_eyes_cols, y_eyes_cols = x_face_cols[:4], y_face_cols[:4]
        # x_nose_cols, y_nose_cols = x_face_cols[4:8], y_face_cols[4:8]
        x_lips_cols, y_lips_cols = x_face_cols[8:12], y_face_cols[8:12]
        x_face_cols, y_face_cols = x_face_cols[12:], y_face_cols[12:]
        x_middle_cols, y_middle_cols = x_eyes_cols[2:].append(x_lips_cols[:2]), y_eyes_cols[2:].append(y_lips_cols[:2])
        # Replicate AOI lines to feat the number of gaze data points
        df_aoi = df_aoi.loc[df_aoi.index.repeat(33.3333333333333333333)].reset_index(drop=True)
        ## Prepare results df
        if args.stimulus == 'words':
            freq = stim_name.split('_')[1][0]
            mismatch = stim_name.split('_')[1][1]
            pseudo = 1 if 'pseudo' in stim_name else 0
            cols = ['subject','stim','frame','freq','mismatch','pseudo','hand','eyes','lips','middle','face','left']
        if args.stimulus == 'sentences':
            predictable = 0 if (int(stim_name.split('_')[1]) % 2) == 0 else 1
            cols = ['subject','stim','frame','predictable','hand','eyes','lips','middle','face','left']
        df_gaze_aoi = pd.DataFrame(index=np.arange(len(df_aoi)), columns=cols)
        df_gaze_aoi['subject'], df_gaze_aoi['stim'] = subject, stim_name
        if args.stimulus == 'words':
            df_gaze_aoi['freq'], df_gaze_aoi['mismatch'], df_gaze_aoi['pseudo'] = freq, mismatch, pseudo
        if args.stimulus == 'sentences':
            df_gaze_aoi['predictable'] = predictable
        df_gaze_aoi['hand'], df_gaze_aoi['eyes'], df_gaze_aoi['lips'], df_gaze_aoi['middle'], df_gaze_aoi['face'], df_gaze_aoi['left'] = 0, 0, 0, 0, 0, 0
        ## Determine in gaze is in each AOI
        with pd.option_context('mode.chained_assignment', None): # To desactivate SettingWithCopyWarning
            for df_aoi_index in df_aoi.index:
                annot_timepoint = annot_of_timepoint(stim_epoch_df['time'].iloc[df_aoi_index], stim_annot)
                df_gaze_aoi['frame'].iloc[df_aoi_index] = df_aoi['frame_number'].iloc[df_aoi_index]
                # Gaze
                if (pd.isna(stim_epoch_df['et_x'].iloc[df_aoi_index]) or pd.isna(stim_epoch_df['et_y'].iloc[df_aoi_index])):
                    df_gaze_aoi['hand'].iloc[df_aoi_index], df_gaze_aoi['eyes'].iloc[df_aoi_index] = 0,0
                    df_gaze_aoi['lips'].iloc[df_aoi_index], df_gaze_aoi['face'].iloc[df_aoi_index] = 0,0
                    continue # If a gaze coordinate is missing, no AOI is hit
                if annot_timepoint != 'fixation':
                    continue # Only take fixations into account
                gaze_point = Point(int(stim_epoch_df['et_x'].iloc[df_aoi_index]), int(stim_epoch_df['et_y'].iloc[df_aoi_index]))
                # Hand
                x_min, y_min = (min(df_aoi[x_hand_cols].iloc[df_aoi_index])-40, min(df_aoi[y_hand_cols].iloc[df_aoi_index])-40)
                x_max, y_max = (max(df_aoi[x_hand_cols].iloc[df_aoi_index])+40, max(df_aoi[y_hand_cols].iloc[df_aoi_index])+40)
                hand_pts = [(x_min, y_max), (x_max, y_max), (x_max, y_min), (x_min, y_min)]
                hand_pts = np.array(hand_pts, dtype=np.int32)
                hand_polygon = Polygon(hand_pts)
                if hand_polygon.contains(gaze_point) or hand_polygon.touches(gaze_point):
                    df_gaze_aoi['hand'].iloc[df_aoi_index] = 1
                # Eyes
                eyes_pts = []
                for x_eyes_pt_interest, y_eyes_pt_interest in zip(x_eyes_cols, y_eyes_cols):
                    eyes_pts.append([df_aoi[x_eyes_pt_interest].iloc[df_aoi_index], df_aoi[y_eyes_pt_interest].iloc[df_aoi_index]])
                eyes_pts = np.array(eyes_pts, dtype=np.int32)
                eyes_polygon = Polygon(eyes_pts)
                if eyes_polygon.contains(gaze_point) or eyes_polygon.touches(gaze_point):
                    if stim_annot['onsets'][0] <= stim_epoch_df['time'].iloc[df_aoi_index] <= stim_annot['onsets'][0] + stim_annot['durations'][0]:
                        continue # Do not mark this fixation on eyes as a hit if it's the first event of the stim (fixation cross bias)
                    else:
                        df_gaze_aoi['eyes'].iloc[df_aoi_index] = 1
                # Lips
                lips_pts = []
                for x_lips_pt_interest, y_lips_pt_interest in zip(x_lips_cols, y_lips_cols):
                    lips_pts.append([df_aoi[x_lips_pt_interest].iloc[df_aoi_index], df_aoi[y_lips_pt_interest].iloc[df_aoi_index]])
                lips_pts = np.array(lips_pts, dtype=np.int32)
                lips_polygon = Polygon(lips_pts)
                if lips_polygon.contains(gaze_point) or lips_polygon.touches(gaze_point):
                    df_gaze_aoi['lips'].iloc[df_aoi_index] = 1
                # Middle of the face (between eyes and lips)
                middle_pts = []
                for x_middle_pt_interest, y_middle_pt_interest in zip(x_middle_cols, y_middle_cols):
                    middle_pts.append([df_aoi[x_middle_pt_interest].iloc[df_aoi_index], df_aoi[y_middle_pt_interest].iloc[df_aoi_index]])
                middle_pts = np.array(middle_pts, dtype=np.int32)
                middle_polygon = Polygon(middle_pts)
                if middle_polygon.contains(gaze_point) or middle_polygon.touches(gaze_point):
                    df_gaze_aoi['middle'].iloc[df_aoi_index] = 1
                # face
                face_pts = []
                for x_face_pt_interest, y_face_pt_interest in zip(x_face_cols, y_face_cols):
                    face_pts.append([df_aoi[x_face_pt_interest].iloc[df_aoi_index], df_aoi[y_face_pt_interest].iloc[df_aoi_index]])
                centroid = np.mean(face_pts, axis=0)
                face_pts = (face_pts - centroid) * 1.3 + centroid # Move each point away from the centroid to enlarge the AOI
                face_pts[:,1] += -55 # Move up all the points
                face_pts[:15,1] += -55 # Move up the upper points to include the foreface and hair
                face_pts = np.array(face_pts, dtype=np.int32)
                face_polygon = Polygon(face_pts)
                if face_polygon.contains(gaze_point) or face_polygon.touches(gaze_point):
                    df_gaze_aoi['face'].iloc[df_aoi_index] = 1
                # Laterality
                lat_pt1 = (df_aoi['x_face152'].iloc[df_aoi_index], df_aoi['y_face152'].iloc[df_aoi_index])
                lat_pt2 = (df_aoi['x_face10'].iloc[df_aoi_index], df_aoi['y_face10'].iloc[df_aoi_index])
                lat_pt1, lat_pt2 = (lat_pt1 - centroid) * 1.3 + centroid, (lat_pt2 - centroid) * 1.3 + centroid # Apply the same translations as for creating the face AOI
                lat_pt1[1], lat_pt2[1] = lat_pt1[1]-55, lat_pt2[1]-55-55
                lat_vect, gaze_vect = np.array(lat_pt2)-np.array(lat_pt1), np.array((int(gaze_point.x),int(gaze_point.y)))-np.array(lat_pt1)
                cross_product = np.cross(lat_vect, gaze_vect)
                if cross_product < 0:
                    df_gaze_aoi['left'].iloc[df_aoi_index] = 1        
        df_subject.concat(df_gaze_aoi)
    df_subject = pd.concat(df_subject, ignore_index=True)
    df_subject.to_csv(os.path.join(args.path2deriv,f'eye-tracking/aoi_hit_tables/{args.stimulus}_sub-{sub_name}_aoi.csv'))


def compute_nb_events(args, subjects, epochs, et_annotations):
    print('Retrieving nb of saccades and fixations for each stim x participant...')
    if args.stimulus == 'words':
        conditions_list = ['freq', 'mismatch', 'pseudo']
    elif args.stimulus == 'sentences':
        conditions_list = ['predictable']
    else:
        conditions_list = ['rounding', 'place_articulation', 'handshape', 'position',
                           'consonant', 'vowel']
    cols = ['subject', 'group', 'stimulus'] + conditions_list
    nb_events_df = pd.DataFrame(columns=cols)
    if any([sub.startswith('3') for sub in subjects]):
        print('Control participant(s) included: keeping only stims common to all')
        if args.stimulus == 'words':
            common_stim = ['word_h0_03_pseudo', 'word_h0_04_pseudo', 'word_h0_06', 'word_h0_09', 'word_h0_11', 'word_h0_14', 'word_h0_17', 'word_h0_18', 'word_h0_19', 'word_h0_23', 'word_h0_29', 'word_h0_31',
                           'word_h1_01_pseudo', 'word_h1_03', 'word_h1_04_pseudo', 'word_h1_09', 'word_h1_15', 'word_h1_21', 'word_h1_23', 'word_h1_25', 'word_h1_27', 'word_h1_28', 'word_h1_32', 'word_h1_37',
                           'word_h2_02', 'word_h2_02_pseudo', 'word_h2_04', 'word_h2_04_pseudo', 'word_h2_07', 'word_h2_09', 'word_h2_10', 'word_h2_11', 'word_h2_12', 'word_h2_18', 'word_h2_33', 'word_h2_36',
                           'word_l0_03_pseudo', 'word_l0_05_pseudo', 'word_l0_09', 'word_l0_12', 'word_l0_17', 'word_l0_18', 'word_l0_24', 'word_l0_29', 'word_l0_33', 'word_l0_36', 'word_l0_38', 'word_l0_40',
                           'word_l1_01', 'word_l1_02', 'word_l1_03', 'word_l1_04', 'word_l1_04_pseudo', 'word_l1_05', 'word_l1_05_pseudo', 'word_l1_06', 'word_l1_07', 'word_l1_08', 'word_l1_09', 'word_l1_10',
                           'word_l2_01', 'word_l2_01_pseudo', 'word_l2_02_pseudo', 'word_l2_07', 'word_l2_11', 'word_l2_12', 'word_l2_16', 'word_l2_27',  'word_l2_29', 'word_l2_35', 'word_l2_36', 'word_l2_38']
        elif args.stimulus == 'sentences':
            common_stim = [f'sent_{num}' for num in range(71,101)]
    for epochs_sub, et_annotations_sub in tqdm(zip(epochs, et_annotations), total=len(subjects)):
        for i_annot, et_annot in enumerate(et_annotations_sub):
            if epochs_sub.metadata['event_type'].iloc[i_annot] in ['word_onset','sentence_onset','syllable_onset']:
                subject = str(int(epochs_sub.metadata['subject'].iloc[i_annot]))
                if args.stimulus == 'sentences':
                    stim = f"sent_{int(epochs_sub.metadata['num_sentence'].iloc[i_annot])}"
                elif args.stimulus == 'words':
                    f"word_{epochs_sub.metadata['freq'].iloc[i_annot]}{int(epochs_sub.metadata['mismatch'].iloc[i_annot])}_{int(epochs_sub.metadata['num_in_cat'].iloc[i_annot]):02d}{'_pseudo' if epochs_sub.metadata['pseudo'].iloc[i_annot]==1 else ''}"
                else:
                    stim = epochs_sub.metadata['syllable'].iloc[i_annot]
                if 'Words' in args.task or 'Sentences' in args.task:
                    if any([sub.startswith('3') for sub in subjects]) and stim not in common_stim:
                        continue
                # If bad trial, reject it
                stim_epoch_df = epochs_sub[i_annot].to_data_frame()
                stim_epoch_df, _ = compute_data_loss_reject(args, stim_epoch_df)
                if stim_epoch_df is None:
                    continue
                if args.stimulus == 'words':
                    stim_characteristics_dict = {'freq': epochs_sub.metadata['freq'].iloc[i_annot],
                                                 'mismatch': str(int(epochs_sub.metadata['mismatch'].iloc[i_annot])),
                                                 'pseudo': int(epochs_sub.metadata['pseudo'].iloc[i_annot])}
                elif args.stimulus == 'sentences':
                    stim_characteristics_dict = {'predictable': int(epochs_sub.metadata['predictable'].iloc[i_annot])}
                    stim_characteristics_dict.update({'subject': subject,
                                                      'group': subject[0],
                                                      'stimulus': stim})
                else:
                    stim_characteristics_dict = {'rounding': epochs_sub.metadata['rounding'].iloc[i_annot],
                                                 'place_articulation': epochs_sub.metadata['place_articulation'].iloc[i_annot],
                                                 'handshape': epochs_sub.metadata['handshape'].iloc[i_annot],
                                                 'position': epochs_sub.metadata['position'].iloc[i_annot],
                                                 'consonant': epochs_sub.metadata['consonant'].iloc[i_annot],
                                                 'vowel': epochs_sub.metadata['vowel'].iloc[i_annot],}
                    stim_characteristics_dict.update({'subject': subject,
                                                      'group': subject[0],
                                                      'stimulus': stim})
                # nb_events_df = nb_events_df.concat({**stim_characteristics_dict, **{'event_type': 'fixations', 'nb_events': et_annot['annotations'].count('fixation')}},
                #                                    ignore_index=True)
                new_row = pd.DataFrame([{**stim_characteristics_dict, **{'event_type': 'saccades', 'nb_events': et_annot['annotations'].count('saccade')}}])
                nb_events_df = pd.concat([nb_events_df, new_row], ignore_index=True)
    ## Statistics
    file = open(os.path.join(args.path2deriv,f'nb_events_{args.stimulus}.txt'), 'w')
    sys.stdout = file
    for cond in conditions_list:
        contingency = pd.crosstab(nb_events_df['group'], nb_events_df[cond])
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        print(f"Saccade ratio between the groups in cond {cond}: {chi2:.3f}, p-value = {p:.3f}")
    for cond in conditions_list:
        contingency = pd.crosstab(nb_events_df[cond], nb_events_df['group'])
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        print(f"Saccade ratio between feature versions in cond {cond}: {chi2:.3f}, p-value = {p:.3f}")
    sys.stdout = sys.__stdout__
    file.close()
    ## Plotting
    from utils.viz import plot_nb_events
    plot_nb_events(args, nb_events_df)
     
    
def select_stims_for_cond(args, df, cond):
    if cond == 'all':
        df_cond = df
    if args.stimulus == 'words':
        if cond == 'pseudo_0_freq_h':
            df_cond = df[(df['pseudo']=='0') & (df['freq']=='h')]
        elif cond == 'pseudo_0_freq_l':
            df_cond = df[(df['pseudo']=='0') & (df['freq']=='l')]
        elif cond == 'pseudo_0_mismatch_0':
            df_cond = df[(df['pseudo']=='0') & (df['mismatch']=='0')]   
        elif cond == 'pseudo_0_mismatch_1':
            df_cond = df[(df['pseudo']=='0') & (df['mismatch']=='1')]
        elif cond == 'pseudo_0_mismatch_2':
            df_cond = df[(df['pseudo']=='0') & (df['mismatch']=='2')]
        elif cond == 'pseudo_0_mismatch_0_freq_h':
            df_cond = df[(df['pseudo']=='0') & (df['mismatch']=='0') & (df['freq']=='h')]
        elif cond == 'pseudo_0_mismatch_1_freq_h':
            df_cond = df[(df['pseudo']=='0') & (df['mismatch']=='1') & (df['freq']=='h')]
        elif cond == 'pseudo_0_mismatch_2_freq_h':
            df_cond = df[(df['pseudo']=='0') & (df['mismatch']=='2') & (df['freq']=='h')]
        elif cond == 'pseudo_0_mismatch_0_freq_l':
            df_cond = df[(df['pseudo']=='0') & (df['mismatch']=='0') & (df['freq']=='l')]
        elif cond == 'pseudo_0_mismatch_1_freq_l':
            df_cond = df[(df['pseudo']=='0') & (df['mismatch']=='1') & (df['freq']=='l')]
        elif cond == 'pseudo_0_mismatch_2_freq_l':
            df_cond = df[(df['pseudo']=='0') & (df['mismatch']=='2') & (df['freq']=='l')]
        elif cond == 'pseudo_0':
            df_cond = df[df['pseudo']=='0']
        elif cond == 'pseudo_1':
            df_cond = df[df['pseudo']=='1']
    if args.stimulus == 'sentences':
        if cond == 'predictable_0':
            df_cond = df[df['predictable']=='0'] 
        elif cond == 'predictable_1':
            df_cond = df[df['predictable']=='1']
    if args.stimulus == 'syllables':
        if cond == 'rounded':
            df_cond = df[df['rounding']=='rounded']
        elif cond == 'unrounded':
            df_cond = df[df['rounding']=='rounded']
        elif cond == 'bilabial':
            df_cond = df[df['place_articulation']=='bilabial']
        elif cond == 'coronal':
            df_cond = df[df['place_articulation']=='coronal']
        elif cond == 'one_finger':
            df_cond = df[df['handshape']=='one_finger']
        elif cond == 'five_fingers':
            df_cond = df[df['handshape']=='five_fingers']
        elif cond == 'ear':
            df_cond = df[df['position']=='ear']
        elif cond == 'chin':
            df_cond = df[df['position']=='chin']
        elif cond == 'm':
            df_cond = df[df['consonant']=='m']
        elif cond == 'p':
            df_cond = df[df['consonant']=='p']
        elif cond == 'd':
            df_cond = df[df['consonant']=='d']
        elif cond == 't':
            df_cond = df[df['consonant']=='t']
        elif cond == 'a':
            df_cond = df[df['vowel']=='a']
        elif cond == 'e':
            df_cond = df[df['vowel']=='e']
        elif cond == 'o':
            df_cond = df[df['vowel']=='o']
        elif cond == 'u':
            df_cond = df[df['vowel']=='u']
    return df_cond
    
    
def compute_proportion_hits(sub_aoi_hits_df, proportion_hits_list):
    all_aoi_columns = [col for col in list(sub_aoi_hits_df.columns) if col not in ['subject', 'stim', 'frame',
                                                                                   'predictable', 'freq', 'mismatch', 'pseudo']]
    if sub_aoi_hits_df['stim'].iloc[0][:4] == 'sent':
        nb_col = 68
        df_last_word_time = pd.read_csv('utils/sent_last_word_time.csv') # Time from the begining of the video, in ms
    else:
        nb_col = 37
    sub_results_df = pd.DataFrame(columns=[str(num) for num in range(0,nb_col)]) # For now name of cols are the number of futur elements in stim_dict
    for stim in sub_aoi_hits_df['stim'].unique():
        sub_aoi_hits_df_stim_all = sub_aoi_hits_df[sub_aoi_hits_df['stim']==stim].reset_index()
        sub_aoi_hits_df_stim_list = [sub_aoi_hits_df_stim_all]
        if stim[:4]=='sent':
            last_word_time = int(df_last_word_time[df_last_word_time['num_sentence']==int(stim[5:])]['last_word_time'])
            sub_aoi_hits_df_stim_last_word = sub_aoi_hits_df_stim_all[last_word_time:].reset_index()
            sub_aoi_hits_df_stim_list.append(sub_aoi_hits_df_stim_last_word)
        stim_proportion_dict = {}
        for i_df, sub_aoi_hits_df_stim in enumerate(sub_aoi_hits_df_stim_list):
            upper_sum = len(sub_aoi_hits_df_stim[(sub_aoi_hits_df_stim['eyes'] == 1) | (sub_aoi_hits_df_stim['middle'] == 1)])
            upper_lips_sum = upper_sum + sub_aoi_hits_df_stim['lips'].sum()
            nb_timepoints = len(sub_aoi_hits_df_stim)
            # Main sanity checks
            stim_proportion_hits_all = sub_aoi_hits_df_stim[all_aoi_columns].apply(lambda row: 1 in row.values, axis=1).sum()
            stim_proportion_hits_all_hand_and_face = len(sub_aoi_hits_df_stim[(sub_aoi_hits_df_stim['hand'] == 1) & (sub_aoi_hits_df_stim['face'] == 1)])/stim_proportion_hits_all*100 if stim_proportion_hits_all != 0 else 0
            stim_proportion_hits_all_hand_and_lips = len(sub_aoi_hits_df_stim[(sub_aoi_hits_df_stim['hand'] == 1) & (sub_aoi_hits_df_stim['lips'] == 1)])/stim_proportion_hits_all*100 if stim_proportion_hits_all != 0 else 0
            # Proportion in AOIs
            stim_proportion_hits_face = sub_aoi_hits_df_stim['face'].sum()/nb_timepoints*100
            stim_proportion_hits_all_face = sub_aoi_hits_df_stim['face'].sum()/stim_proportion_hits_all*100 if stim_proportion_hits_all != 0 else np.nan
            stim_proportion_hits_eyes = sub_aoi_hits_df_stim['eyes'].sum()/nb_timepoints*100
            stim_proportion_hits_face_eyes = sub_aoi_hits_df_stim['eyes'].sum()/sub_aoi_hits_df_stim['face'].sum()*100 if stim_proportion_hits_face != 0 else np.nan
            stim_proportion_hits_lips = sub_aoi_hits_df_stim['lips'].sum()/nb_timepoints*100
            stim_proportion_hits_face_lips = sub_aoi_hits_df_stim['lips'].sum()/sub_aoi_hits_df_stim['face'].sum()*100 if stim_proportion_hits_face != 0 else np.nan
            stim_proportion_hits_face_lips_no_other = sub_aoi_hits_df_stim['lips'].sum()/upper_lips_sum*100 if upper_lips_sum != 0 else np.nan
            stim_proportion_hits_middle = sub_aoi_hits_df_stim['middle'].sum()/nb_timepoints*100
            stim_proportion_hits_face_middle = sub_aoi_hits_df_stim['middle'].sum()/sub_aoi_hits_df_stim['face'].sum()*100 if stim_proportion_hits_face != 0 else np.nan
            stim_proportion_hits_face_upper_face = upper_sum/sub_aoi_hits_df_stim['face'].sum()*100 if stim_proportion_hits_face != 0 else np.nan
            stim_proportion_hits_face_upper_face_no_other = upper_sum/upper_lips_sum*100 if upper_lips_sum != 0 else np.nan
            stim_proportion_hits_hand = sub_aoi_hits_df_stim['hand'].sum()/nb_timepoints*100
            stim_proportion_hits_all_hand = sub_aoi_hits_df_stim['hand'].sum()/stim_proportion_hits_all*100 if stim_proportion_hits_all != 0 else np.nan
            stim_proportion_hits_all_other = 100 - stim_proportion_hits_all_face - stim_proportion_hits_all_hand + stim_proportion_hits_all_hand_and_face
            stim_proportion_hits_face_other = 100 - stim_proportion_hits_face_eyes - stim_proportion_hits_face_lips - stim_proportion_hits_face_middle
            # Laterality proportions
            stim_proportion_lat_face_left = sub_aoi_hits_df_stim[sub_aoi_hits_df_stim['face'] == 1]['left'].sum()/sub_aoi_hits_df_stim['face'].sum()*100 if stim_proportion_hits_face.sum() != 0 else np.nan
            stim_proportion_lat_face_right = 100 - stim_proportion_lat_face_left
            stim_proportion_lat_eyes_left = sub_aoi_hits_df_stim[sub_aoi_hits_df_stim['eyes'] == 1]['left'].sum()/sub_aoi_hits_df_stim['eyes'].sum()*100 if stim_proportion_hits_eyes.sum() != 0 else np.nan
            stim_proportion_lat_eyes_right = 100 - stim_proportion_lat_eyes_left
            stim_proportion_lat_lips_left = sub_aoi_hits_df_stim[sub_aoi_hits_df_stim['lips'] == 1]['left'].sum()/sub_aoi_hits_df_stim['lips'].sum()*100 if stim_proportion_hits_lips.sum() != 0 else np.nan
            stim_proportion_lat_lips_right = 100 - stim_proportion_lat_lips_left
            stim_proportion_lat_middle_left = sub_aoi_hits_df_stim[sub_aoi_hits_df_stim['middle'] == 1]['left'].sum()/sub_aoi_hits_df_stim['middle'].sum()*100 if stim_proportion_hits_middle.sum() != 0 else np.nan
            stim_proportion_lat_middle_right = 100 - stim_proportion_lat_middle_left
            stim_proportion_lat_upper_face_left = sub_aoi_hits_df_stim[(sub_aoi_hits_df_stim['eyes'] == 1) | (sub_aoi_hits_df_stim['middle'] == 1)]['left'].sum()/(sub_aoi_hits_df_stim['eyes'].sum()+sub_aoi_hits_df_stim['middle'].sum())*100 if (stim_proportion_hits_eyes.sum()+stim_proportion_hits_middle.sum()) != 0 else np.nan
            stim_proportion_lat_upper_face_right = 100 - stim_proportion_lat_upper_face_left
            stim_characteristics_dict = {'freq': str(sub_aoi_hits_df_stim.iloc[0]['freq']),
                                         'mismatch': str(sub_aoi_hits_df_stim.iloc[0]['mismatch']),
                                         'pseudo': str(sub_aoi_hits_df_stim.iloc[0]['pseudo'])} if stim.startswith('word') else {'predictable': str(sub_aoi_hits_df_stim.iloc[0]['predictable'])}
            sfx = '_last_word' if i_df == 1 else ''
            stim_proportion_dict.update({f'proportion_hits_all{sfx}': [stim_proportion_hits_all.mean()*100],
                                       f'proportion_hits_face{sfx}': [stim_proportion_hits_face],
                                       f'proportion_hits_all_face{sfx}': [stim_proportion_hits_all_face],
                                       f'proportion_hits_eyes{sfx}': [stim_proportion_hits_eyes],
                                       f'proportion_hits_face_eyes{sfx}': [stim_proportion_hits_face_eyes],
                                       f'proportion_hits_lips{sfx}': [stim_proportion_hits_lips],
                                       f'proportion_hits_face_lips{sfx}': [stim_proportion_hits_face_lips],
                                       f'proportion_hits_face_lips_no_other{sfx}': [stim_proportion_hits_face_lips_no_other],
                                       f'proportion_hits_middle{sfx}': [stim_proportion_hits_middle],
                                       f'proportion_hits_face_middle{sfx}': [stim_proportion_hits_face_middle],
                                       f'proportion_hits_face_upper_face{sfx}': [stim_proportion_hits_face_upper_face],
                                       f'proportion_hits_face_upper_face_no_other{sfx}': [stim_proportion_hits_face_upper_face_no_other],
                                       f'proportion_hits_hand{sfx}': [stim_proportion_hits_hand],
                                       f'proportion_hits_all_hand{sfx}': [stim_proportion_hits_all_hand],
                                       f'proportion_hits_all_other{sfx}': [stim_proportion_hits_all_other],
                                       f'proportion_hits_face_other{sfx}': [stim_proportion_hits_face_other],
                                       f'proportion_hits_all_hand_and_face{sfx}': [stim_proportion_hits_all_hand_and_face],
                                       f'proportion_hits_all_hand_and_lips{sfx}': [stim_proportion_hits_all_hand_and_lips],
                                       f'proportion_lat_face_left{sfx}': [stim_proportion_lat_face_left],
                                       f'proportion_lat_face_right{sfx}': [stim_proportion_lat_face_right],
                                       f'proportion_lat_eyes_left{sfx}': [stim_proportion_lat_eyes_left],
                                       f'proportion_lat_eyes_right{sfx}':[stim_proportion_lat_eyes_right],
                                       f'proportion_lat_lips_left{sfx}': [stim_proportion_lat_lips_left],
                                       f'proportion_lat_lips_right{sfx}': [stim_proportion_lat_lips_right],
                                       f'proportion_lat_middle_left{sfx}': [stim_proportion_lat_middle_left],
                                       f'proportion_lat_middle_right{sfx}': [stim_proportion_lat_middle_right],
                                       f'proportion_lat_upper_face_left{sfx}': [stim_proportion_lat_upper_face_left],
                                       f'proportion_lat_upper_face_right{sfx}': [stim_proportion_lat_upper_face_right],
                                       f'total_all{sfx}': [stim_proportion_hits_all_face + stim_proportion_hits_all_hand + stim_proportion_hits_all_other - stim_proportion_hits_all_hand_and_face],
                                       f'total_face{sfx}': [stim_proportion_hits_face_eyes + stim_proportion_hits_face_lips + stim_proportion_hits_face_middle + stim_proportion_hits_face_other],
                                       f'total_face_no_other{sfx}': [stim_proportion_hits_face_lips_no_other + stim_proportion_hits_face_upper_face_no_other]})
        stim_characteristics_dict = {'freq': str(sub_aoi_hits_df_stim.iloc[0]['freq']),
                                     'mismatch': str(sub_aoi_hits_df_stim.iloc[0]['mismatch']),
                                     'pseudo': str(sub_aoi_hits_df_stim.iloc[0]['pseudo'])} if stim.startswith('word') else {'predictable': str(sub_aoi_hits_df_stim.iloc[0]['predictable'])}
        stim_dict = {**{'subject': str(sub_aoi_hits_df_stim.iloc[0]['subject']),
                        'group': str(sub_aoi_hits_df_stim.iloc[0]['subject'])[0],
                        'stim': stim},
                     **stim_characteristics_dict,
                     **stim_proportion_dict}
        if len(sub_results_df) == 0:
            if stim.startswith('sent'):
                sub_results_df = sub_results_df.iloc[:, :-2] # Only one condition in sentences
            sub_results_df.columns = stim_dict.keys()
        sub_results_df = pd.concat([sub_results_df, pd.DataFrame(stim_dict)], ignore_index=True)
        sub_results_df = sub_results_df.astype(float, errors='ignore')
    proportion_hits_list.append(sub_results_df)
        
        
def proportion_aoi_hits(args, subjects):
    ## Collect all relevant individual aoi hit tables in a big list
    print(f'## Retrieving aoi hit files for stim {args.stimulus} from subjects {subjects}')
    aoi_hit_files = []
    for i_subject, sub in tqdm(enumerate(subjects), total=len(subjects)):
        aoi_hit_files.append(glob(os.path.join(args.path2deriv,f'eye-tracking/aoi_hit_tables/{args.stimulus}_sub-{sub}_aoi.csv'))[0])
    print(f'\n{len(aoi_hit_files)} aoi hit files found')
    ## Create a dictionary containing participants' aoi hit table lists for each condition
    all_subjects_df_list = [pd.read_csv(df_subject).iloc[: , 1:] for df_subject in aoi_hit_files]
    if any([sub.startswith('3') for sub in subjects]):
        all_subjects_df_list = keep_common_stims(args, all_subjects_df_list)
    ## Compute proportion of samples within each aoi for each participant
    proportion_hits_list = mp.Manager().list()
    with mp.Pool(mp.cpu_count()) as pool:
        pool.starmap(compute_proportion_hits, [(sub_df, proportion_hits_list) for sub_df in all_subjects_df_list])
    pool.close()
    pool.join()
    proportion_hits_list = list(proportion_hits_list)
    proportion_hits_df = pd.concat(proportion_hits_list)
    all_aoi_columns = [col for col in list(proportion_hits_df.columns) if col not in ['subject', 'group', 'stim', 'frame',
                                                                                      'predictable', 'freq', 'mismatch', 'pseudo']]
    proportion_hits_df[all_aoi_columns] = proportion_hits_df[all_aoi_columns].astype('float')
    if args.stimulus == 'words':
        proportion_hits_df['mismatch'] = proportion_hits_df['mismatch'].astype(str) # Convert mismatch values to categorical str type
    for col in proportion_hits_df.select_dtypes(include=['number']).columns:
        proportion_hits_df[col] = proportion_hits_df[col].map(lambda x: round(x,2))
    ## Exclude trials with no AOI hit at all (= in practice, those that had been excluded because of high data loss)
    proportion_hits_df = proportion_hits_df[proportion_hits_df['proportion_hits_all'] != 0]
    ## Statistics
    # proportion_aoi_hits_detailed_analysis(args, subjects, proportion_hits_df)
    proportion_correlation(args, proportion_hits_df)
    ## Plotting
    if args.plot_proportion_aoi_hits:
        from utils.viz import plot_proportion_aoi_hits_boxplots, plot_proportion_aoi_hits_bars, plot_proportion_correlation
        plot_proportion_aoi_hits_boxplots(args, proportion_hits_df)
        # plot_proportion_aoi_hits_bars(args, proportion_hits_df)
        # plot_proportion_correlation(args, proportion_hits_df)


def compute_duration_hits(sub_aoi_hits_df, duration_hits_list):
    ## Compute laterality columns
    sub_aoi_hits_df['right'] = sub_aoi_hits_df['left'].apply(lambda x: 1 if x == 0 else 0) # Add 'right' column, relevant for this analysis
    sub_aoi_hits_df['face_left'] = sub_aoi_hits_df.apply(lambda row: 1 if row['face'] == 1 and row['left'] == 1 else 0, axis=1)
    sub_aoi_hits_df['face_right'] = sub_aoi_hits_df.apply(lambda row: 1 if row['face'] == 1 and row['right'] == 1 else 0, axis=1)
    sub_aoi_hits_df['lips_left'] = sub_aoi_hits_df.apply(lambda row: 1 if row['lips'] == 1 and row['left'] == 1 else 0, axis=1)
    sub_aoi_hits_df['lips_right'] = sub_aoi_hits_df.apply(lambda row: 1 if row['lips'] == 1 and row['right'] == 1 else 0, axis=1)
    sub_aoi_hits_df['eyes_left'] = sub_aoi_hits_df.apply(lambda row: 1 if row['eyes'] == 1 and row['left'] == 1 else 0, axis=1)
    sub_aoi_hits_df['eyes_right'] = sub_aoi_hits_df.apply(lambda row: 1 if row['eyes'] == 1 and row['right'] == 1 else 0, axis=1)
    sub_aoi_hits_df['middle_left'] = sub_aoi_hits_df.apply(lambda row: 1 if row['middle'] == 1 and row['left'] == 1 else 0, axis=1)
    sub_aoi_hits_df['middle_right'] = sub_aoi_hits_df.apply(lambda row: 1 if row['middle'] == 1 and row['right'] == 1 else 0, axis=1)
    sub_aoi_hits_df = sub_aoi_hits_df.drop(['left','right'], axis=1)
    ## Compute durations
    all_aoi_columns = [col for col in list(sub_aoi_hits_df.columns) if col not in ['subject', 'stim', 'frame',
                                                                                   'predictable', 'freq', 'mismatch', 'pseudo']]
    sub_results_df = pd.DataFrame(columns=[str(num) for num in range(0,6+len(all_aoi_columns))]) # For now name of cols are the number of futur elements in stim_dict
    for stim in sub_aoi_hits_df['stim'].unique():
        sub_aoi_hits_df_stim = sub_aoi_hits_df[sub_aoi_hits_df['stim']==stim]
        aoi_durations_dict = {}
        for aoi in all_aoi_columns:
            aoi_clusters = []
            for key, group in groupby(sub_aoi_hits_df_stim[aoi]):
                if key == 1:
                    aoi_clusters.append(len(list(group)))
            aoi_durations_dict[aoi] = [np.average(aoi_clusters)] if len(aoi_clusters) > 0 else [0]
        
        stim_characteristics_dict = {'freq': str(sub_aoi_hits_df_stim.iloc[0]['freq']),
                                     'mismatch': str(sub_aoi_hits_df_stim.iloc[0]['mismatch']),
                                     'pseudo': str(sub_aoi_hits_df_stim.iloc[0]['pseudo'])} if stim.startswith('word') else {'predictable': str(sub_aoi_hits_df_stim.iloc[0]['predictable'])}
        stim_dict = {**{'subject': str(sub_aoi_hits_df.iloc[0]['subject']),
                     'group': str(sub_aoi_hits_df.iloc[0]['subject'])[0],
                     'stim': stim},
                     **stim_characteristics_dict,
                     **aoi_durations_dict}
        if len(sub_results_df) == 0:
            if stim.startswith('sent'):
                sub_results_df = sub_results_df.iloc[:, :-2] # Only one condition in sentences
            sub_results_df.columns = stim_dict.keys()
        sub_results_df = sub_results_df.concat(pd.DataFrame(stim_dict), ignore_index=True)
    duration_hits_list.append(sub_results_df)


def duration_aoi_hits(args, subjects):
    ## Collect all relevant individual aoi hit tables in a big list
    print(f'## Retrieving aoi hit files for stim {args.stimulus} from subjects {subjects}')
    aoi_hit_files = []
    for i_subject, sub in tqdm(enumerate(subjects), total=len(subjects)):
        aoi_hit_files.append(glob(os.path.join(args.path2deriv,f'eye-tracking/aoi_hit_tables/{args.stimulus}_sub-{sub}_aoi.csv'))[0])
    print(f'\n{len(aoi_hit_files)} aoi hit files found')
    ## Create a dictionary containing participants' aoi hit table lists for each condition
    all_subjects_df_list = [pd.read_csv(df_subject).iloc[: , 1:] for df_subject in aoi_hit_files]
    if any([sub.startswith('3') for sub in subjects]):
        all_subjects_df_list = keep_common_stims(args, all_subjects_df_list)
    all_subjects_df_list = select_stims_for_cond(args, all_subjects_df_list)
    ## Compute duration of samples within each aoi for each participant
    duration_hits_list = mp.Manager().list()
    with mp.Pool(mp.cpu_count()) as pool:
        pool.starmap(compute_duration_hits, [(sub_df, duration_hits_list) for sub_df in all_subjects_df_list])
    pool.close()
    pool.join()
    duration_hits_list = list(duration_hits_list)
    duration_hits_df = pd.concat(duration_hits_list)
    all_aoi_columns = [col for col in list(duration_hits_df.columns) if col not in ['subject', 'group', 'stim', 'frame',
                                                                                   'predictable', 'freq', 'mismatch', 'pseudo']]
    duration_hits_df[all_aoi_columns] = duration_hits_df[all_aoi_columns].astype('float')
    for col in duration_hits_df.select_dtypes(include=['number']).columns:
        duration_hits_df[col] = duration_hits_df[col].map(lambda x: round(x,2))
    ## Exclude trials with no AOI hit at all (= those that had been excluded because of high data loss)
    duration_hits_df = duration_hits_df[(duration_hits_df[all_aoi_columns] != 0).any(axis=1)]
    ## Plotting
    from utils.viz import plot_duration_aoi_hits
    duration_hits_df = duration_hits_df.groupby(['subject', 'group']).mean().reset_index()
    duration_hits_df_melted = pd.melt(duration_hits_df, id_vars=['subject', 'group'],
                                      value_vars=['hand', 'face', 'eyes', 'lips', 'middle',
                                                 'face_left', 'face_right',
                                                 'lips_left', 'lips_right',
                                                 'eyes_left', 'eyes_right',
                                                 'middle_left', 'middle_right'],
                                      var_name='hit_type', value_name='duration')
    plot_duration_aoi_hits(args, duration_hits_df_melted)
        
        
def inspect_aoi_movements(args): # Compute and analyse AOI displacements for the two experiments
    pd.set_option('display.max_rows', None, 'display.max_columns', None)
    video_files = glob('utils/stim_videos/videos/*/*.mp4')
    aoi_displacement_df = pd.DataFrame(columns=['stim_type','stim_name','center_displacement'])
    for i_stim, video_file in tqdm(enumerate(video_files), total=len(video_files)):
        stim_name = os.path.splitext(os.path.basename(video_file))[0]
        # AOI coordinates
        df_aoi_file = glob(os.path.join(args.path2videocoords, f'*/{stim_name}_aoi_coordinates.csv'))[0]
        df_aoi = pd.read_csv(os.path.join(df_aoi_file))
        x_hand_cols = df_aoi.filter(regex='x_r_hand*').columns
        y_hand_cols = df_aoi.filter(regex='y_r_hand*').columns
        x_face_cols = df_aoi.filter(regex='x_face*').columns
        y_face_cols = df_aoi.filter(regex='y_face*').columns
        x_face_cols, y_face_cols = x_face_cols[12:], y_face_cols[12:]
        x_eyes_cols, y_eyes_cols = x_face_cols[:4], y_face_cols[:4]
        x_lips_cols, y_lips_cols = x_face_cols[8:12], y_face_cols[8:12]
        x_middle_cols, y_middle_cols = x_eyes_cols[2:].append(x_lips_cols[:2]), y_eyes_cols[2:].append(y_lips_cols[:2])
        x_lips_center, y_lips_center = None, None
        displacement_center_video = 0
        for frame_num in df_aoi.index:
            lips_pts = []
            for x_lips_pt_interest, y_lips_pt_interest in zip(x_lips_cols, y_lips_cols):
                lips_pts.append([df_aoi[x_lips_pt_interest].iloc[frame_num], df_aoi[y_lips_pt_interest].iloc[frame_num]])
            lips_pts = np.array(lips_pts, dtype=np.int32)
            x_lips_center_previous, y_lips_center_previous = x_lips_center, y_lips_center
            x_lips_center, y_lips_center = np.mean(lips_pts[:,0]), np.mean(lips_pts[:,1])
            if frame_num == 0:
                displacement_center_frame = 0
            else:
                displacement_center_frame = sqrt((x_lips_center-x_lips_center_previous)**2+(y_lips_center-y_lips_center_previous)**2)
            displacement_center_video = displacement_center_video + displacement_center_frame
        aoi_displacement_df.loc[i_stim] = np.repeat(np.nann, len(aoi_displacement_df.columns))
        aoi_displacement_df['stim_type'].iloc[i_stim] = stim_name[:4]
        aoi_displacement_df['stim_name'].iloc[i_stim] = stim_name
        aoi_displacement_df['center_displacement'].iloc[i_stim] = displacement_center_video/len(df_aoi)
    aoi_displacement_df['center_displacement'] = pd.to_numeric(aoi_displacement_df['center_displacement'])
    print(aoi_displacement_df.groupby('stim_type').describe())
    print(f"Words normality: {pg.normality(aoi_displacement_df[aoi_displacement_df['stim_type']=='word']['center_displacement'])}")
    print(f"Sentences normality: {pg.normality(aoi_displacement_df[aoi_displacement_df['stim_type']=='sent']['center_displacement'])}")
    print(f"T-test between stim types:\n{pg.ttest(aoi_displacement_df[aoi_displacement_df['stim_type']=='word']['center_displacement'], aoi_displacement_df[aoi_displacement_df['stim_type']=='sent']['center_displacement'])}")
    fig = plt.figure(figsize=(8,10))
    sns.boxplot(y='center_displacement', x='stim_type',
                data=aoi_displacement_df, palette='pastel', legend=False)
    sns.stripplot(y='center_displacement', x='stim_type', dodge=True,
                  data=aoi_displacement_df, palette='dark', legend=False)
    plt.ylabel('Total displacement for the mouth AOI center (normalized by nb of frames; in px)', fontsize=12)
    plt.title('Total displacement for the mouth AOI center', fontsize=16, pad=25)
    fig.savefig(os.path.join(args.path2figures,'aoi_movements/aoi_movements_mouth.png'))

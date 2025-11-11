#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CUSTIME metadata generation script: creates the metadata from the csv log from the paradigm
and outputs a pandas DataFrame for an MNE epochs object
"""


#%% Import modules

import os
import re
import warnings
import pandas as pd
import numpy as np
from glob import glob


#%% Generate epoch metadata for the 'words' experiment with CS users

warnings.simplefilter(action='ignore', category=FutureWarning) # To desactivate the df['col'][row_indexer] = value warning

def generate_metadata_wordsusers(self, raw_file):
    with pd.option_context('mode.chained_assignment', None): # To desactivate SettingWithCopyWarning
        # Retrieve the name of the task for the considered file
        task = raw_file[raw_file.find('task-')+5:raw_file.find('_run')]
        ##  Import the behavioral log file as a DataFrame
        if raw_file.endswith('fif'):
            csv_file_name = os.path.join(self.path2data, f'sub-{self.subject}/ses-{self.session_day}/beh',
                                         os.path.basename(raw_file).replace('_eeg','_beh').replace('.fif','.csv'))
            index_session = csv_file_name.find('_task-') # Append session info in file name if processing eeg
            csv_file_name = csv_file_name[:index_session] + f'_ses-{self.session_time}' + csv_file_name[index_session:]
        elif raw_file.endswith('asc'):
            csv_file_name = os.path.join(self.path2data, f'sub-{self.subject}/ses-{self.session_day}/beh',
                                         os.path.basename(raw_file).replace('.asc','_beh.csv'))
            csv_file_name = re.sub(r'ses-\d+', 'ses-*', csv_file_name)
        csv_file = glob(csv_file_name)
        assert len(csv_file) == 1, 'No corresponding behavioral file or more than one behavioral file found with these arguments!'
        csv_file = csv_file[0]
        df_log = pd.read_csv(csv_file, encoding='latin-1')
        ## Append missing information for s21 run 1's last trial (don't know why this is missing)
        if self.subject == '21' and raw_file[raw_file.find('run-')+4:raw_file.find('_beh')] == '1':
            df_log['response_validity'].iloc[134] = 0
            df_log['response_type'].iloc[134] = 'MISS'
        ## Handle the few cases with late button press logged in the wrong trial (see handwritten notes p28 ad 44)
        for index in df_log.index:
            if pd.notnull(df_log['click_time'].iloc[index]) and pd.isna(df_log['response_time'].iloc[index]):
                print(f'Late button press logged in the wrong trial at index {index}, putting it back to the previous trial.')
                # Transfer the click info to preciding word
                if pd.isnull(df_log['click_time'].iloc[index-1]): # If no button press was already recorded in the preceding trial
                    df_log['click_marker'].iloc[index-1] = 128 + df_log['word_marker'].iloc[index] # For some reason, such events were formed with the second word (wrong), but ok because we never use events to select trials
                    if df_log['pseudo'].iloc[index-1] == 0:
                        df_log['response_validity'].iloc[index-1] = 1
                        df_log['response_type'].iloc[index-1] = 'CR'
                    else:
                        df_log['response_validity'].iloc[index-1] = 0
                        df_log['response_type'].iloc[index-1] = 'MISS'
                    df_log['click_time'].iloc[index-1] = df_log['click_time'].iloc[index]
                    df_log['response_time'].iloc[index-1] = df_log['click_time'].iloc[index] - df_log['word_onset_time'].iloc[index-1]
                else: # If a button press was already recorded, create a button_press row 
                    df_log.loc[index-0.5] = np.repeat(np.nan, len(df_log.columns))
                    df_log['word'].loc[index-0.5] = df_log['word'].iloc[index] + '_bis' # Flag for preventing the creation of a button_press row below
                    df_log['freq'].loc[index-0.5] = df_log['freq'].iloc[index]
                    df_log['mismatch'].loc[index-0.5] = df_log['mismatch'].iloc[index]
                    df_log['num_in_cat'].loc[index-0.5] = df_log['num_in_cat'].iloc[index]
                    df_log['pseudo'].loc[index-0.5] = df_log['pseudo'].iloc[index]
                    df_log['click_marker'].loc[index-0.5] = 128 + df_log['word_marker'].iloc[index]
                    if df_log['pseudo'].iloc[index-1] == 0:
                        df_log['response_validity'].loc[index-0.5] = 1
                        df_log['response_type'].loc[index-0.5] = 'CR'
                    else:
                        df_log['response_validity'].loc[index-0.5] = 0
                        df_log['response_type'].loc[index-0.5] = 'MISS'
                    df_log['click_time'].loc[index-0.5] = df_log['click_time'].iloc[index]
                    df_log['response_time'].loc[index-0.5] = df_log['click_time'].iloc[index] - df_log['word_onset_time'].iloc[index-1]
                # Delete the click info on the current trial
                df_log['word'].iloc[index] = df_log['word'].iloc[index] + '_pb' # Flag for handling the order fixation/button_press below
                df_log['click_marker'].iloc[index] = np.nan
                df_log['click_time'].iloc[index] = np.nan
                if df_log['pseudo'].iloc[index] == 0:
                    df_log['response_validity'].iloc[index] = 1
                    df_log['response_type'].iloc[index] = 'CR'
                else:
                    df_log['response_validity'].iloc[index] = 0
                    df_log['response_type'].iloc[index] = 'MISS'
        df_log = df_log.sort_index().reset_index(drop = True)
        ##  Dissociate 'word_onset' and 'button_press' events: 1 row/event, vs 1/trial in the original behavioral log
        for index in df_log.index:
            if pd.notnull(df_log['click_marker'].iloc[index]) and '_bis' not in df_log['word'].iloc[index]:
                duplicated_row = df_log.iloc[index]
                duplicated_row['word_marker'] = np.nan 
                df_log['click_marker'].iloc[index] = np.nan
                df_log.loc[index+0.5] = duplicated_row
        df_log = df_log.sort_index().reset_index(drop = True)
        
        ## Create a metadata df and append relevant information from the log df
        df_metadata = pd.DataFrame(columns = ['event_marker', 'event_type',
                                              'word', 'freq', 'mismatch', 'num_in_cat', 'pseudo',
                                              'response_time', 'response_validity',
                                              'fixation_duration',
                                              'exp_type', 'group', 'subject', 'run'])
        for index in df_log.index:
            # Create an empty row
            df_metadata.loc[index] = np.repeat(np.nan, len(df_metadata.columns))
            # Append relevant information in the metadata for 'word_onset' and 'button_press' events
            df_metadata['word'].loc[index] = df_log['word'].loc[index]
            df_metadata['freq'].loc[index] = df_log['freq'].loc[index]
            df_metadata['mismatch'].loc[index] = df_log['mismatch'].loc[index]
            df_metadata['num_in_cat'].loc[index] = df_log['num_in_cat'].loc[index]
            df_metadata['pseudo'].loc[index] = df_log['pseudo'].loc[index]
            df_metadata['fixation_duration'].loc[index] = df_log['fixation_duration'].loc[index]
            # Append relevant information in the metadata for 'word_onset' events
            if pd.notnull(df_log['word_marker'].iloc[index]):
                df_metadata['event_type'].loc[index] = 'word_onset'
                df_metadata['event_marker'].loc[index] = df_log['word_marker'].loc[index]
            # Append relevant information in the metadata for 'button_press' events
            if pd.notnull(df_log['click_marker'].iloc[index]):
                df_metadata['event_type'].loc[index] = 'button_press'
                df_metadata['event_marker'].loc[index] = df_log['click_marker'].loc[index]
                df_metadata['response_time'].loc[index] = df_log['response_time'].loc[index]
                df_metadata['response_validity'].loc[index] = df_log['response_validity'].loc[index]
        # Append 'fixation_cross' rows before each 'word_onset' row
        for index in df_metadata.index:
            if df_metadata['event_type'].loc[index] == 'word_onset':
                if '_pb' in df_metadata['word'].loc[index]: # Handle the few cases with late click logged in the wrong trial (see handwritten notes p28 and 44), where the button_press should be after the fixation
                    df_metadata.loc[index-1.5] = np.repeat(np.nan, len(df_metadata.columns))    
                    df_metadata['event_marker'].loc[index-1.5], df_metadata['event_type'].loc[index-1.5] = 1, 'fixation_cross'
                    df_metadata['fixation_duration'].loc[index-1.5] = df_metadata['fixation_duration'].loc[index]
                    df_metadata['word'].loc[index] = df_metadata['word'].loc[index].replace('_pb', '')
                else:
                    df_metadata.loc[index-0.5] = np.repeat(np.nan, len(df_metadata.columns))
                    df_metadata['event_marker'].loc[index-0.5], df_metadata['event_type'].loc[index-0.5] = 1, 'fixation_cross'
                    df_metadata['fixation_duration'].loc[index-0.5] = df_metadata['fixation_duration'].loc[index]
            if '_bis' in df_metadata['word'].loc[index]:
                df_metadata['word'].loc[index] = df_metadata['word'].loc[index].replace('_bis', '')
        for index in df_metadata.index:
            df_metadata['exp_type'].loc[index] = task
            df_metadata['subject'].loc[index] = df_log['subject'].loc[0]
            df_metadata['group'].loc[index] = int(str(df_log['subject'].loc[0])[0])
            df_metadata['run'].loc[index] = df_log['run'].loc[0]
        df_metadata = df_metadata.sort_index().reset_index(drop = True)
    
    return df_metadata


#%% Generate epoch metadata for the 'words' experiment with control participants

def generate_metadata_wordsctrl(self, raw_file):
    with pd.option_context('mode.chained_assignment', None): # To desactivate SettingWithCopyWarning
        # Retrieve the name of the task for the considered file
        task = raw_file[raw_file.find('task-')+5:raw_file.find('_run')]
        ##  Import the behavioral log file as a DataFrame
        if raw_file.endswith('fif'):
            csv_file_name = os.path.join(self.path2data, f'sub-{self.subject}/ses-{self.session_day}/beh',
                                         os.path.basename(raw_file).replace('_eeg','_beh').replace('.fif','.csv'))
            index_session = csv_file_name.find('_task-') # Append session info in file name if processing eeg
            csv_file_name = csv_file_name[:index_session] + f'_ses-{self.session_time}' + csv_file_name[index_session:]
        elif raw_file.endswith('asc'):
            csv_file_name = os.path.join(self.path2data, f'sub-{self.subject}/ses-{self.session_day}/beh',
                                         os.path.basename(raw_file).replace('.asc','_beh.csv'))
            csv_file_name = re.sub(r'ses-\d+', 'ses-*', csv_file_name)
        csv_file = glob(csv_file_name)
        assert len(csv_file) == 1, 'No corresponding behavioral file or more than one behavioral file found with these arguments!'
        csv_file = csv_file[0]
        df_log = pd.read_csv(csv_file, encoding='latin-1')
        
        ## Create a metadata df and append relevant information from the log df
        df_metadata = pd.DataFrame(columns = ['event_marker', 'event_type',
                                              'word', 'freq', 'mismatch', 'num_in_cat', 'pseudo',
                                              'fixation_duration',
                                              'exp_type', 'group', 'subject', 'run'])
        for index in df_log.index:
            # Create an empty row
            df_metadata.loc[index] = np.repeat(np.nan, len(df_metadata.columns))
            # Append relevant information in the metadata for 'word_onset' event
            df_metadata['event_type'].loc[index] = 'word_onset'
            df_metadata['event_marker'].loc[index] = df_log['word_marker'].loc[index]
            df_metadata['word'].loc[index] = df_log['word'].loc[index]
            df_metadata['freq'].loc[index] = df_log['freq'].loc[index]
            df_metadata['mismatch'].loc[index] = df_log['mismatch'].loc[index]
            df_metadata['num_in_cat'].loc[index] = df_log['num_in_cat'].loc[index]
            df_metadata['pseudo'].loc[index] = df_log['pseudo'].loc[index]
            df_metadata['fixation_duration'].loc[index] = df_log['fixation_duration'].loc[index]
        # Append 'fixation_cross' rows before each 'word_onset' row
        for index in df_metadata.index:
            if df_metadata['event_type'].loc[index] == 'word_onset':
                df_metadata.loc[index-0.5] = np.repeat(np.nan, len(df_metadata.columns))
                df_metadata['event_marker'].loc[index-0.5], df_metadata['event_type'].loc[index-0.5] = 1, 'fixation_cross'
                df_metadata['fixation_duration'].loc[index-0.5] = df_metadata['fixation_duration'].loc[index]
        for index in df_metadata.index:
            df_metadata['exp_type'].loc[index] = task
            df_metadata['subject'].loc[index] = df_log['subject'].loc[0]
            df_metadata['group'].loc[index] = int(str(df_log['subject'].loc[0])[0])
            df_metadata['run'].loc[index] = df_log['run'].loc[0]
        df_metadata = df_metadata.sort_index().reset_index(drop = True)
    
    return df_metadata


#%% Generate epoch metadata for the 'sentences' experiment with CS users

def generate_metadata_sentencesusers(self, raw_file):
    with pd.option_context('mode.chained_assignment', None): # To desactivate SettingWithCopyWarning
        # Retrieve the name of the task for the considered file
        task = raw_file[raw_file.find('task-')+5:raw_file.find('_run')]
        ##  Import the behavioral log file as a DataFrame
        if raw_file.endswith('fif'):
            csv_file_name = os.path.join(self.path2data, f'sub-{self.subject}/ses-{self.session_day}/beh',
                                         os.path.basename(raw_file).replace('_eeg','_beh').replace('.fif','.csv'))
            index_session = csv_file_name.find('_task-') # Append session info in file name if processing eeg
            csv_file_name = csv_file_name[:index_session] + f'_ses-{self.session_time}' + csv_file_name[index_session:]
        elif raw_file.endswith('asc'):
            csv_file_name = os.path.join(self.path2data, f'sub-{self.subject}/ses-{self.session_day}/beh',
                                         os.path.basename(raw_file).replace('.asc','_beh.csv'))
            csv_file_name = re.sub(r'ses-\d+', 'ses-*', csv_file_name)
        csv_file = glob(csv_file_name)
        assert len(csv_file) == 1, 'No corresponding behavioral file or more than one behavioral file found with these arguments!'
        csv_file = csv_file[0]
        df_log = pd.read_csv(csv_file, encoding='latin-1')
        ##  Dissociate 'question_onset' and 'button_press' events: 1 row/event, vs 1/trial in the original behavioral log
        for index in df_log.index:
            if pd.notnull(df_log['click_marker'].iloc[index]):
                duplicated_row = df_log.iloc[index]
                df_log['click_marker'].iloc[index] = np.nan
                duplicated_row['is_question'] = 0
                df_log.loc[index + 0.5] = duplicated_row
        df_log = df_log.sort_index().reset_index(drop = True)
        
        ## Create a metadata df and append relevant information from the log df
        df_metadata = pd.DataFrame(columns = ['event_marker', 'event_type',
                                              'sentence', 'num_pair', 'predictable', 'num_sentence', 'diff_of_prob',
                                              'response_time', 'response_validity',
                                              'fixation_duration',
                                              'exp_type', 'group', 'subject', 'run'])
        for index in df_log.index:
            # Create an empty row
            df_metadata.loc[index] = np.repeat(np.nan, len(df_metadata.columns))
            # Append relevant information in the metadata for 'sentence_onset', 'question_onset' and 'button_press' events
            df_metadata['sentence'].loc[index] = df_log['sentence'].loc[index]
            df_metadata['fixation_duration'].loc[index] = df_log['fixation_duration'].loc[index]
            # Append relevant information in the metadata for 'sentence_onset' events
            if pd.notnull(df_log['sentence_marker'].iloc[index]):
                df_metadata['event_type'].loc[index] = 'sentence_onset'
                df_metadata['event_marker'].loc[index] = df_log['sentence_marker'].loc[index]
                df_metadata['num_pair'].loc[index] = df_log['num_pair'].loc[index]
                df_metadata['predictable'].loc[index] = df_log['predictable'].loc[index]
                df_metadata['num_sentence'].loc[index] = df_log['num_sentence'].loc[index]
                df_metadata['diff_of_prob'].loc[index] = df_log['diff_of_prob'].loc[index]
            # Append relevant information in the metadata for 'button_press' events
            if pd.notnull(df_log['click_marker'].iloc[index]):
                df_metadata['event_type'].loc[index] = 'button_press'
                df_metadata['event_marker'].loc[index] = df_log['click_marker'].loc[index]
                df_metadata['response_time'].loc[index] = df_log['response_time'].loc[index]
                df_metadata['response_validity'].loc[index] = df_log['response_validity'].loc[index]
            # Append relevant information in the metadata for 'question_onset' events
            if df_log['is_question'].iloc[index] == 1:
                df_metadata['event_type'].loc[index] = 'question_onset'
                df_metadata['event_marker'].loc[index] = 112
        # Append 'fixation_cross' rows before each 'sentence_onset' row
        for index in df_metadata.index:
            if df_metadata['event_type'].loc[index] == 'sentence_onset':
                df_metadata.loc[index-0.5] = np.repeat(np.nan, len(df_metadata.columns))
                df_metadata['event_marker'].loc[index-0.5], df_metadata['event_type'].loc[index-0.5] = 1, 'fixation_cross'
                df_metadata['fixation_duration'].loc[index-0.5] = df_metadata['fixation_duration'].loc[index]
        for index in df_metadata.index:
            df_metadata['exp_type'].loc[index] = task
            df_metadata['subject'].loc[index] = df_log['subject'].loc[0]
            df_metadata['group'].loc[index] = int(str(df_log['subject'].loc[0])[0])
            df_metadata['run'].loc[index] = df_log['run'].loc[0]
        df_metadata = df_metadata.sort_index().reset_index(drop = True)
    
    return df_metadata
    

#%% Generate epoch metadata for the 'sentences' experiment with control participants

def generate_metadata_sentencesctrl(self, raw_file):
    with pd.option_context('mode.chained_assignment', None): # To desactivate SettingWithCopyWarning
        # Retrieve the name of the task for the considered file
        task = raw_file[raw_file.find('task-')+5:raw_file.find('_run')]
        ##  Import the behavioral log file as a DataFrame
        if raw_file.endswith('fif'):
            csv_file_name = os.path.join(self.path2data, f'sub-{self.subject}/ses-{self.session_day}/beh',
                                         os.path.basename(raw_file).replace('_eeg','_beh').replace('.fif','.csv'))
            index_session = csv_file_name.find('_task-') # Append session info in file name if processing eeg
            csv_file_name = csv_file_name[:index_session] + f'_ses-{self.session_time}' + csv_file_name[index_session:]
        elif raw_file.endswith('asc'):
            csv_file_name = os.path.join(self.path2data, f'sub-{self.subject}/ses-{self.session_day}/beh',
                                         os.path.basename(raw_file).replace('.asc','_beh.csv'))
            csv_file_name = re.sub(r'ses-\d+', 'ses-*', csv_file_name)
        csv_file = glob(csv_file_name)
        assert len(csv_file) == 1, 'No corresponding behavioral file or more than one behavioral file found with these arguments!'
        csv_file = csv_file[0]
        df_log = pd.read_csv(csv_file, encoding='latin-1')
        
        ## Create a metadata df and append relevant information from the log df
        df_metadata = pd.DataFrame(columns = ['event_marker', 'event_type',
                                              'sentence', 'num_pair', 'predictable', 'num_sentence', 'diff_of_prob',
                                              'fixation_duration',
                                              'exp_type', 'group', 'subject', 'run'])
        for index in df_log.index:
            # Create an empty row
            df_metadata.loc[index] = np.repeat(np.nan, len(df_metadata.columns))
            # Append relevant information in the metadata for 'sentence_onset' events
            df_metadata['event_type'].loc[index] = 'sentence_onset'
            df_metadata['event_marker'].loc[index] = df_log['sentence_marker'].loc[index]
            df_metadata['sentence'].loc[index] = df_log['sentence'].loc[index]
            df_metadata['num_pair'].loc[index] = df_log['num_pair'].loc[index]
            df_metadata['predictable'].loc[index] = df_log['predictable'].loc[index]
            df_metadata['num_sentence'].loc[index] = df_log['num_sentence'].loc[index]
            df_metadata['diff_of_prob'].loc[index] = df_log['diff_of_prob'].loc[index]
            df_metadata['fixation_duration'].loc[index] = df_log['fixation_duration'].loc[index]
        # Append 'fixation_cross' rows before each 'sentence_onset' row
        for index in df_metadata.index:
            if df_metadata['event_type'].loc[index] == 'sentence_onset':
                df_metadata.loc[index-0.5] = np.repeat(np.nan, len(df_metadata.columns))
                df_metadata['event_marker'].loc[index-0.5], df_metadata['event_type'].loc[index-0.5] = 1, 'fixation_cross'
                df_metadata['fixation_duration'].loc[index-0.5] = df_metadata['fixation_duration'].loc[index]
        for index in df_metadata.index:
            df_metadata['exp_type'].loc[index] = task
            df_metadata['subject'].loc[index] = df_log['subject'].loc[0]
            df_metadata['group'].loc[index] = int(str(df_log['subject'].loc[0])[0])
            df_metadata['run'].loc[index] = df_log['run'].loc[0]
        df_metadata = df_metadata.sort_index().reset_index(drop = True)
    
    return metadata

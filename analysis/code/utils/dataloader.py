#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DataLoader() class: Handles Eye-tracking data for epoching and plotting results of the CUSTIME experiment
"""


# %% Import modules

import os
import re
import mne
import pickle
import warnings
import numpy as np
import pandas as pd
from glob import glob
from autoreject import get_rejection_threshold
from utils.generate_metadata import *
from utils.contrasts import load_contrasts
from utils.viz import clean_names


# %% DataLoader class definition

class DataLoader():
    def __init__(self, args, subject, script):
        print(f'\n\n\n###### Processing subject {subject} ######\n')
        self.subject = subject
        self.subjects = args.subjects
        self.task = args.task
        # self.stimulus = args.stimulus
        self.run = args.run
        self.report = args.report
        self.session_day = args.session_day
        self.session_time = args.session_time
        self.path2data = args.path2data
        self.path2deriv = args.path2deriv
        self.path2figures = args.path2figures
        self.ref_file = f'sub-{self.subject}/ses-{self.session_day}/foldertype/' \
                        f'sub-{self.subject}_task-{self.task}_run-{self.run}_datatype.ext'
        self.overwrite_report = args.overwrite_report
        elif script == 'behav':
            pass
        elif script == 'et':
            self.contrast = load_contrasts(args)[args.contrast]
            self.contrast_name = args.contrast
            pass

    # %% Generate et events

    def generate_et_events_annotations(self, raw, metadata):
        print('\nGenerate et events')
        message_annotations = {str(event_id) for event_id in set(
            raw.annotations.description) if event_id not in ['fixation', 'saccade', 'blink']}
        message_annotations = {event_id: int("".join(
            [s for s in event_id if s.isdigit()])) for event_id in message_annotations}
        events, event_dict = mne.events_from_annotations(
            raw, event_id=message_annotations, verbose=None)
        # Only keep fixation/saccade/blink annotations to avoid future confusions
        filtered_annotations = [ann for ann in raw.annotations if ann['description'] in [
            'fixation', 'saccade', 'blink']]
        onsets = [ann['onset'] for ann in filtered_annotations]
        durations = [ann['duration'] for ann in filtered_annotations]
        descriptions = [ann['description'] for ann in filtered_annotations]
        new_annotations = mne.Annotations(onsets, durations, descriptions)
        raw.set_annotations(new_annotations)
        # Test the correspondance between events and metadata
        for row in metadata.index:
            # Change FA to HIT when relevant (we want to get rid of the RT limit that was implemented in the stimulation script)
            if 'Syll' in metadata['exp_type'].iloc[row]:
                if events[row][2] == 31 and metadata['event_marker'].iloc[row] == 95:
                    print(f'Changing FA into HIT in event number {row}')
                    events[row][2] = 95
            assert metadata['event_marker'].iloc[row] == events[row][
                2], f"Mismatch between event and metadata on row {row}: event = {events[row][2]} and metadata = {metadata['event_marker'].iloc[row]}"
        return events, event_dict


    def create_et_raws(self):
        print('\n## Creating eye-tracking raw file(s)')
        et_files = []
        for task in self.task:
            et_fn = self.ref_file.replace('foldertype', 'beh').replace(
                '_task', f'_ses-{self.session_time}_task').replace(f'{self.task}', f'{task}').replace('_datatype.ext', '.asc')
            et_fn = glob(os.path.join(self.path2data, et_fn))
            et_files.append(et_fn)
        # Flatten the list of files
        et_files = [file for task_files in et_files for file in task_files]
        et_files.sort()
        print(f'{len(et_files)} run(s) found.')
        et_raws = [None]*len(et_files)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            for i_run in range(len(et_files)):
                et_raws[i_run] = mne.io.read_raw_eyelink(et_files[i_run])
                # Change 'BAD_blink' to 'blink' to avoid dropping the epochs containing a blink
                bad_blink_indices = [i for i, desc in enumerate(
                    et_raws[i_run].annotations.description) if desc == 'BAD_blink']
                for idx in bad_blink_indices:
                    et_raws[i_run].annotations.description[idx] = 'blink'
                # For the two participants with both eyes recorded, average the channels
                if len(mne.pick_channels_regexp(et_raws[i_run].info['ch_names'], 'xpos_.')) == 2:
                    # Chose not to average as events are computed separately for each eye, would not now which ones to consider
                    print('Detecting the two eyes, keeping the right eye')
                    et_raws[i_run].drop_channels(
                        ['xpos_left', 'ypos_left', 'pupil_left'])
                # Rename channels, only processing existing channels

                def rename_channels(raw, rename_dict):
                    current_ch_names = raw.ch_names
                    filtered_rename_dict = {
                        old: new for old, new in rename_dict.items() if old in current_ch_names}
                    raw.rename_channels(filtered_rename_dict)
                rename_dict = {'xpos_right': 'et_x', 'ypos_right': 'et_y', 'pupil_right': 'et_pupil',
                               'xpos_left': 'et_x', 'ypos_left': 'et_y', 'pupil_left': 'et_pupil'}
                rename_channels(et_raws[i_run], rename_dict)
        return et_files, et_raws


    def load_behav_df(self):
        print('\n## Loading behavioral file(s)')
        csv_files = []
        for task in self.task:
            csv_files_name = self.ref_file.replace('foldertype', 'beh').replace(
                '_task', f'_ses-{self.session_time}_task').replace('datatype.ext', 'beh.csv')
            csv_files_task = glob(os.path.join(self.path2data, csv_files_name))
            csv_files.append(csv_files_task)
        # Flatten the list of files
        csv_files = [file for task_files in csv_files for file in task_files]
        csv_files.sort()
        print(f'{len(csv_files)} corresponding run(s) found.')
        behav_dfs = [None]*len(csv_files)
        for i_run in range(len(csv_files)):
            behav_df_run = pd.read_csv(csv_files[i_run], encoding='latin-1')
            # Change FA to HIT when relevant (we want to get rid of the RT limit that was implemented in the stimulation script)
            if 'Syll' in csv_files[i_run]:
                for index in behav_df_run.index:
                    if behav_df_run['target'].iloc[index] == 1 and pd.notnull(behav_df_run['click_time'].iloc[index]):
                        behav_df_run['click_marker'].iloc[index] = 95
                        behav_df_run['response_validity'].iloc[index] = 1
                        behav_df_run['response_type'].iloc[index] = 'HIT'
            # Handle the few cases with late FA logged in the wrong trial (see handwritten notes p28)
            if 'SyllCS' in csv_files[i_run]:
                for index in behav_df_run.index:
                    if pd.notnull(behav_df_run['click_time'].iloc[index]) and behav_df_run['response_type'].iloc[index] == 'FA' and pd.isnull(behav_df_run['response_time'].iloc[index]):
                        print(
                            f'Late FA logged in the wrong trial at index {index}.')
                        if pd.isnull(behav_df_run['click_time'].iloc[index-1]):
                            # Transfer the click info to preciding trial only if no button press was recorded in the preceding trial
                            print('Putting it back to the previous trial.')
                            behav_df_run['click_marker'].iloc[index-1] = 31
                            behav_df_run['response_validity'].iloc[index-1] = 0
                            behav_df_run['response_type'].iloc[index-1] = 'FA'
                            behav_df_run['click_time'].iloc[index -
                                                            1] = behav_df_run['click_time'].iloc[index]
                            behav_df_run['response_time'].iloc[index-1] = behav_df_run['click_time'].iloc[index] - \
                                behav_df_run['syllable_onset_time'].iloc[index-1]
                        # Delete the click info on the current trial
                        behav_df_run['click_marker'].iloc[index] = np.nan
                        behav_df_run['click_time'].iloc[index] = np.nan
                        if behav_df_run['target'].iloc[index] == 0:
                            behav_df_run['response_validity'].iloc[index] = 1
                            behav_df_run['response_type'].iloc[index] = 'CR'
                        else:
                            behav_df_run['response_validity'].iloc[index] = 0
                            behav_df_run['response_type'].iloc[index] = 'MISS'
            behav_dfs[i_run] = behav_df_run
        if behav_dfs:  # Only concatenate if the list is not empty
            behav_df = pd.concat(behav_dfs, ignore_index=True)
        else:
            behav_df = pd.DataFrame()  # Create an empty DataFrame
        return behav_df


    def load_initialize_report(self, report_type=None):
        print('\n## Loading/Initializing report file')
        report_fn = self.ref_file.replace(
            'foldertype/', '').replace('datatype.ext', 'report.hdf5')
        if 'preproc' not in report_type:
            report_fn = report_fn.replace(f'_run-{self.run}', '')
        report_file = glob(os.path.join(self.path2deriv, 'reports', report_fn))
        assert len(
            report_file) <= 1, 'More than one report file found with these arguments!'
        if len(report_file) == 0:
            print(
                f'No report file corresponding to {report_file} was found, initializing a new one.')
            report = mne.Report(
                title=f'Report - sub-{self.subject}_task-{self.task}_run-{self.run}')
            mne.sys_info(fid=None, show_paths=True, dependencies='user')
            report.add_sys_info(title='System information',
                                tags=('mne_sysinfo', f'run_{self.run}'))
            print('***** Added syst info in report')
        if report_type == 'preproc_init' and self.overwrite_report:
            print('Overwriting the report file.')
            report = mne.Report(
                title=f'Report - sub-{self.subject}_task-{self.task}')
            mne.sys_info(fid=None, show_paths=True, dependencies='user')
            report.add_sys_info(title='System information',
                                tags=('mne_sysinfo', f'run_{self.run}'))
            print('***** Added syst info in report')
        else:  # If a corresponding report file was found and we don't want to overwrite it, load it
            print('Loading the report file.')
            report_file = report_file[0]
            report = mne.open_report(report_file)
        return report
    

    # %% Save

    def save_et_raw(self, et_raw):
        print('\n## Saving eye-tracking raw file')
        et_raw_name = self.ref_file.replace('foldertype', 'beh').replace(
            'ses-*', f'ses-{self.session_day}').replace('datatype.ext', 'beh.fif')
        et_raw_dir = os.path.join(
            self.path2deriv, 'preprocessed_raws', os.path.dirname(et_raw_name))
        os.makedirs(et_raw_dir, exist_ok=True)
        et_raw.save(os.path.join(
            et_raw_dir, os.path.basename(et_raw_name)), overwrite=True)


    def save_report(self, report, report_type=''):
        print('\n## Saving report file')
        report_name = self.ref_file.replace('foldertype/', '').replace(
            'ses-*', f'ses-{self.session_day}').replace('datatype.ext', 'report.hdf5').replace(f'_run-{self.run}', '')
        report_dir = os.path.join(
            self.path2deriv, 'reports', os.path.dirname(report_name))
        os.makedirs(report_dir, exist_ok=True)
        report.save(os.path.join(
            report_dir, os.path.basename(report_name)), overwrite=True)
        report.save(os.path.join(report_dir, os.path.basename(
            report_name.replace('hdf5', 'html'))), overwrite=True)


    def save_epochs(self, args, epochs, data_type):
        print('\n## Saving epochs')
        run_name, task_name = clean_names(args)
        if "vs" in self.contrast_name:  # For contrasts containing group comparison, save epochs under the base contrast's name
            groups = re.search(
                r'^(.*)_([0-9](?:\+[0-9])?)(?:vs)([0-9](?:\+[0-9])?)$', args.contrast)
            contrast = groups.group(1)
            print("Contrast contains group comparison, save epochs under the base contrast's name")
        else:
            contrast = self.contrast_name
        epochs_file = f'contrast-{contrast}_sub-{self.subject}_task-{task_name}_run-{run_name}_epo.fif'
        epochs_dir = os.path.join(
            self.path2deriv, f'epochs/sub-{self.subject}/ses-{args.session_day}/{task_name}/{data_type}/{contrast}')
        os.makedirs(epochs_dir, exist_ok=True)
        epochs.save(os.path.join(epochs_dir, epochs_file), overwrite=True)


    def save_et_annotations(self, args, et_annotations):
        print('\n## Saving eye-tracking annotations (fix/sacc/blink)')
        run_name, task_name = clean_names(args)
        annots_file = f'contrast-{self.contrast_name}_sub-{self.subject}_task-{task_name}_run-{run_name}_annot.pkl'
        annots_dir = os.path.join(
            self.path2deriv, f'epochs/sub-{self.subject}/ses-{args.session_day}/{task_name}/beh/{self.contrast_name}')
        os.makedirs(annots_dir, exist_ok=True)
        with open(os.path.join(annots_dir, annots_file), 'wb') as pkl:
            pickle.dump(et_annotations, pkl)
        print('Eye-tracking annotations saved')


    # %% Epoch

    def generate_metadata(self, raw_file, i_run=None):
        # Retrieve the name of the task for the considered file
        task = raw_file[raw_file.find('task-')+5:raw_file.find('_run')]
        if task == 'WordsUsers':
            metadata = generate_metadata_wordsusers(self, raw_file)
            # Not implemented in eye-tracking functions
            if hasattr(self, 'add_syllables') and self.add_syllables:
                metadata = add_syll_in_metadata_words(self, metadata)
        if task == 'WordsCtrl':
            metadata = generate_metadata_wordsctrl(self, raw_file)
            if hasattr(self, 'add_syllables') and self.add_syllables:
                metadata = add_syll_in_metadata_words(self, metadata)
        if task == 'SyllCS' or task == 'SyllReading':
            metadata = generate_metadata_syllables(self, raw_file)
        if task == 'SentencesUsers':
            metadata = generate_metadata_sentencesusers(self, raw_file)
            if hasattr(self, 'add_syllables') and self.add_syllables:
                metadata = add_syll_in_metadata_sentences(self, metadata)
        if task == 'SentencesCtrl':
            metadata = generate_metadata_sentencesctrl(self, raw_file)
            if hasattr(self, 'add_syllables') and self.add_syllables:
                metadata = add_syll_in_metadata_sentences(self, metadata)
        if self.contrast_name == 'eye_tracking' and task == 'SyllCS' and self.subject == '216' and i_run == 0:
            metadata = metadata[:453] # Some events were not registered, don't know why
        metadata['event_marker'] = metadata['event_marker'].astype(int)
        return metadata


    def epoch_data(self, raw, events, metadata, report):
        picks = ['eyegaze', 'pupil']
        baseline = None
        self.contrast == 'eye_tracking'
        quality_check_metadata = {}
        epochs_original = mne.Epochs(raw, events, # Just to have the metadata of unprocessed raws
                                     tmin=self.contrast['epoch_times'][0],
                                     tmax=self.contrast['epoch_times'][1],
                                     metadata=metadata,
                                     reject=None,
                                     reject_by_annotation=False,
                                     preload=True,
                                     baseline=baseline,
                                     picks=picks)
        quality_check_metadata['original'] = epochs_original.metadata
        print(f"# {len(quality_check_metadata['original'])} epochs in original raw\n")
        epochs = mne.Epochs(raw, events,
                            tmin=self.contrast['epoch_times'][0],
                            tmax=self.contrast['epoch_times'][1],
                            metadata=metadata,
                            reject=None,
                            reject_by_annotation=True,
                            preload=True,
                            baseline=baseline,
                            picks=picks)
        # If an epoch was dropped because too long for the raw, recreate it
        # List events that were droppped
        dropped_epochs_index = []
        for i_drop_log, drop in enumerate(epochs.drop_log):
            if drop == ('TOO_SHORT',) and ('onset' in metadata.iloc[i_drop_log]['event_type'] or metadata.iloc[i_drop_log]['event_type'] == 'button_press'):
                dropped_epochs_index.append(i_drop_log)
        epochs.drop_log = tuple([item for idx, item in enumerate(
            list(epochs.drop_log)) if idx not in dropped_epochs_index])
        if len(dropped_epochs_index) > 0:
            # Recreate corresponding epochs with their metadata and the remaining time of recording
            print(
                f'{len(dropped_epochs_index)} relevant dropped epoch(s) to recreate...')
            recreated_epochs = []
            original_samples = []
            nb_reconstructed_epochs = 0
            for i_drop_log in dropped_epochs_index:
                event_sample = events[i_drop_log][0]
                event_time = event_sample / raw.info['sfreq']
                tmin = self.contrast['epoch_times'][0]
                # Seconds remaining in the recording after the event
                dropped_tmax = raw.times[-1] - event_time
                if dropped_tmax < tmin:
                    print(
                        f'Skipped epoch {i_drop_log}: not enough data ({dropped_tmax:.3f}s left, need ≥ {tmin:.3f}s).')
                    continue
                dropped_epoch = mne.Epochs(raw, [events[i_drop_log]],
                                           tmin=self.contrast['epoch_times'][0],
                                           tmax=dropped_tmax,
                                           reject=None,
                                           reject_by_annotation=True,
                                           preload=True,
                                           baseline=baseline,
                                           picks=picks)
                # Pad epoch data with zeros to reach the time of other epochs (permits concatenation) + fetch its metadata
                dropped_epoch_array = dropped_epoch.get_data(copy=False)
                nb_channels = epochs.get_data().shape[1]
                time_to_add = np.zeros((1, nb_channels, len(
                    epochs.times) - len(dropped_epoch.times)))
                dropped_epoch_array = np.concatenate(
                    (dropped_epoch_array, time_to_add), axis=2)
                dropped_epoch_metadata = pd.DataFrame(
                    metadata.iloc[i_drop_log]).transpose()
                dropped_epoch = mne.EpochsArray(dropped_epoch_array, info=epochs.info,
                                                tmin=self.contrast['epoch_times'][0],
                                                metadata=dropped_epoch_metadata, events=dropped_epoch.events)
                recreated_epochs.append(dropped_epoch)
                # Not sure why concatenation makes new samples for recreated epochs
                original_samples.append(events[i_drop_log][0])
                nb_reconstructed_epochs = nb_reconstructed_epochs + 1
            if nb_reconstructed_epochs > 0:
                with warnings.catch_warnings():
                    # 'This filename does not conform to MNE naming conventions...'
                    warnings.simplefilter('ignore', RuntimeWarning)
                    for i_recreated_epoch, _ in enumerate(recreated_epochs):
                        epochs = mne.concatenate_epochs(
                            [epochs, recreated_epochs[i_recreated_epoch]])
                    epochs.events[-nb_reconstructed_epochs:][:,
                                                             0] = original_samples
        quality_check_metadata['preprocessed'] = epochs.metadata
        print(f"# {len(quality_check_metadata['preprocessed'])} epochs in processed raw\n")
        print('Computing autoreject threshold...')
        reject_criteria = get_rejection_threshold(epochs, decim=2)
        epochs.drop_bad(reject=reject_criteria)
        quality_check_metadata['autoreject'] = epochs.metadata
        print(f"# {len(quality_check_metadata['autoreject'])} epochs after autoreject threshold application\n")
        if self.report:
            if data_type == 'et':
                report.add_events(events=events, title='ET events',
                                  sfreq=raw.info['sfreq'], replace=True,
                                  tags=('epoch', 'analysis', 'ctrl', 'et', f'run_{self.run}'))
                report.add_figure(fig=epochs.plot_drop_log(), title='ET bad epochs log',
                                  replace=True, tags=('epoch', 'analysis', 'ctrl', 'et', f'run_{self.run}'))
        return epochs


    def add_epoch_annotations(self, raw, epochs, report):
        epochs_annotations = []
        for epoch_idx, epoch in enumerate(epochs):
            epoch_start = epochs.times[0] + \
                epochs.events[epoch_idx, 0] / raw.info['sfreq']
            epoch_end = epochs.times[-1] + \
                epochs.events[epoch_idx, 0] / raw.info['sfreq']
            epoch_annots = raw.annotations[(raw.annotations.onset >= epoch_start) &
                                           (raw.annotations.onset + raw.annotations.duration <= epoch_end)]
            epochs_annotations.append({
                'epoch_idx': epoch_idx,
                'annotations': [str(annot['description']) for annot in epoch_annots],
                'onsets': [round(annot['onset']-epochs.events[epoch_idx, 0]/raw.info['sfreq'], 3) for annot in epoch_annots],
                'durations': [annot['duration'] for annot in epoch_annots]
            })
        # Epochs are longer than real stims so annotations exceed stim length at this stage
        return epochs_annotations

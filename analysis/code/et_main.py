#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Import modules

import argparse
import multiprocessing as mp
from utils.dataloader import DataLoader
from mne import concatenate_epochs
from utils.stats import retrieve_ids, retrieve_epochs, create_aoi_hit_tables, et_quality_check,\
    compute_precision_accuracy, detect_smooth_pursuits, compute_nb_events, proportion_aoi_hits, duration_aoi_hits, inspect_aoi_movements
from utils.viz import plot_aoi_on_vid, plot_pupil, plot_heatmap, plot_aoi_on_vid_empty
from utils.eyetracking import create_aoi_tables


#%% Parse arguments

parser = argparse.ArgumentParser()

# Data to analyse
parser.add_argument('--report', '--rep', type=bool, default=False)
parser.add_argument('--contrast', '--c', default='eye_tracking')
parser.add_argument('--swap_queries', '--swap', '--sq', default=False)
parser.add_argument('--task', '--t', type=str, default=['WordsUsers', 'WordsCtrl'],
                    nargs='+', choices=['WordsUsers', 'SentencesUsers',
                                        'WordsCtrl', 'SentencesCtrl'])
# Data
parser.add_argument('--subjects', '--s', type=str, default=['*'],
                    help='Subjects numbers, e.g., 11 then 21 or * (all) or 1 (group 1) or 1 then 2 (groups 1 and 2)' +\
                    'Group 1: deaf cs users; group 2: hearing cs users; group 3: hearing cs naive controls',
                    nargs='+')
parser.add_argument('--subjects_to_exclude', '--s_ex', type=str, default=['phantom','pilote', '24'],
                    nargs='*')
parser.add_argument('--session_day', '--day', default='*')
parser.add_argument('--session_time', '--time', default='*')
parser.add_argument('--run', '--r', default='*', # Si Words/SentencesUsers, mettre *
                    choices=['1', '2', '3', '4', '*'])
# Parameters
parser.add_argument('--loss_rate_criteria', '--loss', type=int, default=25,
                    help='In %')
parser.add_argument('--alpha', '--alpha_level', type=float, default=95,
                    help='Determine the one minus alpha value for confidence intervals (typically 95, 99 or 99.9)')
parser.add_argument('--cond_dict', '--cond', type=dict, default={'words': ['all','pseudo_0_freq_h','pseudo_0_freq_l',
                                                                           'pseudo_0_mismatch_0','pseudo_0_mismatch_1','pseudo_0_mismatch_2',
                                                                           'pseudo_0_mismatch_0_freq_h', 'pseudo_0_mismatch_0_freq_l',
                                                                           'pseudo_0_mismatch_1_freq_h', 'pseudo_0_mismatch_1_freq_l',
                                                                           'pseudo_0_mismatch_2_freq_h', 'pseudo_0_mismatch_2_freq_l',
                                                                           'pseudo_0','pseudo_1'],
                                                                 'sentences': ['all','predictable_0','predictable_1'],
                                                                 'syllables': ['all','rounded','unrounded','bilabial', 'coronal',
                                                                               'one_finger', 'five_fingers', 'ear', 'chin',
                                                                               'm', 'p', 'd', 't', 'a', 'e', 'o', 'u']})
# Paths
parser.add_argument('--path2data', default='../data')
parser.add_argument('--path2deriv', default='../data/derivatives')
parser.add_argument('--path2figures', default='../figures/eye-tracking')
parser.add_argument('--path2syllonsets', default='utils/syllable_onsets')
parser.add_argument('--path2videocoords', default='utils/stim_videos/video_coordinates')
# First order computations
parser.add_argument('--analysis_level_1', '--level_1', '--analysis_order_1', '--order_1', type=bool, default=False)
parser.add_argument('--recompute_epochs_annotations', type=bool, default=False)
parser.add_argument('--create_aoi_hit_tables', '--table_aoi', '--hit_aoi', type=bool, default=False)
# Second order computations
parser.add_argument('--analysis_level_2', '--level_2', '--analysis_order_2', '--order_2', type=bool, default=True)
parser.add_argument('--quality_check', '--qc', type=bool, default=False)
parser.add_argument('--compute_precision_accuracy', '--precision_accuracy', type=bool, default=False)
parser.add_argument('--detect_smooth_pursuits', '--smooth_pursuits', '--pursuits', type=bool, default=False)
parser.add_argument('--compute_nb_events', '--nb_events', type=bool, default=False)
parser.add_argument('--proportion_aoi_hits', '--prop_aoi', '--prop_hits', type=bool, default=True)
parser.add_argument('--duration_aoi_hits', '--dur_aoi', '--dur_hits', type=bool, default=False)
# Other computations
parser.add_argument('--create_aoi_tables', type=bool, default=False)
parser.add_argument('--plot_aoi_on_vid', '--p_aoi', type=bool, default=False)
parser.add_argument('--plot_aoi_on_vid_empty', type=bool, default=False)
parser.add_argument('--inspect_aoi_movements', type=bool, default=False)
parser.add_argument('--plot_proportion_aoi_hits', '--plot_proportion', type=bool, default=False)
parser.add_argument('--plot_pupil', '--pupil', type=bool, default=False)
parser.add_argument('--plot_heatmap', '--heatmap', type=bool, default=False)
# Plot and analyses parameters
parser.add_argument('--parallelization', '--para', '--mp', '--p', type=bool, default=False,
                    help='Parallelize the 1st level analysis (if yes the plots will not be showed inline and will be saved regardless of the save options)')
parser.add_argument('--add_syllables', '--syll', type=bool, default=False)
parser.add_argument('--generate_syllables', '--gen_syll', type=bool, default=False)
parser.add_argument('--autoreject', '--ar', type=bool, default=False)

args = parser.parse_args()


#%% Arrange coordinates original table
# To run only once unless the coordinates for aois are to be changed

if args.create_aoi_tables:
    create_aoi_tables(args)
if args.plot_aoi_on_vid_empty:
    plot_aoi_on_vid_empty(args)
if args.inspect_aoi_movements:
    inspect_aoi_movements(args)    
    
#%% Prepare subject list and analysis

## If processing all participants or one or two group(s), retrieve participant IDs
retrieve_ids(args, 'et')


#%% Plot 1st order figures

if args.analysis_level_1:
    def level_1_et_aoi(subject):
        ## Create the corresponding data object
        data = DataLoader(args, subject, 'et')
        
        if args.recompute_epochs_annotations:
            ## Load et raw data
            et_files, raws = data.create_et_raws()
        
        
    #%% Single subject epoch
    
            metadata, epochs = [None]*len(et_files), [None]*len(et_files)
            events, event_dict = [None]*len(et_files), [None]*len(et_files)
            et_annotations = [None]*len(et_files)
            for i_run, (i_raw, i_et_file) in enumerate(zip(raws, et_files)):
                metadata[i_run] = data.generate_metadata(i_et_file, i_run)
                events[i_run], event_dict[i_run] = data.generate_et_events_annotations(i_raw, metadata[i_run])
                epochs[i_run] = data.epoch_data(i_raw, events[i_run], metadata[i_run], report, 'et')
                et_annotations[i_run] = data.add_epoch_annotations(i_raw, epochs[i_run], report)
            epochs = concatenate_epochs(epochs)
            et_annotations = [epoch_annot for epochs_annot in et_annotations for epoch_annot in epochs_annot]
            et_annotations = [{**d, 'epoch_idx': i} for i, d in enumerate(et_annotations)]
            if args.save_epochs:
                data.save_epochs(args, epochs, 'beh')
                data.save_et_annotations(args, et_annotations)
        else:
            epochs, et_annotations = retrieve_epochs(args, 'beh')
            for i_sub, epo in enumerate(epochs):
                if str(int(epo.metadata['subject'][0])) == subject:
                    epochs, et_annotations = epochs[i_sub], et_annotations[i_sub]
                    break
        if args.plot_aoi_on_vid:
            plot_aoi_on_vid(args, subject, epochs, et_annotations)
        if args.create_aoi_hit_tables:
            create_aoi_hit_tables(args, subject, epochs, et_annotations)
        
    if args.parallelization:
        pool = mp.Pool(mp.cpu_count())
        pool.map(level_1_et_aoi, [subject for subject in args.subjects])
        pool.close()
        pool.join()
    else:
        [level_1_et_aoi(subject) for subject in args.subjects]


#%% Plot 2nd order figures

if args.analysis_level_2:
    epochs, et_annotations = retrieve_epochs(args, 'beh')
    if args.compute_precision_accuracy:
        compute_precision_accuracy(args, args.subjects, epochs, et_annotations)
    if args.detect_smooth_pursuits:
        detect_smooth_pursuits(args, args.subjects, epochs, et_annotations)
    if args.quality_check:
        et_quality_check(args, args.subjects, epochs)
    if args.compute_nb_events:
        compute_nb_events(args, args.subjects, epochs, et_annotations)
    if args.plot_heatmap:
        plot_heatmap(args, args.subjects, epochs, et_annotations)
    if args.plot_aoi_on_vid:
        plot_aoi_on_vid(args, args.subjects, epochs, et_annotations)
    if args.proportion_aoi_hits:
        proportion_aoi_hits(args, args.subjects)
    if args.duration_aoi_hits:
        duration_aoi_hits(args, args.subjects)
    if args.plot_pupil:
        plot_pupil(args, args.subjects, epochs)

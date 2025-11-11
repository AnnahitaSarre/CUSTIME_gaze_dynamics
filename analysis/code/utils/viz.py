#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CUSTIME plot functions
"""


#%% Import modules

import os
import mne
import pickle
import cv2
import numpy as np
import pandas as pd
import mpltern
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import itertools
from scipy import stats
from itertools import product, count
from utils.contrasts import load_contrasts, new_styles
from utils.eyetracking import annot_of_timepoint
from utils.stats import select_stim_epoch, select_stims_for_cond, mean_encoding_feature_versions, find_channel_indices
from utils.cleaning import clean_names, keep_common_videos


#%% Generate ET plots

def plot_aoi_on_vid(args, subject, epochs, et_annotations): # Visualise and save marked video with AOI and gaze displayed
    print(f'\n## Plotting aoi for subject(s) {subject}')
    if args.stimulus == 'words' or args.stimulus == 'sentences':
        video_files = glob(f'utils/stim_videos/videos/{args.stimulus}/*.mp4')
    else:
        video_files = glob(f'utils/stim_videos/videos/*/{args.stimulus}.mp4')
    # If considering group 3, only consider stimuli that were shown to the 3 groups
    if (type(subject) is list and any([sub.startswith('3') for sub in subject])) or (type(subject) is not list and subject.startswith('3')):
       video_files = keep_common_videos(args, subject, video_files)
    for video_file in tqdm(video_files):
        stim_name = os.path.splitext(os.path.basename(video_file))[0]
        print(f'\nMarking stim {stim_name}...')
        ## Create query and fetch corresponding epochs
        stim_epoch_df, _, stim_annot = select_stim_epoch(args, subject, epochs, et_annotations, stim_name)
        if stim_epoch_df is None:
            print(f'Rejecting gaze from subject {subject} for stim {stim_name}: skipping video')
            continue # Entirely pass the video if dealing with a single subject with a rejected epoch
        elif isinstance(stim_epoch_df, pd.DataFrame):
            stim_epoch_df = stim_epoch_df.iloc[::30, :].reset_index(drop=True) # because videos have 30 frames/second
        elif isinstance(stim_epoch_df, list):
            for i_epoch, epoch in enumerate(stim_epoch_df):
                if epoch is not None:
                    stim_epoch_df[i_epoch] = epoch.iloc[::30, :].reset_index(drop=True) # because videos have 30 frames/second
        ## Prepare video marking and aoi parameters
        input_video = cv2.VideoCapture(video_file)
        output_video_dir = os.path.join(args.path2figures, f'aoi_visualisation/{args.stimulus}')
        os.makedirs(output_video_dir, exist_ok=True)
        output_video_fn = f'{stim_name}_sub-{subject}_aoi.avi'
        if len(output_video_fn) >= 255: # Number of characters in file name should be lower or equal to 255
            output_video_fn = f'{stim_name}_sub-{args.group}_aoi.avi'
        output_video = cv2.VideoWriter(os.path.join(output_video_dir, output_video_fn), cv2.VideoWriter_fourcc(*'XVID'), 30.0, (2560,  1600))
        # AOI coordinates
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
        ## Mark the video
        aoi_color = (234, 72, 40)
        aoi_line_thickness = 3
        frame_num = 0
        while input_video.isOpened():
            ret, frame = input_video.read()
            if not ret:
                break
            # Mark hand
            hand_min = (min(df_aoi[x_hand_cols].iloc[frame_num])-40, min(df_aoi[y_hand_cols].iloc[frame_num])-40)
            hand_max = (max(df_aoi[x_hand_cols].iloc[frame_num])+40, max(df_aoi[y_hand_cols].iloc[frame_num])+40)
            if hand_min != (0,0) and hand_max != (0,0): # Exclude situations were the hand was lost and is at (0,0)...
                frame_marked = cv2.rectangle(frame, hand_min, hand_max, aoi_color, aoi_line_thickness)
            else:
                frame_marked = frame
            # Mark eyes
            eyes_pts = []
            for x_eyes_pt_interest, y_eyes_pt_interest in zip(x_eyes_cols, y_eyes_cols):
                eyes_pts.append([df_aoi[x_eyes_pt_interest].iloc[frame_num], df_aoi[y_eyes_pt_interest].iloc[frame_num]])
            eyes_pts = np.array(eyes_pts, dtype=np.int32)
            # frame_marked = cv2.polylines(frame_marked, [eyes_pts], True, aoi_color, aoi_line_thickness)
            # Mark middle of the face (between eyes and lips)
            middle_pts = []
            for x_middle_pt_interest, y_middle_pt_interest in zip(x_middle_cols, y_middle_cols):
                middle_pts.append([df_aoi[x_middle_pt_interest].iloc[frame_num], df_aoi[y_middle_pt_interest].iloc[frame_num]])
            middle_pts = np.array(middle_pts, dtype=np.int32)
            # frame_marked = cv2.polylines(frame_marked, [middle_pts], True, aoi_color, aoi_line_thickness)
            # Mark upper face (addition of the eyes and middle AOIs)
            upperface_pts = np.concatenate([eyes_pts[:3], middle_pts[1:][::-1], np.expand_dims(eyes_pts[0], axis=0)])
            frame_marked = cv2.polylines(frame_marked, [upperface_pts], True, aoi_color, aoi_line_thickness)
            # Mark lips
            lips_pts = []
            for x_lips_pt_interest, y_lips_pt_interest in zip(x_lips_cols, y_lips_cols):
                lips_pts.append([df_aoi[x_lips_pt_interest].iloc[frame_num], df_aoi[y_lips_pt_interest].iloc[frame_num]])
            lips_pts = np.array(lips_pts, dtype=np.int32)
            frame_marked = cv2.polylines(frame_marked, [lips_pts], True, aoi_color, aoi_line_thickness)
            # Mark face
            face_pts = []
            for x_face_pt_interest, y_face_pt_interest in zip(x_face_cols, y_face_cols):
                face_pts.append([df_aoi[x_face_pt_interest].iloc[frame_num], df_aoi[y_face_pt_interest].iloc[frame_num]])
            centroid = np.mean(face_pts, axis=0)
            face_pts = (face_pts - centroid) * 1.3 + centroid # Move each point away from the centroid to enlarge the AOI
            face_pts[:,1] += -55 # Move up all the points
            face_pts[:15,1] += -55 # Move up the upper points to include the foreface and hair
            face_pts = np.array(face_pts, dtype=np.int32)
            frame_marked = cv2.polylines(frame_marked, [face_pts], True, aoi_color, aoi_line_thickness)
            # Mark laterality
            lat_pt1 = (df_aoi['x_face152'].iloc[frame_num], df_aoi['y_face152'].iloc[frame_num])
            lat_pt2 = (df_aoi['x_face10'].iloc[frame_num], df_aoi['y_face10'].iloc[frame_num])
            lat_pt1, lat_pt2 = (lat_pt1 - centroid) * 1.3 + centroid, (lat_pt2 - centroid) * 1.3 + centroid # Apply the same translations as for creating the face AOI
            lat_pt1[1], lat_pt2[1] = lat_pt1[1]-55, lat_pt2[1]-55-55
            direction = (lat_pt2-lat_pt1) / np.linalg.norm(lat_pt2-lat_pt1)  # Normalize direction vector
            extended_pt1, extended_pt2 = np.array(lat_pt1)+2000*direction, np.array(lat_pt2)-2000*direction
            extended_pt1, extended_pt2 = tuple(extended_pt1.astype(int)), tuple(extended_pt2.astype(int))
            frame_marked = cv2.line(frame_marked, extended_pt1, extended_pt2,
                                    aoi_color, aoi_line_thickness)
            # Mark participant(s)' gaze according to the group and current gaze event (normal, fixation, saccade)
            gaze_colors = {'1': {'no_event': (150, 0, 0), 'fixation': (60, 0, 0), 'blink': (300, 300, 300),
                                 'saccade': (255, 0, 0), 'multiple_events': (0, 0, 0)},
                           '2': {'no_event': (0, 0, 150), 'fixation': (0, 0, 60), 'blink': (300, 300, 300),
                                 'saccade': (0, 0, 255), 'multiple_events': (0, 0, 0)},
                           '3': {'no_event': (0, 150, 0), 'fixation': (0, 60, 0), 'blink': (300, 300, 300),
                                 'saccade': (0, 255, 0), 'multiple_events': (0, 0, 0)}}
            if type(subject) is not list: # 1st order analysis
                annot_timepoint = annot_of_timepoint(stim_epoch_df['time'].iloc[frame_num], stim_annot)
                if not(pd.isna(stim_epoch_df['et_x'].iloc[frame_num]) or pd.isna(stim_epoch_df['et_y'].iloc[frame_num])): # Don't plot the subject's gaze if a coordinate is missing in this frame
                    gaze_pt = (int(stim_epoch_df['et_x'].iloc[frame_num]), int(stim_epoch_df['et_y'].iloc[frame_num]))
                    frame_marked = cv2.circle(frame_marked, gaze_pt, 15, gaze_colors[subject[0]][annot_timepoint], -1)
            if type(subject) is list: # 2nd order analysis
                for i_subject, ind_epoch_df in enumerate(stim_epoch_df):
                    if ind_epoch_df is None:
                        continue # Don't plot the subject's gaze if dealing with a single subject with a rejected epoch
                    if pd.isna(ind_epoch_df['et_x'].iloc[frame_num]) or pd.isna(ind_epoch_df['et_y'].iloc[frame_num]):
                        continue # Don't plot the subject's gaze if a coordinate is missing in this frame
                    annot_timepoint = annot_of_timepoint(ind_epoch_df['time'].iloc[frame_num], stim_annot[i_subject])
                    gaze_pt = (int(ind_epoch_df['et_x'].iloc[frame_num]), int(ind_epoch_df['et_y'].iloc[frame_num]))
                    frame_marked = cv2.circle(frame_marked, gaze_pt, 15, gaze_colors[subject[i_subject][0]][annot_timepoint], -1)
            output_video.write(frame_marked)
            frame_num = frame_num + 1
        input_video.release()
        output_video.release()
        cv2.destroyAllWindows()
        

def plot_aoi_on_vid_empty(args): # Visualise and save marked video with AOI and gaze displayed
    if args.stimulus == 'words' or args.stimulus == 'sentences':
        video_files = glob(f'utils/stim_videos/videos/{args.stimulus}/*.mp4')
    else:
        video_files = glob(f'utils/stim_videos/videos/*/{args.stimulus}.mp4')
    for video_file in tqdm(video_files):
        stim_name = os.path.splitext(os.path.basename(video_file))[0]
        print(f'\nMarking stim {stim_name}...')
        ## Prepare video marking and aoi parameters
        input_video = cv2.VideoCapture(video_file)
        output_video_dir = os.path.join(args.path2figures, f'aoi_visualisation/{args.stimulus}')
        os.makedirs(output_video_dir, exist_ok=True)
        output_video_fn = f'{stim_name}_sub-None_aoi.avi'
        if len(output_video_fn) >= 255: # Number of characters in file name should be lower or equal to 255
            output_video_fn = f'{stim_name}_sub-{args.group}_aoi.avi'
        output_video = cv2.VideoWriter(os.path.join(output_video_dir, output_video_fn), cv2.VideoWriter_fourcc(*'XVID'), 30.0, (2560,  1600))
        # AOI coordinates
        df_aoi_file = glob(os.path.join(args.path2videocoords, f'*/{stim_name}_aoi_coordinates.csv'))[0]
        df_aoi = pd.read_csv(os.path.join(df_aoi_file))
        x_hand_cols = df_aoi.filter(regex='x_r_hand*').columns
        y_hand_cols = df_aoi.filter(regex='y_r_hand*').columns
        x_face_cols = df_aoi.filter(regex='x_face*').columns
        y_face_cols = df_aoi.filter(regex='y_face*').columns
        x_eyes_cols, y_eyes_cols = x_face_cols[:4], y_face_cols[:4]
        x_lips_cols, y_lips_cols = x_face_cols[8:12], y_face_cols[8:12]
        x_face_cols, y_face_cols = x_face_cols[12:], y_face_cols[12:]
        x_middle_cols, y_middle_cols = x_eyes_cols[2:].append(x_lips_cols[:2]), y_eyes_cols[2:].append(y_lips_cols[:2])
        ## Mark the video
        aoi_color = (0, 0, 255)
        aoi_line_thickness = 4
        frame_num = 0
        while input_video.isOpened():
            ret, frame = input_video.read()
            if not ret:
                break
            # Mark hand
            hand_min = (min(df_aoi[x_hand_cols].iloc[frame_num])-40, min(df_aoi[y_hand_cols].iloc[frame_num])-40)
            hand_max = (max(df_aoi[x_hand_cols].iloc[frame_num])+40, max(df_aoi[y_hand_cols].iloc[frame_num])+40)
            if hand_min != (0,0) and hand_max != (0,0): # Exclude situations were the hand was lost and is at (0,0)...
                frame_marked = cv2.rectangle(frame, hand_min, hand_max, aoi_color, aoi_line_thickness)
            else:
                frame_marked = frame
            # Mark face
            face_pts = []
            for x_face_pt_interest, y_face_pt_interest in zip(x_face_cols, y_face_cols):
                face_pts.append([df_aoi[x_face_pt_interest].iloc[frame_num], df_aoi[y_face_pt_interest].iloc[frame_num]])
            centroid = np.mean(face_pts, axis=0)
            face_pts = (face_pts - centroid) * 1.3 + centroid # Move each point away from the centroid to enlarge the AOI
            face_pts[:,1] += -55 # Move up all the points
            face_pts[:15,1] += -55 # Move up the upper points to include the foreface and hair
            face_pts = np.array(face_pts, dtype=np.int32)
            frame_marked = cv2.polylines(frame_marked, [face_pts], True, aoi_color, aoi_line_thickness)
            frame_marked = cv2.fillPoly(frame_marked, [face_pts], color=(120,152,164))
            # Mark eyes
            eyes_pts = []
            for x_eyes_pt_interest, y_eyes_pt_interest in zip(x_eyes_cols, y_eyes_cols):
                eyes_pts.append([df_aoi[x_eyes_pt_interest].iloc[frame_num], df_aoi[y_eyes_pt_interest].iloc[frame_num]])
            eyes_pts = np.array(eyes_pts, dtype=np.int32)
            # frame_marked = cv2.polylines(frame_marked, [eyes_pts], True, aoi_color, aoi_line_thickness)
            # Mark middle of the face (between eyes and lips)
            middle_pts = []
            for x_middle_pt_interest, y_middle_pt_interest in zip(x_middle_cols, y_middle_cols):
                middle_pts.append([df_aoi[x_middle_pt_interest].iloc[frame_num], df_aoi[y_middle_pt_interest].iloc[frame_num]])
            middle_pts = np.array(middle_pts, dtype=np.int32)
            # frame_marked = cv2.polylines(frame_marked, [middle_pts], True, aoi_color, aoi_line_thickness)
            # Mark upper face (addition of the eyes and middle AOIs)
            upperface_pts = np.concatenate([eyes_pts[:3], middle_pts[1:][::-1], np.expand_dims(eyes_pts[0], axis=0)])
            frame_marked = cv2.polylines(frame_marked, [upperface_pts], True, aoi_color, aoi_line_thickness)
            # Mark lips
            lips_pts = []
            for x_lips_pt_interest, y_lips_pt_interest in zip(x_lips_cols, y_lips_cols):
                lips_pts.append([df_aoi[x_lips_pt_interest].iloc[frame_num], df_aoi[y_lips_pt_interest].iloc[frame_num]])
            lips_pts = np.array(lips_pts, dtype=np.int32)
            frame_marked = cv2.polylines(frame_marked, [lips_pts], True, aoi_color, aoi_line_thickness)
            # Mark laterality
            lat_pt1 = (df_aoi['x_face152'].iloc[frame_num], df_aoi['y_face152'].iloc[frame_num])
            lat_pt2 = (df_aoi['x_face10'].iloc[frame_num], df_aoi['y_face10'].iloc[frame_num])
            lat_pt1, lat_pt2 = (lat_pt1 - centroid) * 1.3 + centroid, (lat_pt2 - centroid) * 1.3 + centroid # Apply the same translations as for creating the face AOI
            lat_pt1[1], lat_pt2[1] = lat_pt1[1]-55, lat_pt2[1]-55-55
            direction = (lat_pt2-lat_pt1) / np.linalg.norm(lat_pt2-lat_pt1)  # Normalize direction vector
            extended_pt1, extended_pt2 = np.array(lat_pt1)+2000*direction, np.array(lat_pt2)-2000*direction
            extended_pt1, extended_pt2 = tuple(extended_pt1.astype(int)), tuple(extended_pt2.astype(int))
            frame_marked = cv2.line(frame_marked, extended_pt1, extended_pt2,
                                    aoi_color, aoi_line_thickness)
            output_video.write(frame_marked)
            frame_num = frame_num + 1
        input_video.release()
        output_video.release()
        cv2.destroyAllWindows()
        
        
def plot_heatmap(args, subject, epochs, et_annotations):
    print(f'\n## Plotting heatmap for subject(s) {subject}')
    video_files = glob(f'utils/stim_videos/videos/{args.stimulus}/*.mp4')
    background_image = cv2.imread(f'utils/background_{args.stimulus}.jpg')
    background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
    height, width, _ = background_image.shape
    white_image = np.ones((height, width, 3), dtype=np.uint8) * 255
    background_image = cv2.addWeighted(background_image, 0.2, white_image, 1 - 0.2, 0) # Fade background image
    # If considering group 3, only consider stimuli that were shown to the 3 groups
    if (type(subject) is list and any([sub.startswith('3') for sub in subject])) or (type(subject) is not list and subject.startswith('3')):
       video_files = keep_common_videos(args, subject, video_files)
    coords_df = []
    for video_file in tqdm(video_files):
        stim_name = os.path.splitext(os.path.basename(video_file))[0]
        print(f'\nDetermining aoi hits for stim {stim_name}')
        # Fetch AOI coordinates and compute height and width for all frames
        df_aoi_file = glob(os.path.join(args.path2videocoords, f'*/{stim_name}_aoi_coordinates.csv'))[0]
        df_aoi = pd.read_csv(os.path.join(df_aoi_file))
        df_aoi['mouth_width'] = np.sqrt(abs(df_aoi['x_face425']-df_aoi['x_face205'])**2 + abs(df_aoi['y_face425']-df_aoi['y_face205'])**2)
        df_aoi['mouth_height'] = np.sqrt(abs(df_aoi['x_face205']-df_aoi['x_face169'])**2 + abs(df_aoi['y_face205']-df_aoi['y_face169'])**2)
        ## Create query and fetch corresponding epochs
        stim_epoch_df, _, stim_annot = select_stim_epoch(args, subject, epochs, et_annotations, stim_name)
        df_aoi = df_aoi.loc[df_aoi.index.repeat(33.333333333)].reset_index(drop=True) # Replicate AOI lines to feat the number of gaze data points
        if stim_epoch_df is None:
            print(f'Rejecting gaze from subject {subject} for stim {stim_name}: skipping stimulus')
            continue # Entirely pass the stimulus if dealing with a single subject with a rejected epoch
        else:
            for i_sub_df, sub_df in enumerate(stim_epoch_df):
                if sub_df is None:
                    continue
                else:
                    sub_df['annotation'] = [annot_of_timepoint(timepoint, stim_annot[i_sub_df]) for timepoint in sub_df['time']]
                    sub_df = sub_df[:len(df_aoi)]
            stim_epoch_df = pd.concat(stim_epoch_df)
            stim_epoch_df = stim_epoch_df.reset_index()
        coords_df.append(stim_epoch_df)
    coords_df = pd.concat(coords_df).dropna()
    
    for event_type in ['fixation','saccade','all']:
        if event_type == 'all':
            coords_event_df = coords_df
        else:
            coords_event_df = coords_df[coords_df['annotation'] == event_type]
        # list_event_coords = list(zip(coords_event_df['et_x_relative'].astype('int64'), coords_event_df['et_y_relative'].astype('int64')))
        list_event_coords = list(zip(coords_event_df['et_x'].astype('int64'), coords_event_df['et_y'].astype('int64')))
        heatmap = np.zeros((height, width), dtype=np.float32)
        for (x, y) in list_event_coords:
            if 0 < x < width and 0 < y < height:
                heatmap[y, x] += 1
        heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
        heatmap_normalized = cv2.normalize(heatmap, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        heatmap_colored = cv2.applyColorMap((heatmap_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        combined_image = cv2.addWeighted(background_image, 1 - 0.5, heatmap_colored, 0.5, 0)
        # Display the combined image
        fig = plt.figure(figsize=(10, 8))
        plt.imshow(combined_image)
        plt.axis('off')
        plt.title(f'Gaze heatmap for subject(s) {args.group} in {args.stimulus} ({event_type})')
        plt.show()
        fig.savefig(os.path.join(args.path2figures,
                    f'heatmaps/heatmap_stim-{args.stimulus}_sub-{args.group}_type-{event_type}.png'))
                    
    
def plot_quality_check(args, qc_overall, qc_participants, qc_stim):
    # Overall analysis plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    qc_overall.set_index('Overall Metric')['Value'].plot(kind='bar', ax=ax)
    ax.set_title(f'Overall Stimuli Rejection Rate (stim-{args.stimulus}, sub-{args.group}) \n Rejection criteria: > {args.loss_rate_criteria}% of samples lost')
    ax.set_ylabel('Value (%)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(args.path2figures, f'et_quality_check/qc_overall_stim-{args.stimulus}_sub-{args.group}_loss-{args.loss_rate_criteria}.png'))
    # Participants analysis plotting
    fig, ax = plt.subplots(figsize=(14, 6))
    qc_participants.plot(ax=ax, x='subject', y=['rejection_rate'], kind='bar', width=0.8)
    ax.set_title(f'Stimuli Rejection Rate by Participant (stim-{args.stimulus}, sub-{args.group}) \n Rejection criteria: > {args.loss_rate_criteria}% of samples lost')
    ax.set_ylabel('Mean Rejection Rate (%)')
    ax.set_xlabel('Subject ID')
    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(args.path2figures, f'et_quality_check/qc_participants_stim-{args.stimulus}_sub-{args.group}_loss-{args.loss_rate_criteria}.png'))
    # Stimuli analysis plotting
    fig, ax = plt.subplots(figsize=(14, 6))
    qc_stim.set_index('stimulus')[['rejection_rate']].plot(ax=ax, kind='bar', width=0.8)
    ax.set_title(f'Participants Rejection Rate by Stimulus (stim-{args.stimulus}, sub-{args.group}) \n Rejection criteria: > {args.loss_rate_criteria}% of samples lost')
    ax.set_ylabel('Mean Rejection Rate (%)')
    ax.set_xlabel('Stimulus')
    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(args.path2figures, f'et_quality_check/qc_stimuli_stim-{args.stimulus}_sub-{args.group}_loss-{args.loss_rate_criteria}.png'))
    
    
def plot_proportion_aoi_hits_boxplots(args, aoi_hit_df):
    for cond in args.cond_dict[args.stimulus]:
        sfx = '_last_word' if 'predictable' in cond else ''
        df_cond = select_stims_for_cond(args, aoi_hit_df, cond)
        df_cond = df_cond[df_cond['subject'] != '12'] # Exclude the outlier
        # df_cond.loc[df_cond['group'].isin(['1', '2']), 'group'] = '1+2' # Group CS users
        groups_list = list(df_cond['group'].unique())
        groups_list.reverse() # To display the bars in the increasing order
        group_dict = {'1':'Deaf users', '2':'Hearing users', '1+2':'CS users', '3':'Controls'}
        figsize = (5, 12)
        figsize_mismatch = (14,12)
        df_cond['group'] = df_cond['group'].replace(group_dict)
        fig_path = os.path.join(args.path2figures,'proportion_aoi_hits')
        cond_list = ['freq','mismatch','pseudo'] if args.stimulus == 'words' else ['predictable']
        df_cond_with_cond = round(df_cond.groupby(['subject','group']+cond_list, as_index=False).mean(numeric_only=True),2)
        df_cond_without_cond = round(df_cond.groupby(['subject','group'], as_index=False).mean(numeric_only=True),2)
        ## Repartition of face hits
        plt.figure(figsize=figsize)
        sns.boxplot(data=df_cond_without_cond, y=f'proportion_hits_face_lips_no_other{sfx}', hue='group', widths=0.5, legend=False, palette='pastel')
        sns.stripplot(data=df_cond_without_cond, y=f'proportion_hits_face_lips_no_other{sfx}', hue='group', dodge=True, legend=False, palette='dark')
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylim(-3, 103)
        plt.axhline(y=50, color='grey', linestyle = ':')
        plt.xlabel('Group', fontsize=18)
        plt.ylabel('Proportion of lips fixation (%)', fontsize=20)
        plt.title(f'Proportion of lips fixations among face fixations in {cond}', fontsize=14, pad=12)
        plt.savefig(os.path.join(fig_path, f'repartion_face_stim-{args.stimulus}_cond-{cond}_gr-{groups_list}.png'))
        ## Left-right asymmetry
        aoi_list = ['face','lips','upper_face']
        for aoi in aoi_list:
            plt.figure(figsize=figsize)
            sns.boxplot(data=df_cond_without_cond, y=f'proportion_lat_{aoi}_left{sfx}', hue='group', widths=0.5, legend=False, palette='pastel')
            sns.stripplot(data=df_cond_without_cond, y=f'proportion_lat_{aoi}_left{sfx}', hue='group', dodge=True, legend=False, palette='dark')
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.ylim(-3, 103)
            plt.axhline(y=50, color='grey', linestyle = ':') 
            plt.xlabel('Group', fontsize=18)
            plt.ylabel('Proportion of left fixation (%)', fontsize=20)
            plt.title(f'Fixation asymmetry within the {aoi} region in {cond}', fontsize=14, pad=12)
            plt.savefig(os.path.join(fig_path, f'asymmetry_{aoi}_stim-{args.stimulus}_cond-{cond}_gr-{groups_list}.png'))
            if args.stimulus == 'words':
                plt.figure(figsize=figsize_mismatch)
                sns.boxplot(data=df_cond_with_cond, x='mismatch', y=f'proportion_lat_{aoi}_left{sfx}', hue='group', widths=0.5, legend=False, palette='pastel')
                sns.stripplot(data=df_cond_with_cond, x='mismatch', y=f'proportion_lat_{aoi}_left{sfx}', hue='group', dodge=True, legend=False, palette='dark')
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
                plt.ylim(-3, 103)
                plt.axhline(y=50, color='grey', linestyle = ':') 
                plt.xlabel('Group', fontsize=18)
                plt.ylabel('Proportion of left fixation (%)', fontsize=20)
                plt.title(f'Fixation asymmetry within the {aoi} region in {cond}', fontsize=14, pad=12)
                plt.savefig(os.path.join(fig_path, f'asymmetry_{aoi}_mismatch_stim-{args.stimulus}_cond-{cond}_gr-{groups_list}.png'))
        
    
def plot_proportion_aoi_hits_bars(args, aoi_hit_df):
    for cond in args.cond_dict[args.stimulus]:
        sfx = '_last_word' if 'predictable' in cond else ''
        df_cond = select_stims_for_cond(args, aoi_hit_df, cond)
        df_cond = df_cond[df_cond['subject'] != '12'] # Exclude the outlier
        # df_cond.loc[df_cond['group'].isin(['1', '2']), 'group'] = '1+2' # Group CS users
        groups_list = list(df_cond['group'].unique())
        groups_list.reverse() # To display the bars in the increasing order
        group_dict = {'1':'Deaf', '2':'Hearing', '1+2':'CS users', '3':'Control'}
        nb_groups = len(groups_list)
        fig_path = os.path.join(args.path2figures,'proportion_aoi_hits_bars')
        df_cond = round(df_cond.groupby('group', as_index=False).mean(numeric_only=True),2)
        ## Figure for repartition among aois
        fig, ax = plt.subplots(2, 1, figsize=(7, 1.5*nb_groups), sharex=True,
                               gridspec_kw={'hspace': 0.2}, constrained_layout=True)
        ticks_list = [*range(0, 101, 10)]
        bar_height = 0.8
        handles, labels = [], []
        cmap = plt.get_cmap('Pastel2')
        # cmap = colors.ListedColormap(['#060a79', '#555bf6', # Poster color version
        #                               '#060a79', '#555bf6', '#0b13da', '#b6b9fb'])
        # General repartition of hits
        for idx, group in enumerate(groups_list):
            df_cond_group = df_cond.loc[int(group)]
            group_name = group_dict[group]
            # Calculate normalization factors and modify group means accordingly to reach 100% in the following plot
            total_hits_all = (df_cond_group[f'proportion_hits_all_face{sfx}'] + df_cond_group[f'proportion_hits_all_hand{sfx}'])
            df_cond_group[f'proportion_hits_all_face{sfx}'] = (df_cond_group[f'proportion_hits_all_face{sfx}']/total_hits_all)*100
            df_cond_group[f'proportion_hits_all_hand{sfx}'] = (df_cond_group[f'proportion_hits_all_hand{sfx}']/total_hits_all)*100
            ax[0].barh([group_name], df_cond_group[f'proportion_hits_all_face{sfx}'], height=bar_height, color=cmap(0), label='Face')
            ax[0].barh([group_name], df_cond_group[f'proportion_hits_all_hand{sfx}'], height=bar_height, left=df_cond_group[f'proportion_hits_all_face{sfx}'], color=cmap(1), label='Hand')
            ax[0].set_xlim(0, 100)
        ax[0].set_title('Repartition of fixations between the hand and the face', fontsize=13)
        handles = handles + ax[0].get_legend_handles_labels()[0]
        labels = labels + ax[0].get_legend_handles_labels()[1]
        # Repartition of face hits
        for idx, group in enumerate(groups_list):
            df_cond_group = df_cond.loc[int(group)]
            group_name = group_dict[group]
            # Normalize to reach 100% in the following plot
            total_hits_face = (df_cond_group[f'proportion_hits_face_upper_face{sfx}'] + df_cond_group[f'proportion_hits_face_lips{sfx}'] + df_cond_group[f'proportion_hits_face_other{sfx}'])
            df_cond_group[f'proportion_hits_face_upper_face{sfx}'] = (df_cond_group[f'proportion_hits_face_upper_face{sfx}']/total_hits_face)*100
            df_cond_group[f'proportion_hits_face_lips{sfx}'] = (df_cond_group[f'proportion_hits_face_lips{sfx}']/total_hits_face)*100
            df_cond_group[f'proportion_hits_face_other{sfx}'] = (df_cond_group[f'proportion_hits_face_other{sfx}']/total_hits_face)*100
            ax[1].barh([group_name], df_cond_group[f'proportion_hits_face_upper_face{sfx}'], height=bar_height, color=cmap(2), label='Upper face')
            ax[1].barh([group_name], df_cond_group[f'proportion_hits_face_lips{sfx}'], left=df_cond_group[f'proportion_hits_face_upper_face{sfx}'], height=bar_height, color=cmap(3), label='Lips')
            ax[1].barh([group_name], df_cond_group[f'proportion_hits_face_other{sfx}'], left=df_cond_group[f'proportion_hits_face_upper_face{sfx}'] + df_cond_group[f'proportion_hits_face_lips{sfx}'], height=bar_height, color=cmap(5), label='Other')
        ax[1].set_xlabel('Percentage')
        ax[1].set_title('Repartition of fixations within the face', fontsize=13)
        handles = handles + ax[1].get_legend_handles_labels()[0]
        labels = labels + ax[1].get_legend_handles_labels()[1]
        # Arrange main figure
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(), fontsize=12, loc='lower right', bbox_to_anchor=(0.9,-0.15), ncol=len(labels), bbox_transform=fig.transFigure)
        fig.suptitle(f'Repartition of fixations (cond: {cond}; groups {groups_list})', fontsize=16)
        plt.xticks(ticks=ticks_list)
        plt.show()
        fig.savefig(os.path.join(fig_path, f'repartion_face_hand_stim-{args.stimulus}_cond-{cond}_gr-{groups_list}.png'))
        ## Figure for laterality
        fig, ax = plt.subplots(3, 1, figsize=(8, 2*nb_groups), sharex=True,
                               gridspec_kw={'hspace': 0.2}, constrained_layout=True)
        handles, labels = [], []
        # Laterality of face hits
        for idx, group in enumerate(groups_list):
            df_cond_group = df_cond.loc[int(group)]
            group_name = group_dict[group]
            ax[0].barh([group_name], df_cond_group[f'proportion_lat_face_left{sfx}'], height=bar_height, color=cmap(1), label='Left')
            ax[0].barh([group_name], df_cond_group[f'proportion_lat_face_right{sfx}'], height=bar_height, left=df_cond_group[f'proportion_lat_face_left{sfx}'], color=cmap(0), label='Right')
            ax[0].set_xlim(0, 100)
        ax[0].set_title('Fixation asymmetry on the whole face', fontsize=13)
        handles = handles + ax[0].get_legend_handles_labels()[0]
        labels = labels + ax[0].get_legend_handles_labels()[1]
        # Laterality of upper face hits
        for idx, group in enumerate(groups_list):
            df_cond_group = df_cond.loc[int(group)]
            group_name = group_dict[group]
            ax[1].barh([group_name], df_cond_group[f'proportion_lat_upper_face_left{sfx}'], height=bar_height, color=cmap(1), label='Left')
            ax[1].barh([group_name], df_cond_group[f'proportion_lat_upper_face_right{sfx}'], height=bar_height, left=df_cond_group[f'proportion_lat_upper_face_left{sfx}'], color=cmap(0), label='Right')
            ax[1].set_xlim(0, 100)
        ax[1].set_title('Fixation asymmetry on the upper face', fontsize=13)
        handles = handles + ax[1].get_legend_handles_labels()[0]
        labels = labels + ax[1].get_legend_handles_labels()[1]
        # Laterality of lips hits
        for idx, group in enumerate(groups_list):
            df_cond_group = df_cond.loc[int(group)]
            group_name = group_dict[group]
            ax[2].barh([group_name], df_cond_group[f'proportion_lat_lips_left{sfx}'], height=bar_height, color=cmap(1), label='Left')
            ax[2].barh([group_name], df_cond_group[f'proportion_lat_lips_right{sfx}'], height=bar_height, left=df_cond_group[f'proportion_lat_lips_left{sfx}'], color=cmap(0), label='Right')
            ax[2].set_xlim(0, 100)
        ax[2].set_xlabel('Percentage')
        ax[2].set_title('Fixation asymmetry on the lips', fontsize=13)
        handles = handles + ax[2].get_legend_handles_labels()[0]
        labels = labels + ax[2].get_legend_handles_labels()[1]
        # Arrange main figure
        by_label = dict(zip(labels, handles))
        plt.xticks(ticks=ticks_list)
        fig.legend(by_label.values(), by_label.keys(), fontsize=12, loc='lower right', bbox_to_anchor=(0.9,-0.15), ncol=len(labels), bbox_transform=fig.transFigure)
        fig.suptitle(f'Fixation asymmetry in {args.stimulus} (cond: {cond}; groups {groups_list})', fontsize=16)
        fig.savefig(os.path.join(fig_path, f'asymmetry_stim-{args.stimulus}_cond-{cond}_gr-{groups_list}.png'))
        plt.show()
        
        
def plot_proportion_correlation(args, proportion_hits_df):
    df_cov = pd.read_csv(os.path.join(args.path2deriv,'behavioral/custime_covariates.csv'))
    df_cov['subject'] = df_cov['subject'].astype('str')
    df_corr = proportion_hits_df.merge(df_cov, on='subject')
    aoi_list = ['proportion_hits_face_lips_no_other','proportion_lat_face_left',
                'proportion_lat_lips_left','proportion_lat_upper_face_left']
    cov_list = ['sentences_all_scores','lpc_age']
    df_corr = df_corr.groupby('subject').median()
    for cov, aoi in product(cov_list,aoi_list):
        df_corr_aoi = df_corr[df_corr[aoi].notna()]
        label = f'{aoi} x {cov}'
        plt.plot(df_corr_aoi[cov], df_corr_aoi[aoi], marker='o', linestyle='none', markeredgecolor='black', label=label)
        slope, intercept = np.polyfit(df_corr_aoi[cov], df_corr_aoi[aoi], 1)
        regression_line = slope * df_corr_aoi[cov] + intercept
        plt.plot(df_corr_aoi[cov], regression_line, linewidth=2)
        plt.plot(df_corr_aoi[cov], regression_line, linewidth=0.5, color='black')
        plt.xlabel(f"{cov}", fontsize=15)
        plt.ylabel(f'{aoi}', fontsize=15)
        plt.legend(bbox_to_anchor=(0.5, -0.15), fontsize=15, title_fontsize=15, loc='upper center', borderaxespad=1.5)
        plt.show()
        
        
def plot_nb_events(args, nb_events_df):
    groups = list(set([gr[0] for gr in args.subjects]))
    hue_order = [gr for gr in groups]
    hue_order.sort()
    for cond in args.cond_dict[args.stimulus]:
        fig = plt.figure(figsize=(8,10))
        df_cond = select_stims_for_cond(args, nb_events_df, cond)
        plot_df = df_cond.groupby(['group','subject']).agg(nb_events=('nb_events','mean')).reset_index()
        sns.boxplot(y='nb_events', hue='group', hue_order=hue_order,
                    data=plot_df, palette='pastel')
        sns.stripplot(y='nb_events', hue='group', hue_order=hue_order, dodge=True,
                      data=plot_df, palette='dark', legend=False)
        plt.ylabel('Mean number of events per stimulus', fontsize=16)
        plt.title(f'Number of fixations and saccades in condition {cond}', fontsize=20, pad=30)
        handles, labels = fig.gca().get_legend_handles_labels()
        plt.legend(handles[:3], labels[:3], title='Group')
        fig.savefig(os.path.join(args.path2figures,f'nb_events/{args.stimulus}_{cond}_nb_events.png'), bbox_inches='tight')


def plot_duration_aoi_hits(args, aoi_hit_df_melted):
    for cond in args.cond_dict[args.stimulus]:
        df_cond_melted = select_stims_for_cond(args, aoi_hit_df_melted, cond)
        df_cond_melted = df_cond_melted[df_cond_melted['subject']!='12'] # Exclude outlier s12 from plotting
        hue_order = ['1', '2', '3'] if any([gr == '3' for gr in df_cond_melted['group']]) else ['1','2']
        fig = plt.figure(figsize=(8,10))
        sns.boxplot(x='hit_type', y='duration', hue='group', hue_order=hue_order,
                    data=df_cond_melted, palette='pastel')
        sns.stripplot(x='hit_type', y='duration', hue='group', hue_order=hue_order, dodge=True,
                      data=df_cond_melted, palette='dark')
        plt.xlabel('AOI', fontsize=16)
        plt.ylabel('Duration of fixation', fontsize=16)
        plt.title(f'Duration of hits for each AOI (ms) \nCondition: {cond}', fontsize=25, pad=30)
        handles, labels = fig.gca().get_legend_handles_labels()
        plt.legend(handles[:3], labels[:3], title='Group')
        fig.savefig(os.path.join(args.path2figures,f'duration_aoi_hits/{args.stimulus}_duration_{cond}.png'), bbox_inches='tight')
    

def plot_pupil(args, subjects, epochs):
    print(f'\n## Ploting pupil data for subjects {subjects}')
    if args.stimulus == 'words' or args.stimulus == 'sentences':
        video_files = glob(f'utils/stim_videos/videos/{args.stimulus}/*.mp4')
    else:
        video_files = glob(f'utils/stim_videos/videos/*/{args.stimulus}.mp4')
    # If considering group 3, only consider stimuli that were shown to the 3 groups
    if any([sub.startswith('3') for sub in subjects]):
       video_files = keep_common_videos(args, subjects, video_files)
    qc_df = pd.DataFrame()
    for video_file in tqdm(video_files):
        stim_name = os.path.splitext(os.path.basename(video_file))[0]
        print(f'\nAnalysing stimulus {stim_name}...')
        ## Create query for each stim*participant
        stim_epoch_df, qc_df_stim = select_stim_epoch(args, subjects, epochs, None, stim_name)
        qc_df = qc_df.concat(qc_df_stim, ignore_index=True)
        def mean_of_list_ignore_nan(lst):
            if isinstance(lst, list):
                valid_items = [item for item in lst if not np.isnan(item)]
                if valid_items:  # Check if the list is not empty after removing np.nan
                    return np.mean(valid_items)
            return np.nan
        def std_of_list_ignore_nan(lst):
            if isinstance(lst, list):
                valid_items = [item for item in lst if not np.isnan(item)]
                if valid_items:  # Check if the list is not empty after removing np.nan
                    return np.std(valid_items)
            return np.nan
        qc_df['et_pupil_mean'] = qc_df['et_pupil'].apply(mean_of_list_ignore_nan)
        qc_df['et_pupil_std'] = qc_df['et_pupil'].apply(std_of_list_ignore_nan)
    pupil_participants = qc_df.groupby('subject').agg(mean_pupil_area=('et_pupil_mean', 'mean'), mean_std_pupil_area=('et_pupil_std','mean')).reset_index()
    pupil_participants.to_csv(os.path.join(args.path2deriv,f'eye-tracking/pupil/pupil_mean_std_participants_stim-{args.stimulus}_sub-{args.group}.csv'))
    pupil_stim = qc_df.groupby('stimulus').agg(mean_pupil_area=('et_pupil_mean', 'mean'), mean_std_pupil_area=('et_pupil_std','mean')).reset_index()
    pupil_stim.to_csv(os.path.join(args.path2deriv,f'eye-tracking/pupil/pupil_mean_std_stimuli_stim-{args.stimulus}_sub-{args.group}.csv'))
    ## Plot results
    # Participants mean pupil area plotting
    fig, ax = plt.subplots(figsize=(14, 6))
    pupil_participants.plot(ax=ax, x='subject', y=['mean_pupil_area'], kind='bar', width=0.8)
    ax.set_title(f'Mean Pupil Area by Participant (stim-{args.stimulus}, sub-{args.group})')
    ax.set_ylabel('Mean Pupil Area')
    ax.set_xlabel('Subject ID')
    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(args.path2figures, f'et_pupil/mean_pupil_participants_stim-{args.stimulus}_sub-{args.group}.png'))
    # Stimuli mean pupil area plotting
    fig, ax = plt.subplots(figsize=(14, 6))
    pupil_stim.plot(ax=ax, x='stimulus', y=['mean_pupil_area'], kind='bar', width=0.8)
    ax.set_title(f'Mean Pupil Area by Stimulus (stim-{args.stimulus}, sub-{args.group})')
    ax.set_ylabel('Mean Pupil Area')
    ax.set_xlabel('Stimulus')
    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(args.path2figures, f'et_pupil/mean_pupil_stimuli_stim-{args.stimulus}_sub-{args.group}.png'))
    # Participants mean std pupil area plotting
    fig, ax = plt.subplots(figsize=(14, 6))
    pupil_participants.plot(ax=ax, x='subject', y=['mean_std_pupil_area'], kind='bar', width=0.8)
    ax.set_title(f'Mean Pupil Area STD by Participant (stim-{args.stimulus}, sub-{args.group})')
    ax.set_ylabel('Mean Pupil Area STD')
    ax.set_xlabel('Subject ID')
    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(args.path2figures, f'et_pupil/std_pupil_participants_stim-{args.stimulus}_sub-{args.group}.png'))
    # Stimuli mean std pupil area plotting
    fig, ax = plt.subplots(figsize=(14, 6))
    pupil_stim.plot(ax=ax, x='stimulus', y=['mean_std_pupil_area'], kind='bar', width=0.8)
    ax.set_title(f'Mean Pupil Area STD by Stimulus (stim-{args.stimulus}, sub-{args.group})')
    ax.set_ylabel('Mean Pupil Area STD')
    ax.set_xlabel('Stimulus')
    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(args.path2figures, f'et_pupil/std_pupil_stimuli_stim-{args.stimulus}_sub-{args.group}.png'))
    
    
#%% Generate behavioral plots

def plot_group_difference(args, behav_df, metric):
    metric_info = behav_df.groupby(['subject', 'group'])[metric].mean().reset_index()
    sns.boxplot(x=metric_info['group'], y=metric_info[metric], data=metric_info,
                palette='pastel').set(title=f'{metric} by group (stim-{args.stimulus})')
    sns.stripplot(x=metric_info['group'], y=metric_info[metric], palette='dark')
    plt.show()


def plot_validity_difference(args, behav_df, metric):
    if not behav_df[(behav_df['response_type'] == 'CR') | (behav_df['response_type'] == 'MISS')].empty:
        behav_df.drop(behav_df[(behav_df['response_type'] == 'CR') | (behav_df['response_type'] == 'MISS')].index, inplace=True)
    metric_info = behav_df.groupby(['subject', 'group', 'response_type'])[metric].mean().reset_index()
    plt.figure(figsize=(12,10))
    sns.boxplot(x='response_type', y=metric, hue='group', data=metric_info,  palette='pastel', legend=False)
    sns.stripplot(x='response_type', y=metric, hue='group', dodge=True, data=metric_info, palette='dark', legend=False)
    new_labels = ['FA', 'HIT']
    plt.xticks(ticks=plt.gca().get_xticks(), labels=new_labels, fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Response type', fontsize=15, labelpad=11)
    plt.ylabel(f'Mean {metric}', fontsize=15, labelpad=10)
    plt.title(f'{metric} by response type (all words)', fontsize= 20, y=1.025)
    plt.show()
        
    
def plot_behav_word(args, behav_df, metric):
    # Plot mismatch in all words
    metric_info = behav_df.groupby(['subject', 'group', 'mismatch'])[metric].mean().reset_index()
    plt.figure(figsize=(12,10))
    sns.boxplot(x='mismatch', y=metric, hue='group', data=metric_info,  palette='pastel', legend=False)
    sns.stripplot(x='mismatch', y=metric, hue='group', dodge=True, data=metric_info, palette='dark', legend=False)
    new_labels = ['0', '1', '2']
    plt.xticks(ticks=plt.gca().get_xticks(), labels=new_labels, fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Number of mismatch', fontsize=15, labelpad=11)
    plt.ylabel(f'Mean {metric}', fontsize=15, labelpad=10)
    plt.title(f'{metric} by mismatch (all words)', fontsize= 20, y=1.025)
    plt.show()
    # Plot lexicality in all words
    metric_info = behav_df.groupby(['subject', 'group', 'pseudo'])[metric].mean().reset_index()
    plt.figure(figsize=(12,10))
    sns.boxplot(x='pseudo', y=metric, hue='group', data=metric_info,  palette='pastel', legend=False)
    sns.stripplot(x='pseudo', y=metric, hue='group', dodge=True, data=metric_info, palette='dark', legend=False)
    new_labels = ['0', '1']
    plt.xticks(ticks=plt.gca().get_xticks(), labels=new_labels, fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Lexicality', fontsize=15, labelpad=11)
    plt.ylabel(f'Mean {metric}', fontsize=15, labelpad=10)
    plt.title(f'{metric} by lexicality (all words)', fontsize= 20, y=1.025)
    plt.show()
    # Plot freq in real words
    behav_nonpseudo_df = behav_df[behav_df['pseudo']=='0']
    metric_nonpseudo_info = behav_nonpseudo_df.groupby(['subject', 'group', 'freq'])[metric].mean().reset_index()
    plt.figure(figsize=(12,10))
    sns.boxplot(x='freq', y=metric, hue='group', data=metric_nonpseudo_info,  palette='pastel', legend=False)
    sns.stripplot(x='freq', y=metric, hue='group', dodge=True, data=metric_nonpseudo_info, palette='dark', legend=False)
    new_labels = ['high', 'low']
    plt.xticks(ticks=plt.gca().get_xticks(), labels=new_labels, fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Real word frequency', fontsize=15, labelpad=11)
    plt.ylabel(f'Mean {metric}', fontsize=15, labelpad=10)
    plt.title(f'{metric} by frequency (real words)', fontsize= 20, y=1.025)
    plt.show()
        

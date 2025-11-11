#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUSTIME eye-tracking functions for the et_plot script
"""

#%% Import modules

import os
import pandas as pd
from glob import glob
from tqdm import tqdm


#%% Create aoi coordinates tables for stim out of the original tables
# To run only once unless the coordinates for aois are to be changed

def create_aoi_tables(args):
    print(f'\n## Creating aoi tables for {args.stimulus}')
    video_coords_files = glob(os.path.join(args.path2videocoords, f'{args.stimulus}/originals/*_coordinates.csv'))
    df_aoi_columns = ['fn_video', 'frame_number',
                      #  Eyes AOI coordinates
                      'x_face71', 'y_face71',
                      'x_face301', 'y_face301',
                      'x_face345', 'y_face345',
                      'x_face116', 'y_face116',
                      # Nose AOI coordinates
                      'x_face114', 'y_face114',
                      'x_face343', 'y_face343',
                      'x_face327', 'y_face327',
                      'x_face98', 'y_face98',
                      # Lips AOI coordinates
                      'x_face205', 'y_face205',
                      'x_face425', 'y_face425',
                      'x_face394', 'y_face394',
                      'x_face169', 'y_face169']
                      # Head contour x AOI coordinates
    df_aoi_columns += [f'x_face{num}' for num in [127, 162, 21, 54, 103, 67,
                                                  109, 10, 338, 297, 332, 284,
                                                  251, 389, 356, 454, 323, 361,
                                                  288, 397, 365, 379, 378, 400,
                                                  377, 152, 148, 176, 149, 150,
                                                  136, 172, 58, 132, 93, 234]]
                      # Head contour y AOI coordinates
    df_aoi_columns += [f'y_face{num}' for num in [127, 162, 21, 54, 103, 67,
                                                  109, 10, 338, 297, 332, 284,
                                                  251, 389, 356, 454, 323, 361,
                                                  288, 397, 365, 379, 378, 400,
                                                  377, 152, 148, 176, 149, 150,
                                                  136, 172, 58, 132, 93, 234]]
                      # Right hand x AOI coordinates
    df_aoi_columns += [f'x_r_hand{num}' for num in range(0,21)]
                      # Right hand y AOI coordinates
    df_aoi_columns += [f'y_r_hand{num}' for num in range(0,21)]
    for video_coords_file in tqdm(video_coords_files):
        df_aoi = pd.read_csv(video_coords_file, usecols=df_aoi_columns)
        df_aoi = df_aoi.reindex(df_aoi_columns, axis='columns') # Re-index columns to have the coordinates in the right order
        df_aoi = df_aoi.fillna(0)
        df_aoi = df_aoi.rename(columns={'x_face0': 'x_lat0', 'y_face0': 'y_lat0',
                                        'x_face17': 'x_lat17', 'y_face17': 'y_lat17'})
        # Convert mediapipe coords to video coords in px
        video_width = 2560
        video_height = 1600
        x_hand_cols = df_aoi.filter(regex='x_r_hand*').columns
        y_hand_cols = df_aoi.filter(regex='y_r_hand*').columns
        x_face_cols = df_aoi.filter(regex='x_face*').columns
        y_face_cols = df_aoi.filter(regex='y_face*').columns
        x_lat_cols = df_aoi.filter(regex='x_lat*').columns
        y_lat_cols = df_aoi.filter(regex='y_lat*').columns
        df_aoi[x_hand_cols] = df_aoi[x_hand_cols]*video_width
        df_aoi[y_hand_cols] = df_aoi[y_hand_cols]*video_height
        df_aoi[x_face_cols] = df_aoi[x_face_cols]*video_width
        df_aoi[y_face_cols] = df_aoi[y_face_cols]*video_height
        df_aoi[x_lat_cols] = df_aoi[x_lat_cols]*video_width
        df_aoi[y_lat_cols] = df_aoi[y_lat_cols]*video_height
        # Convert coordinates to integers
        fn_video_col = df_aoi['fn_video']
        df_aoi = df_aoi.loc[:, df_aoi.columns != 'fn_video'].astype(int)
        df_aoi.insert(0, 'fn_video', fn_video_col)
        # Pick stim to put in fn here
        stim_name = video_coords_file[video_coords_file.find('originals/')+len('originals/'):video_coords_file.find('_coordonates.csv')-len('coordonates.csv')]
        df_aoi.to_csv(os.path.join(args.path2videocoords, f'{args.stimulus}/{stim_name}_aoi_coordinates.csv'))
        
        
#%% 

def annot_of_timepoint(timepoint, stim_annot):
    annot_timepoint = []
    for time_annot_onset, time_annot_duration, time_annot_name in zip(stim_annot['onsets'], stim_annot['durations'], stim_annot['annotations']):
        if timepoint >= time_annot_onset and timepoint < time_annot_onset+time_annot_duration:
            annot_timepoint.append(time_annot_name)
    if len(annot_timepoint)>= 2:
        annot_timepoint = 'multiple_events'
    elif len(annot_timepoint) == 0:
        annot_timepoint = 'no_event'
    else:
        annot_timepoint = annot_timepoint[0]
    return annot_timepoint
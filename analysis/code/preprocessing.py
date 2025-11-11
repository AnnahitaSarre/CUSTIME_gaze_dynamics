#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CUSTIME preprocessing script: loads the raw data and saves an MNE raw object
Processes one raw at a time
"""


#%% Import modules

import argparse
from utils.dataloader import DataLoader


#%% Parse arguments

parser = argparse.ArgumentParser()

# Data
parser.add_argument('--subjects', '--s', type=str, default='11',
                    help='Subject number (only one!), e.g., 11')
parser.add_argument('--session_day', '--day', default='*')
parser.add_argument('--session_time', '--time', default='*')
parser.add_argument('--task', '--t', type=str, default='WordsUsers',
                    choices=['WordsUsers', 'SentencesUsers',
                             'WordsCtrl', 'SentencesCtrl'])
parser.add_argument('--run', '--r', type=int, default=1,
                    choices=[1, 2])
# Paths
parser.add_argument('--path2data', default='../data')
parser.add_argument('--path2deriv', default='../data/derivatives')
parser.add_argument('--path2figures', default='../figures')

args = parser.parse_args()


#%% Preprocessing

## Create the corresponding data object
data = DataLoader(args, args.subjects, 'preproc')

## Generate raw
et_raw = data.generate_et_raw()
data.save_et_raw(et_raw)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CUSTIME behavioral analysis script
"""

#%% Import modules

import argparse
import pandas as pd
import pingouin as pg
from utils.dataloader import DataLoader
from utils.behav_utils import population_analysis, pretest_analysis
from utils.redcap import function_final_table
from utils.stats import retrieve_ids
from utils.viz import plot_group_difference, plot_validity_difference, plot_behav_word
from scipy.stats import norm


#%% Parse arguments

parser = argparse.ArgumentParser()

# Data to analyse
parser.add_argument('--report', '--rep', type=bool, default=False)
parser.add_argument('--contrast', '--c', default='button_press')
parser.add_argument('--swap_queries', '--swap', '--sq', default=False)
parser.add_argument('--task', '--t', type=str, default=['SyllCS'],
                    nargs='+', choices=['SyllCS', 'SyllReading',
                                        'WordsUsers', 'SentencesUsers',
                                        'WordsCtrl', 'SentencesCtrl'])
# Data
parser.add_argument('--subjects', '--s', type=str, default=['*'],
                    help='Subjects numbers, e.g., 11 21 or * (all) or 1 (group 1)',
                    nargs='+')
parser.add_argument('--subjects_to_exclude', '--s_ex', type=str, default=['phantom','pilote'],
                    nargs='*')
parser.add_argument('--session_day', '--day', default='*')
parser.add_argument('--session_time', '--time', default='*')
parser.add_argument('--run', '--r', default='*',
                    choices=['1', '2', '3', '4', '*'])
# Paths
parser.add_argument('--path2data', default='../data')
parser.add_argument('--path2deriv', default='../data')
parser.add_argument('--path2figures', default='../figures')
# Analyses to perform
parser.add_argument('--overwrite_report', type=bool, default=True)

args = parser.parse_args()


#%% Prepare subject list and analysis

pd.set_option('display.max_rows', None, 'display.max_columns', None)

## If processing all participants or one or two group(s), retrieve participant IDs
retrieve_ids(args, 'beh')

behav_df = []
for subject in args.subjects:
    data = DataLoader(args, subject, 'behav')
    behav_df_sub = data.load_behav_df()
    ## Compute d' and criterion for the participant
    if args.stimulus == 'words':
        nb_target = behav_df_sub[behav_df_sub['pseudo']==1]['pseudo'].count()
        nb_non_target = behav_df_sub[behav_df_sub['pseudo']==0]['pseudo'].count()
        hit_rate = len(behav_df_sub[behav_df_sub['response_type']=='HIT'])/(nb_target)
        fa_rate = len(behav_df_sub[behav_df_sub['response_type']=='FA'])/(nb_non_target)
    if args.stimulus == 'sentences':
        nb_target = behav_df_sub[behav_df_sub['is_odd']==1]['is_odd'].count()
        nb_non_target = behav_df_sub[behav_df_sub['is_odd']==0]['is_odd'].count()
        hit_rate = len(behav_df_sub[behav_df_sub['response_type']=='HIT'])/(nb_target)
        fa_rate = len(behav_df_sub[behav_df_sub['response_type']=='FA'])/(nb_non_target)
    if 'syll' in args.stimulus:
        nb_target = behav_df_sub[behav_df_sub['target']==1]['target'].count()
        nb_non_target = behav_df_sub[behav_df_sub['target']==0]['target'].count()
        hit_rate = len(behav_df_sub[behav_df_sub['response_type']=='HIT'])/(nb_target)
        fa_rate = len(behav_df_sub[behav_df_sub['response_type']=='FA'])/(nb_non_target)
    if hit_rate == 1:
        hit_rate = 1-1/(2*nb_target)
    elif hit_rate == 0:
        hit_rate = 1/(2*nb_target)
    if fa_rate == 1:
        fa_rate = 1-1/(2*nb_non_target)
    elif fa_rate == 0:
        fa_rate = 1/(2*nb_non_target)
    behav_df_sub['dprime'] = norm.ppf(hit_rate)-norm.ppf(fa_rate)
    behav_df_sub['criterion'] = -0.5*(norm.ppf(hit_rate)+norm.ppf(fa_rate))
    behav_df_sub['hit_rate'] = hit_rate
    behav_df_sub['fa_rate'] = fa_rate
    behav_df.append(behav_df_sub)
behav_df = pd.concat(behav_df, ignore_index=True)
behav_df['group'] = behav_df['subject'].astype(str).str[0].astype(int)
behav_df['subject'] = behav_df['subject'].astype(str)
behav_df['response_validity'] = behav_df['response_validity'].astype(str)
if args.stimulus == 'words':
    cond = ['pseudo', 'mismatch','freq']
    behav_df['pseudo'] = behav_df['pseudo'].astype(str)
    behav_df['mismatch'] = behav_df['mismatch'].astype(str)
    behav_df['weight'] = (behav_df['pseudo'].apply(lambda x: 1.25 if x == '0' else 5) *
                          behav_df['freq'].apply(lambda x: 2) *
                          behav_df['mismatch'].apply(lambda x: 3))
if args.stimulus == 'sentences':
    cond = ['is_odd']
    behav_df.drop(behav_df[pd.isna(behav_df['is_odd'])].index, inplace=True)
    behav_df['is_odd'] = behav_df['is_odd'].astype(str)
if 'syll' in args.stimulus:
    stim_version = 'male_coder' if 'cs' in args.stimulus else 'uppercase'
    cond = ['rounding','place_articulation','handshape','position'] + [stim_version]
    behav_df['target'] = behav_df['target'].astype(str)
    

#%% Pretest and population analyses

pretest_analysis(args)

population_analysis()


#%% Tasks analyses

# Descriptive statistics
print(f"{behav_df['hit_rate'].describe()}\n")
print(f"{behav_df['fa_rate'].describe()}\n")
print(f"{behav_df['dprime'].describe()}\n")
print(f"{behav_df[['dprime','hit_rate','group']].groupby(['group']).describe()}\n")
print(f"{behav_df['response_time'].describe()}\n")
print(f"{behav_df[['response_time','group']].groupby(['group']).describe()}\n")
if args.stimulus == 'words':
     print(behav_df[['response_time','response_type','group']].groupby(['response_type','group']).describe())
# Compare with 0
for group in set(behav_df['group']):
    res = pg.wilcoxon(x=behav_df.loc[behav_df['group']==group]['dprime'],
                      y=[0 for i in range(len(behav_df.loc[behav_df['group']==group]))])
    print(f"\nComparison of {args.stimulus} dprime with 0 in group {group}: W-val={res.iloc[0]['W-val']}, p-val={res['p-val'].iloc[0]}\n")
# Average relevant data
behav_df = behav_df.groupby(['subject','group'] + cond)[['dprime','response_time']].mean().reset_index()

if args.stimulus == 'sentences':
    anova = pg.anova(data=behav_df, dv='dprime', between='group')
    print(f'\n## General ANOVA results for dprime in sentence questions:\n\n{anova}\n') # => Non significant
    
if 'syll' in args.stimulus:
    anova = pg.anova(data=behav_df, dv='dprime', between=['group',stim_version])
    print(f'\n## General ANOVA results for dprime in syllables:\n\n{anova}\n') # => Significant
    tukey = pg.pairwise_tukey(data=behav_df, dv='dprime', between='group')
    print("Post-hoc results:\n", tukey)
    anova = pg.anova(data=behav_df, dv='response_time', between=['group',stim_version])
    print(f'\n## General ANOVA results for response time in syllables:\n\n{anova}\n') # => Significant
    tukey = pg.pairwise_tukey(data=behav_df, dv='response_time', between='group')
    print("Post-hoc results:\n", tukey)
    
    
if args.stimulus == 'words':
    # All words, testing the effect of group only
    anova = pg.anova(data=behav_df, dv='dprime', between='group')
    print(f'\n## General ANOVA results for dprime in syllables:\n\n{anova}\n') # => Significant
    tukey = pg.pairwise_tukey(data=behav_df, dv='dprime', between='group')
    print("Post-hoc results:\n", tukey)
    # Focus on real words to test effect of freq and mismatch
    behav_real_df = behav_df[behav_df['pseudo']=='0']
    anova = pg.anova(data=behav_real_df, dv='dprime', between=['group','freq','mismatch'])
    print(f'\n## General ANOVA results for dprime in syllables:\n\n{anova}\n') # =>  Non significant

    # All words, testing the effect of group only
    anova = pg.anova(data=behav_df, dv='response_time', between='group')
    print(f'\n## General ANOVA results for response_time in syllables:\n\n{anova}\n') # => Significant
    tukey = pg.pairwise_tukey(data=behav_df, dv='response_time', between='group')
    print("Post-hoc results:\n", tukey)
    # Focus on real words to test effect of freq and mismatch
    anova = pg.anova(data=behav_real_df, dv='response_time', between=['group','freq','mismatch'])
    print(f'\n## General ANOVA results for response_time in syllables:\n\n{anova}\n') # => Significant
    tukey = pg.pairwise_tukey(data=behav_df, dv='response_time', between='mismatch')
    print("Post-hoc results:\n", tukey)
    behav_g1_df = behav_df[behav_df['group']==1]
    tukey = pg.pairwise_tukey(data=behav_g1_df, dv='response_time', between='freq')
    print("Post-hoc results:\n", tukey)
    behav_g2_df = behav_df[behav_df['group']==2]
    tukey = pg.pairwise_tukey(data=behav_g2_df, dv='response_time', between='freq')
    print("Post-hoc results:\n", tukey)
    
    
#%% Plotting

plot_group_difference(args, behav_df, 'response_time')
plot_group_difference(args, behav_df, 'dprime')

if args.stimulus == 'words':
    plot_validity_difference(args, behav_df, 'response_time')
    plot_behav_word(args, behav_df, 'response_time')
    plot_behav_word(args, behav_df, 'dprime')


#%% Compute redcap tables

# df1= function_final_table(nom_etude='custime',type_csv = 'with_comment')
# df2= function_final_table(nom_etude='custime',type_csv = 'without_comment')


#%% Compute covariate table

# questionnaire_file = os.path.join(args.path2deriv,'behavioral/questionnaire/RedCap_data_processed/custime_tableau_final_covariates.csv')
# questionnaire_df = pd.read_csv(questionnaire_file, delimiter=',', encoding='latin-1', index_col=0)
# questionnaire_df['subject'] = questionnaire_df['subject'].astype(str)

# pretest_file = os.path.join(args.path2deriv,'behavioral/pretest/pretest_comprehension_covariates.csv')
# pretest_df = pd.read_csv(pretest_file, delimiter=',', encoding='latin-1', index_col=0)
# pretest_df = pretest_df.drop('group', axis=1)
# pretest_df['subject'] = pretest_df['subject'].astype(str)

# dprime_info, criterion_info, rt_info = dprime_info.drop('group', axis=1), criterion_info.drop('group', axis=1), rt_info.drop('group', axis=1)
# covariates_df = questionnaire_df.merge(dprime_info, on='subject', how='left').merge(criterion_info, on='subject', how='left')
# covariates_df = covariates_df.merge(rt_info, on='subject', how='left').merge(pretest_df, on='subject', how='left')

# covariates_df.to_csv(os.path.join(args.path2deriv,'behavioral/custime_covariates.csv'))

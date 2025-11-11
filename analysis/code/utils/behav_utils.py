#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUSTIME population and comprehension pretest analysis

"""


#%% Import modules

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from scipy.stats import shapiro, levene


def pretest_analysis(args):
    #%% Import csv
    
    csv_path = '/mnt/ata-ST4000NM0165_ZAD8CWEG-part1/custime/custime_analysis/data/derivatives/behavioral/pretest'
    csv_file = os.path.join(csv_path,'custime_pretest.csv')
    df = pd.read_csv(csv_file, delimiter=',', encoding='latin-1')
    
    
    #%% Retrieve total of points in each category
    
    total_syllables = 0
    total_sentences = 0
    total_sentences_easy = 0
    total_sentences_medium = 0
    total_sentences_hard = 0
    
    for col in df.columns[1:]:
        if str(col).startswith('S'):
            total_syllables = total_syllables + df[col].iloc[-1]
        else:
            total_sentences = total_sentences + df[col].iloc[-1]
            if str(col).startswith('F'):
                total_sentences_easy = total_sentences_easy + df[col].iloc[-1]
            elif str(col).startswith('M'):
                total_sentences_medium = total_sentences_medium + df[col].iloc[-1]
            else:
                total_sentences_hard = total_sentences_hard + df[col].iloc[-1]
    
    
    #%% Create a dataframe that will serve as basis in various cases (see below)
    
    subject_array = pd.Series(df['id_sujet'].iloc[:-1], name='subject')
    df_base = subject_array.to_frame()
    df_base['group'] = np.nan
    for index in df_base.index:
        df_base['group'].loc[index] = df_base['subject'].loc[index][0]
        
    
    #%% Create a df to be used for further stats (LC request)
    
    # From CUSPEX script, not arranged for CUSTIME
    # df_stats = df_base.copy()
    # df_stats = pd.DataFrame(np.repeat(df_stats.values, 24, axis=0),columns=df_base.columns) # 1 col/sub/sentence
    # df_stats['sentence'] = (df.loc[~df['stim'].str.startswith('S')].iloc[:,0].tolist())*40
    # df_stats['total'] = (df.loc[~df['stim'].str.startswith('S')].iloc[:,2].tolist())*40
    # df_stats['difficulty'] = df_stats['sentence'].str[0]
    # df_stats['num_try'] = df_stats['sentence'].str[3]
    # df_stats['sentence'] = df_stats['sentence'].str[:2]
    # scores_list = [df.loc[~df['stim'].str.startswith('S')][col].tolist() for col in df.columns[3:]]
    # df_stats['correct_syll'] = [item for row in scores_list for item in row]
    # df_stats['wrong_syll'] = df_stats['total'].astype('float') - df_stats['correct_syll'].astype('float')
    # # df_stats['score'] = df_stats['score'].astype('float')/df_stats['total'].astype('float')*100
    # df_stats = df_stats.drop('total', axis=1)
    
    # df_stats.to_csv('pretest_comprehension_stats.csv')
    
    
    #%% Compute and save global individual results for covariates
    
    ## Copy the begining of df to create the 'scores' dfs
    df_subject_scores_all = df_base.copy()
    df_subject_scores_try1 = df_base.copy()
    df_subject_scores_try2 = df_base.copy()
    
    ## Compute sentences scores for the two groups mixing the two attemps
    df_sentences_all = df[[col for col in df.columns if not col.startswith('S')]].iloc[:-1,1:].astype('float')
    df_subject_scores_all['sentences_all_scores'] = df_sentences_all.sum(axis=1).reset_index(drop=True)
    df_subject_scores_all['sentences_all_scores'] = df_subject_scores_all['sentences_all_scores']/total_sentences*100
    
    df_sentences_easy_all = df[[col for col in df.columns if col.startswith('F')]].iloc[:-1,1:].astype('float')
    df_subject_scores_all['sentences_easy_scores'] = df_sentences_easy_all.sum(axis=1).reset_index(drop=True)
    df_subject_scores_all['sentences_easy_scores'] = df_subject_scores_all['sentences_easy_scores']/total_sentences_easy*100
    
    df_sentences_medium_all = df[[col for col in df.columns if col.startswith('M')]].iloc[:-1,1:].astype('float')
    df_subject_scores_all['sentences_medium_scores'] = df_sentences_medium_all.sum(axis=1).reset_index(drop=True)
    df_subject_scores_all['sentences_medium_scores'] = df_subject_scores_all['sentences_medium_scores']/total_sentences_medium*100
    
    df_sentences_hard_all = df[[col for col in df.columns if col.startswith('D')]].iloc[:-1,1:].astype('float')
    df_subject_scores_all['sentences_hard_scores'] = df_sentences_hard_all.sum(axis=1).reset_index(drop=True)
    df_subject_scores_all['sentences_hard_scores'] = df_subject_scores_all['sentences_hard_scores']/total_sentences_hard*100
    
    df_syllables_all = df[[col for col in df.columns if col.startswith('S')]].iloc[:-1,1:].astype('float')
    df_subject_scores_all['syllables_scores'] = df_syllables_all.sum(min_count=1, axis=1).reset_index(drop=True)
    df_subject_scores_all['syllables_scores'] = df_subject_scores_all['syllables_scores']/total_syllables*100
    
    df_subject_scores_all.to_csv(os.path.join(csv_path,'pretest_comprehension_covariates.csv')) # Use it to create covariates
    
    ## Compute sentences scores for the two groups for the two attemps separately
    # Try 1
    df_sentences_try1 = df[[col for col in df.columns if not col.startswith('S') and col.endswith('1')]].iloc[:-1].astype('float')
    df_subject_scores_try1['sentences_all_scores'] = df_sentences_try1.sum(axis=1).reset_index(drop=True)
    df_subject_scores_try1['sentences_all_scores'] = df_subject_scores_try1['sentences_all_scores']/(total_sentences/2)*100
    
    df_sentences_easy_try1 = df[[col for col in df.columns if col.startswith('F') and col.endswith('1')]].iloc[:-1].astype('float')
    df_subject_scores_try1['sentences_easy_scores'] = df_sentences_easy_try1.sum(axis=1).reset_index(drop=True)
    df_subject_scores_try1['sentences_easy_scores'] = df_subject_scores_try1['sentences_easy_scores']/(total_sentences_easy/2)*100
    
    df_sentences_medium_try1 = df[[col for col in df.columns if col.startswith('M') and col.endswith('1')]].iloc[:-1].astype('float')
    df_subject_scores_try1['sentences_medium_scores'] = df_sentences_medium_try1.sum(axis=1).reset_index(drop=True)
    df_subject_scores_try1['sentences_medium_scores'] = df_subject_scores_try1['sentences_medium_scores']/(total_sentences_medium/2)*100
    
    df_sentences_hard_try1 = df[[col for col in df.columns if col.startswith('D') and col.endswith('1')]].iloc[:-1].astype('float')
    df_subject_scores_try1['sentences_hard_scores'] = df_sentences_hard_try1.sum(axis=1).reset_index(drop=True)
    df_subject_scores_try1['sentences_hard_scores'] = df_subject_scores_try1['sentences_hard_scores']/(total_sentences_hard/2)*100
    
    df_syllables_try1 = df[[col for col in df.columns if col.startswith('S') and col.endswith('1')]].iloc[:-1].astype('float')
    df_subject_scores_try1['syllables_scores'] = df_syllables_try1.sum(min_count=1, axis=1).reset_index(drop=True)
    df_subject_scores_try1['syllables_scores'] = df_subject_scores_try1['syllables_scores']/(total_syllables/2)*100
    
    # Try 2
    df_sentences_try2 = df[[col for col in df.columns if not col.startswith('S') and col.endswith('2')]].iloc[:-1].astype('float')
    df_subject_scores_try2['sentences_all_scores'] = df_sentences_try2.sum(axis=1).reset_index(drop=True)
    df_subject_scores_try2['sentences_all_scores'] = df_subject_scores_try2['sentences_all_scores']/(total_sentences/2)*100
    
    df_sentences_easy_try2 = df[[col for col in df.columns if col.startswith('F') and col.endswith('2')]].iloc[:-1].astype('float')
    df_subject_scores_try2['sentences_easy_scores'] = df_sentences_easy_try2.sum(axis=1).reset_index(drop=True)
    df_subject_scores_try2['sentences_easy_scores'] = df_subject_scores_try2['sentences_easy_scores']/(total_sentences_easy/2)*100
    
    df_sentences_medium_try2 = df[[col for col in df.columns if col.startswith('M') and col.endswith('2')]].iloc[:-1].astype('float')
    df_subject_scores_try2['sentences_medium_scores'] = df_sentences_medium_try2.sum(axis=1).reset_index(drop=True)
    df_subject_scores_try2['sentences_medium_scores'] = df_subject_scores_try2['sentences_medium_scores']/(total_sentences_medium/2)*100
    
    df_sentences_hard_try2 = df[[col for col in df.columns if col.startswith('D') and col.endswith('2')]].iloc[:-1].astype('float')
    df_subject_scores_try2['sentences_hard_scores'] = df_sentences_hard_try2.sum(axis=1).reset_index(drop=True)
    df_subject_scores_try2['sentences_hard_scores'] = df_subject_scores_try2['sentences_hard_scores']/(total_sentences_hard/2)*100
    
    df_syllables_try2 = df[[col for col in df.columns if col.startswith('S') and col.endswith('2')]].iloc[:-1].astype('float')
    df_subject_scores_try2['syllables_scores'] = df_syllables_try2.sum(min_count=1, axis=1).reset_index(drop=True)
    df_subject_scores_try2['syllables_scores'] = df_subject_scores_try2['syllables_scores']/(total_syllables/2)*100
    
    
    #%% Plot participants' results mixing the two attemps
    
    ## Create an extended scores df to permit easier plotting
    df_subject_scores_all_plot = df_subject_scores_all.iloc[:,:2]
    col = df_subject_scores_all_plot.columns
    df_subject_scores_all_plot = pd.DataFrame(np.repeat(df_subject_scores_all_plot.values, 5, axis=0),columns=col)
    df_subject_scores_all_plot['stim_category'] = ['sentences_all','sentences_easy','sentences_medium','sentences_hard','syllables']*43
    scores_series = []
    for index in df_subject_scores_all.index:
        subject_score = df_subject_scores_all.loc[index].values[2:].tolist()
        scores_series = scores_series + subject_score
    df_subject_scores_all_plot['score'] = scores_series
    df_subject_scores_all_plot['group'] = pd.Categorical(df_subject_scores_all_plot['group'], ['1', '2']) # to arrange the order when plotting
    
    ## Scores for all categories displayed by group
    plt.figure(figsize=(12,10))
    df_subject_scores_all_plot = df_subject_scores_all_plot.drop(df_subject_scores_all_plot[df_subject_scores_all_plot['stim_category'] == 'syllables'].index)
    sns.boxplot(x='stim_category', y='score', hue='group', data=df_subject_scores_all_plot,  palette='pastel', legend=False)
    sns.stripplot(x='stim_category', y='score', hue='group', dodge=True, data=df_subject_scores_all_plot, palette='dark', legend=False)
    new_labels = ['All', 'Easy', 'Intermediate', 'Difficult']
    plt.xticks(ticks=plt.gca().get_xticks(), labels=new_labels, fontsize=14)
    plt.yticks(range(0, 101, 10), fontsize=14)
    plt.xlabel('Level of difficulty', fontsize=15, labelpad=11)
    plt.ylabel('Proportion of correctly transcribed phonemes (%)', fontsize=15, labelpad=10)
    # plt.title('CS sentences comprehension scores', fontsize= 20, y=1.025)
    plt.show()
    
    
    #%% Plot participants' results for the two attemps separately
    
    ## Create extended scores dfs to permit easier plotting (try 1)
    df_subject_scores_try1_plot = df_subject_scores_try1.iloc[:,:2]
    col = df_subject_scores_try1_plot.columns
    df_subject_scores_try1_plot = pd.DataFrame(np.repeat(df_subject_scores_try1_plot.values, 5, axis=0),columns=col)
    df_subject_scores_try1_plot['stim_category'] = ['sentences_all','sentences_easy','sentences_medium','sentences_hard','syllables']*43
    scores_series = []
    for index in df_subject_scores_try1.index:
        subject_score = df_subject_scores_try1.loc[index].values[2:].tolist()
        scores_series = scores_series + subject_score
    df_subject_scores_try1_plot['score'] = scores_series
    df_subject_scores_try1_plot['group'] = pd.Categorical(df_subject_scores_try1_plot['group'], ['1', '2']) # to arrange the order when plotting
    
    ## Create extended scores dfs to permit easier plotting (try 2)
    df_subject_scores_try2_plot = df_subject_scores_try2.iloc[:,:2]
    col = df_subject_scores_try2_plot.columns
    df_subject_scores_try2_plot = pd.DataFrame(np.repeat(df_subject_scores_try2_plot.values, 5, axis=0),columns=col)
    df_subject_scores_try2_plot['stim_category'] = ['sentences_all','sentences_easy','sentences_medium','sentences_hard','syllables']*43
    scores_series = []
    for index in df_subject_scores_try1.index:
        subject_score = df_subject_scores_try2.loc[index].values[2:].tolist()
        scores_series = scores_series + subject_score
    df_subject_scores_try2_plot['score'] = scores_series
    df_subject_scores_try2_plot['group'] = pd.Categorical(df_subject_scores_try2_plot['group'], ['1', '2']) # to arrange the order when plotting
    
    ## Scores for all categories displayed by group (one plot per attempt)
    fig, axs = plt.subplots(1,2, figsize=(24,15), sharey=True)
    sns.violinplot(x='stim_category', y='score', hue='group', data=df_subject_scores_try1_plot, inner='quartile', density_norm='count', palette='pastel', legend=False, ax=axs[0]).set(title='Pretest scores by stim category and group (1st attempt)')
    sns.stripplot(x='stim_category', y='score', hue='group', dodge=True, data=df_subject_scores_try1_plot, palette='dark', ax=axs[0])
    sns.violinplot(x='stim_category', y='score', hue='group', data=df_subject_scores_try2_plot, inner='quartile', density_norm='count', palette='pastel', ax=axs[1]).set(title='Pretest scores by stim category and group (2nd attempt)')
    sns.stripplot(x='stim_category', y='score', hue='group', dodge=True, data=df_subject_scores_try2_plot, palette='dark', ax=axs[1])
    
    
    #%% Compute statistics mixing the two attemps
    
    ## Create a df for each group, only keeping the scores
    df_g1_scores_all = df_subject_scores_all[df_subject_scores_all['group']=='1'].iloc[:, 2:-1] # Drop syllable column
    df_g2_scores_all = df_subject_scores_all[df_subject_scores_all['group']=='2'].iloc[:, 2:]
    
    ## Compute basic stats on each group's scores
    df_g1_scores_all.describe()
    df_g2_scores_all.describe()
    
    ## Testing normality with the Shapiro-Wilk test
    df_shapiro_all = pd.DataFrame(columns = df_g2_scores_all.columns, index = ['group 1','group 2'])
    for col in df_g1_scores_all.columns: # Group 1
        shapiro_result = shapiro(df_g1_scores_all[col])
        normality = ['normal' if shapiro_result.pvalue > 0.05 else 'not normal'][0] # if p>0.05 the data is normally distributed
        df_shapiro_all[col].iloc[0] = [shapiro_result.statistic, shapiro_result.pvalue, normality]
    for col in df_g2_scores_all.columns: # Group 2
        shapiro_result = shapiro(df_g2_scores_all[col])
        normality = ['normal' if shapiro_result.pvalue > 0.05 else 'not normal'][0] # if p>0.05 the data is normally distributed
        df_shapiro_all[col].iloc[1] = [shapiro_result.statistic, shapiro_result.pvalue, normality]
    
    ## Assess differences between the two groups
    # Means with Mann–Whitney U test (= Wilcoxon rank-sum test)
    df_2g_mwu = pd.DataFrame(columns = df_g1_scores_all.columns, index = ['U-val','p-val','RBC','CLES'])
    for col in df_g1_scores_all.columns:
        df_2g_mwu[col] = pg.mwu(df_g1_scores_all[col],df_g2_scores_all[col]).values.tolist()[0]
        
    # Compare variance
    pg.homoscedasticity(df_g1_scores_all['sentences_all_scores'], df_g2_scores_all['sentences_all_scores'])
    levene(df_g1_scores_all['sentences_all_scores'], df_g2_scores_all['sentences_all_scores'])
    
    ## Assess differences between sentences' level of difficulty
    df_g1_difficulty = pd.DataFrame(index = ['U-val','p-val','RBC','CLES']) # Group 1
    df_g1_difficulty['easy vs medium'] = pg.mwu(df_g1_scores_all.iloc[:,1],df_g1_scores_all.iloc[:,2]).values.tolist()[0]
    df_g1_difficulty['medium vs hard'] = pg.mwu(df_g1_scores_all.iloc[:,2],df_g1_scores_all.iloc[:,3]).values.tolist()[0]
    df_g1_difficulty['easy vs hard'] = pg.mwu(df_g1_scores_all.iloc[:,1],df_g1_scores_all.iloc[:,3]).values.tolist()[0]
    df_g2_difficulty = pd.DataFrame(index = ['U-val','p-val','RBC','CLES']) # Group 2
    df_g2_difficulty['easy vs medium'] = pg.mwu(df_g2_scores_all.iloc[:,1],df_g2_scores_all.iloc[:,2]).values.tolist()[0]
    df_g2_difficulty['medium vs hard'] = pg.mwu(df_g2_scores_all.iloc[:,2],df_g2_scores_all.iloc[:,3]).values.tolist()[0]
    df_g2_difficulty['easy vs hard'] = pg.mwu(df_g2_scores_all.iloc[:,1],df_g2_scores_all.iloc[:,3]).values.tolist()[0]
    df_g2_difficulty['sentences vs syllables'] = pg.mwu(df_g2_scores_all.iloc[:,0],df_g2_scores_all.iloc[:,4]).values.tolist()[0]
    
    
    #%% Compute statistics on the two attempts separately
    
    ## Try 1
    
    # Create a df for each group, only keeping the scores
    df_g1_scores_try1 = df_subject_scores_try1[df_subject_scores_try1['group']=='1'].iloc[:, 2:-1] # Drop syllable column
    df_g2_scores_try1 = df_subject_scores_try1[df_subject_scores_try1['group']=='2'].iloc[:, 2:]
    
    # Compute basic stats on each group's scores
    df_g1_scores_try1.describe()
    df_g2_scores_try1.describe()
    
    # Testing normality with the Shapiro-Wilk test
    df_shapiro_try1 = pd.DataFrame(columns = df_g2_scores_try1.columns, index = ['group 1','group 2'])
    for col in df_g1_scores_try1.columns: # Group 1
        shapiro_result = shapiro(df_g1_scores_try1[col])
        normality = ['normal' if shapiro_result.pvalue > 0.05 else 'not normal'][0] # if p>0.05 the data is normally distributed
        df_shapiro_try1[col].iloc[0] = [shapiro_result.statistic, shapiro_result.pvalue, normality]
    for col in df_g2_scores_try1.columns: # Group 2
        shapiro_result = shapiro(df_g2_scores_try1[col])
        normality = ['normal' if shapiro_result.pvalue > 0.05 else 'not normal'][0] # if p>0.05 the data is normally distributed
        df_shapiro_try1[col].iloc[1] = [shapiro_result.statistic, shapiro_result.pvalue, normality]
    print(df_shapiro_try1)
    
    ## Try 2
    
    # Create a df for each group, only keeping the scores
    df_g1_scores_try2 = df_subject_scores_try2[df_subject_scores_try2['group']=='1'].iloc[:, 2:-1] # Drop syllable column
    df_g2_scores_try2 = df_subject_scores_try2[df_subject_scores_try2['group']=='2'].iloc[:, 2:]
    
    # Compute basic stats on each group's scores
    df_g1_description = df_g1_scores_try2.describe()
    df_g2_description = df_g2_scores_try2.describe()
    
    # Testing normality with the Shapiro-Wilk test
    df_shapiro_try2 = pd.DataFrame(columns = df_g2_scores_try2.columns, index = ['group 1','group 2'])
    for col in df_g1_scores_try2.columns: # Group 1
        shapiro_result = shapiro(df_g1_scores_try2[col])
        normality = ['normal' if shapiro_result.pvalue > 0.05 else 'not normal'][0] # if p>0.05 the data is normally distributed
        df_shapiro_try2[col].iloc[0] = [shapiro_result.statistic, shapiro_result.pvalue, normality]
    for col in df_g2_scores_try2.columns: # Group 2
        shapiro_result = shapiro(df_g2_scores_try2[col])
        normality = ['normal' if shapiro_result.pvalue > 0.05 else 'not normal'][0] # if p>0.05 the data is normally distributed
        df_shapiro_try2[col].iloc[1] = [shapiro_result.statistic, shapiro_result.pvalue, normality]
    
    # Non normal distributions so Wilcoxon signed-rank test to compare tries 1 and 2 within each participant
    df_tries_diff = pd.DataFrame(columns = df_g2_scores_try1.columns, index = ['group 1','group 2'])
    for col in df_g1_scores_try1.columns: # Group 1
        df_tries_diff[col].iloc[0] = pg.wilcoxon(df_g1_scores_try1[col], df_g1_scores_try2[col])
    for col in df_g2_scores_try1.columns: # Group 1
        df_tries_diff[col].iloc[1] = pg.wilcoxon(df_g2_scores_try1[col], df_g2_scores_try2[col])
    print('Signiticant differences between attemps in all cases except for easy sentences in group 1.')
    print('Small sample so carefull with these results!')
    
    
    #%% Compare the two attempts
    
    for col in df_g1_scores_try1.columns: # Sentences Group 1
        print(f'Group 1 {col}:')
        print(pg.mwu(df_g1_scores_try1[col],df_g1_scores_try2[col])) # => No significant difference
        print(levene(df_g1_scores_try1[col],df_g1_scores_try2[col])) # => No significant difference
    
    for col in df_g2_scores_try1.columns: # Sentences Group 2
        print(f'Group 2 {col}:')
        print(pg.mwu(df_g2_scores_try1[col],df_g2_scores_try2[col])) # => No significant difference
        print(levene(df_g2_scores_try1[col],df_g2_scores_try2[col])) # => No significant difference
    
    # Syllables Group 2
    df_syllables_try1, df_syllables_try2 = df_syllables_try1.dropna(axis=1, how='all'), df_syllables_try2.dropna(axis=1, how='all')
    print('Within participants')
    print(pg.wilcoxon(df_syllables_try1.mean(),df_syllables_try2.mean())) # Within participants => significant difference
    print('For each syllable')
    for syll in range(0,16):
        if (df_syllables_try1.iloc[syll] - df_syllables_try2.iloc[syll] != 0).any():
            print(pg.wilcoxon(df_syllables_try1.iloc[syll],df_syllables_try2.iloc[syll])) # Sample sizes too small
    
    
    #%% Assess scores on syllables
    # !!! Not adapted to CUSTIME
    
    syll_dict = {'da': ['unrounded', 'coronal',], 'de': ['unrounded',], 'do': ['rounded',], 'du': ['rounded',],
                 'ma': ['unrounded',], 'me': ['unrounded',], 'mo': ['rounded',], 'mu': ['rounded',],
                 'pa': ['unrounded',], 'pe': ['unrounded',], 'po': ['rounded',], 'pu': ['rounded',],
                 'ta': ['unrounded',], 'te': ['unrounded',], 'to': ['rounded',], 'tu': ['rounded',]}
    
    # Individual stimuli
    df_syllables_all = df_syllables_all
    df_assess_syllables = pd.concat([df.iloc[24:,1], df_syllables_all.sum(axis=1)/len([col for col in df_syllables_all.columns if col.startswith('2')])*100], axis=1)
    df_assess_syllables = df_assess_syllables.rename(columns={0: 'score'})
    df_assess_syllables = df_assess_syllables.groupby(df['valeur']).aggregate('mean').reset_index()
    plt.figure(figsize=(8,6))
    sns.stripplot(x='valeur', y='score', dodge=True, data=df_assess_syllables.sort_values(by=['score']), palette='dark').set(title='Mean score for each syllable')
    df_assess_syllables.describe()
    
    # Assess further du po te tu
    df_assess_ind_syll = pd.concat([df.iloc[24:,1], df_syllables_all], axis=1)
    df_assess_du = df_assess_ind_syll[df_assess_ind_syll['valeur']=='du'].T
    df_assess_du.columns = ['first_try', 'second_try']
    df_assess_du = df_assess_du[1:]
    plt.figure(figsize=(8,6))
    sns.histplot(x=df_assess_du['first_try'],
                 stat='count', multiple='dodge', palette='dark', shrink=.7).set(title='Scores for syllable du')
    sns.histplot(x=df_assess_du['second_try'],
                 stat='count', multiple='dodge', palette='pastel', shrink=.7).set(title='Scores for syllable du')
    
    # Individual phonemes
    list_all_scores = []
    for phoneme in ['d', 'm', 'p', 't', 'a', 'e', 'o', 'u']:
        list_phoneme_scores = []
        for i_syll in df_assess_syllables.index:
            if phoneme in df_assess_syllables['valeur'].iloc[i_syll]:
                list_phoneme_scores.append(df_assess_syllables['score'].iloc[i_syll])
        list_all_scores.append(np.average(list_phoneme_scores))
    df_assess_phonemes = pd.DataFrame({'phoneme':['d', 'm', 'p', 't', 'a', 'e', 'o', 'u'],'score':list_all_scores})
    plt.figure(figsize=(8,6))
    sns.stripplot(x='phoneme', y='score', dodge=True, data=df_assess_phonemes.sort_values(by=['score']), palette='dark').set(title='Mean score for each phoneme')
    df_assess_phonemes.describe()
    


def population_analysis():
    #%% Import and arrange df
    
    csv_path = '/media/annahita.sarre/custime/custime_analysis/data/derivatives/behavioral/questionnaire/RedCap_data_processed'
    df = pd.read_csv(os.path.join(csv_path,'custime_tableau_final_demography_comm.csv'), sep=',')
    df['group'] = df['group'].astype(str)
    
    
    #%% Basic demographic stats for all 3 groups
    
    print('\n\n### Basic statistics for all group ###\n')
    
    # Arrange df
    df['sex'] = df['sex'].replace([0, 2], ['H','F'])
    df['diplome'] = df['diplome'].replace([0, 1, 2, 3, 4, 5, 6, 7],
                                          ['Certificat détudes primaires','Aucun diplôme','Brevet des collèges, BEPC',
                                           'Baccalauréat général, technologique, professionnel ou équivalent','CAP, BEP ou équivalent',
                                           'BTS, DUT, licence ou équivalent','Master ou équivalent','Doctorat'])
    
    plt.figure(figsize=(8,6))
    sns.violinplot(x=df['group'], y=df['age'], data=df, inner='quartile', palette='pastel', density_norm='count').set(title='Age by group')
    sns.stripplot(x=df['group'], y=df['age'], palette='dark')
    plt.show()
    print(df[['group', 'age']].groupby('group').describe())
    
    plt.figure(figsize=(8,6))
    sns.histplot(x=df['group'][~df['sex'].isna()], stat='count', hue='sex', multiple='dodge', data=df[~df['sex'].isna()], palette='pastel', shrink=.8).set(title='Sex by group')
    plt.show()
    print(df[['group', 'sex']].groupby('group').describe())
    
    plt.figure(figsize=(8,6))
    sns.histplot(x=df['group'][~df['diplome'].isna()], stat='count', hue='diplome',
                 hue_order=('Brevet des collèges, BEPC','Baccalauréat général, technologique, professionnel ou équivalent','BTS, DUT, licence ou équivalent','Master ou équivalent'),
                 multiple='dodge', data=df[~df['diplome'].isna()], palette='pastel', shrink=.8).set(title='Diploma by group')
    plt.show()
    print(df[['group', 'diplome']].groupby('group').describe())
    
    df['diplome'] = df['diplome'].replace(['Certificat détudes primaires','Aucun diplôme','Brevet des collèges, BEPC',
                                           'Baccalauréat général, technologique, professionnel ou équivalent','CAP, BEP ou équivalent',
                                           'BTS, DUT, licence ou équivalent','Master ou équivalent','Doctorat'],
                                          ["Pas d'études supérieures","Pas d'études supérieures","Pas d'études supérieures",
                                           "Pas d'études supérieures","Etudes supérieures","Etudes supérieures",
                                           "Etudes supérieures","Etudes supérieures"])
    plt.figure(figsize=(8,6))
    sns.histplot(x=df['group'][~df['diplome'].isna()], stat='count', hue='diplome',
                 multiple='dodge', data=df[~df['diplome'].isna()], palette='pastel', shrink=.8).set(title='Diploma by group')
    plt.yticks(np.arange(0, 20, 2))
    plt.show()
    print(df[['group', 'diplome']].groupby('group').describe())
    
    plt.figure(figsize=(16,6))
    sns.violinplot(x=df['group'], y=df['laterality score'], data=df, inner='quartile', palette='pastel').set(title='Laterality score by group')
    sns.stripplot(x=df['group'], y=df['laterality score'], palette='dark')
    plt.show()
    print(df[['group', 'laterality score']].groupby('group').describe())
    
    
    #%% Deafness in group 1
    
    print('\n\n### Group 1 deafness and scolarity ###\n')
    
    plt.figure(figsize=(8,6))
    sns.histplot(x=df.loc[df['group'] == '1']['gr1_degre_surdite'], discrete=True, stat='count', multiple='dodge', shrink=.8).set(title='Degre of deafness in group 1')
    plt.show()
    
    print("Progressive deafness onset?: 2 people said that their deafness appeared progressively starting at 1 year old, all the others said it began at 0")
    
    print("\nSevere deafness onset: from the 2 people above, one of them responded 1 year old and the other 2")
    
    print("\nDeafness onset: one person said it wasn't progressive but appeared abruptly at 1, others said it appeared at birth")
    
    plt.figure(figsize=(8,6))
    sns.histplot(x=df.loc[df['group'] == '1']['gr1_nb_appareils'], discrete=True, stat='count', multiple='dodge', palette='pastel', shrink=.8).set(title='Number of hearing devices in group 1')
    plt.show()
    
    plt.figure(figsize=(8,6))
    sns.displot(x=df.loc[df['gr1_nb_appareils'] > 0]['gr1_temps_appareil'], palette='pastel').set(title='Time spent with hearing device(s) in group 1')
    plt.show()
    
    plt.figure(figsize=(8,6))
    sns.histplot(x=df.loc[df['group'] == '1']['gr1_nb_implants'], discrete=True, stat='count', multiple='dodge', palette='pastel', shrink=.8).set(title='Number of cochlear implants in group 1')
    plt.show()
    
    plt.figure(figsize=(8,6))
    sns.displot(x=df.loc[df['gr1_nb_implants'] > 0]['gr1_implant_age'], palette='pastel').set(title='Age of cochlear implantation in group 1')
    plt.show()
    
    plt.figure(figsize=(8,6))
    sns.displot(x=df.loc[df['gr1_nb_implants'] > 0]['gr1_temps_implant'], palette='pastel').set(title='Time spent with cochlear implant(s) in group 1')
    plt.show()
    
    print(f"\nDeafness causes in participants (the '0' and '1' are already in the raw table, no mistake): {df.loc[df['group'] == '1']['gr1_cause_surdite'].values.tolist()}")
    
    print(f"\nOther deaf people in family (g1):\n\n\tFamily relationship: {df.loc[df['group'] == '1']['gr1_degre_parente'].values.tolist()}"+
          f"\n\n\tFamily degree of deafness: {df.loc[df['group'] == '1']['gr1_degre_surdite_famille'].values.tolist()}"+
          f"\n\n\tFamily hearing devices: {df.loc[df['group'] == '1']['gr1_famille_appareil'].values.tolist()}"+
          f"\n\n\tFamily language: {df.loc[df['group'] == '1']['gr1_famille_langage'].values.tolist()}"+
          f"\n\n\tFamily cause of deafness: {df.loc[df['group'] == '1']['gr1_famille_cause'].values.tolist()}")
    
    
    #%% Scolarity and reading in group 1
    
    print("Scolarity: Vast majority were in standard school with coders (some without).")
    
    plt.figure(figsize=(8,6))
    sns.displot(x=df.loc[df['group'] == '1']['gr1_lpc_lecture_age']).set(title='Age of reading acquisition in group 1')
    plt.show()
    print(df.loc[df['group'] == '1']['gr1_lpc_lecture_age'].describe())
    
    print("Difficulty in reading acquisition: 4 declared they have had some. Nowadays they report a excellent (3 people) or good level in reading.")
    
    
    #%% Deaf environment in groups 1 and 2
    
    print('\n\n### Group 1 and 2 deaf environment ###\n')
    
    # Arrange df
    df['group'] = pd.Categorical(df['group'], ['1','2','3']) # Cannot order categories directly in seaborn histplot
    df['Entourage_merged'] = np.nan_to_num(df['Entourage_merged'], nan=9) # Because nan cannot be converted to int
    df['Entourage_merged'] = [int(x) for x in df['Entourage_merged']]
    
    dict_entourage = {'0':'Amical', '1':'Professionnel', '2':'Scolaire ou étudiant', '3': 'Associatif', '4':'Familial'}
    with pd.option_context('mode.chained_assignment', None): # To desactivate SettingWithCopyWarning
        for index in df.loc[df['group'] != '3'].index:
            if df.loc[index]['Entourage_merged'] == 9.0:
                df['Entourage_merged'].loc[index] = "Pas d'entourage"
            else:
                list_entourage = []
                for num in list(str(int(df.loc[index]['Entourage_merged']))):
                    txt = dict_entourage[num]
                    list_entourage.append(txt)
                list_entourage = (' + ').join(list_entourage)
                df['Entourage_merged'].loc[index] = list_entourage
    df['Entourage_merged'] = df['Entourage_merged'].replace(9.0,np.nan) # Turn back 9 to nan
    
    
    plt.figure(figsize=(8,6))
    sns.histplot(x=df.loc[df['group'] != '3']['group'], hue=df.loc[df['group'] != '3']['Entourage_merged'],
                 stat='count', multiple='dodge', palette='pastel', shrink=.7).set(title='Deaf environment of groups 1 and 2')
    plt.show()
    
    print(f"\nGroup 1 participants are both in contact with oralising, with or without LPC, and signing people.\n\
    Details: {df.loc[df['gr1_entourage_langage'] != '0']['gr1_entourage_langage'].values.tolist()}")
          
    print(f"\nSame with group 2 participants, with a bit less LSF.\n\
    Details: {df.loc[df['gr2_entourage_langage'] != '0']['gr2_entourage_langage'].values.tolist()}")
    
    print("\nDeaf people around both groups' participants have a mix of hearing devices and cochlear implants.")
    
    
    #%% LPC in groups 1 and 2
    
    print('\n\n### Group 1 and 2 LPC usage ###\n')
    
    # Arrange df
    df['lpc_freq'] = df['lpc_freq'].replace([0, 1, 2, 3, 4, 5, 6],
                                          ['Tous les jours','Plusieurs fois par semaine','1 fois par semaine',
                                           'Plusieurs fois par mois','1 fois par mois',
                                           'Plusieurs fois par an','1 fois par an ou moins'])
    df['lpc_usefulness'] = df['lpc_niv_comprendre'] - df['lpc_niv_labial']
    
    plt.figure(figsize=(8,6))
    sns.violinplot(x=df.loc[df['group'] != '3']['group'], y=df['lpc_age'], density_norm='count', inner='quartile', palette='pastel', order=['1','2']).set(title='Age of LPC learning in groups 1 and 2')
    sns.stripplot(x=df.loc[df['group'] != '3']['group'], y=df['lpc_age'], data=df.loc[df['group'] != '3'], palette='dark', order=['1','2'])
    plt.show()
    print(df.loc[df['group'] != '3'][['group','lpc_age']].groupby('group').describe())
    
    plt.figure(figsize=(8,6))
    sns.histplot(x=df.loc[df['group'] != '3']['group'], hue=df.loc[df['group'] != '3']['lpc_freq'],
                 hue_order=('Tous les jours','Plusieurs fois par semaine',
                            'Plusieurs fois par mois','1 fois par mois',
                            'Plusieurs fois par an','1 fois par an ou moins'),
                 stat='count', multiple='dodge', palette='pastel', shrink=.7).set(title='Frequency of LPC use in groups 1 and 2')
    plt.show()
    print("The four partcipants who responded 'Plusieurs fois par an' are two implanted deaf CS users who started CS at 1 and 5 years old and two hearing users.\
          All scored rather high on the pretest.")
    
    plt.figure(figsize=(8,6))
    sns.histplot(x=df.loc[df['group'] != '3']['group'], hue=df.loc[df['group'] != '3']['lpc_niv_coder'],
                 stat='count', multiple='dodge', palette='pastel', shrink=.7).set(title='Self-reported level of LPC production in groups 1 and 2 (min 0 max 4)')
    plt.show()
    
    plt.figure(figsize=(8,6))
    sns.histplot(x=df.loc[df['group'] != '3']['group'],hue=df.loc[df['group'] != '3']['lpc_niv_comprendre'],
                 stat='count', multiple='dodge', palette='pastel', shrink=.7).set(title='Self-reported level of LPC comprehension in groups 1 and 2 (min 0 max 4)')
    plt.show()
    
    plt.figure(figsize=(8,6))
    sns.histplot(x=df.loc[df['group'] != '3']['group'], hue=df.loc[df['group'] != '3']['lpc_niv_labial'],
                 stat='count', multiple='dodge', palette='pastel', shrink=.7).set(title='Self-reported level of lipreading in groups 1 and 2 (min 0 max 4)')
    plt.show()
    
    plt.figure(figsize=(8,6))
    sns.histplot(x=df.loc[df['group'] != '3']['group'], hue=df.loc[df['group'] != '3']['lpc_usefulness'],
                 stat='count', multiple='dodge', palette='pastel', shrink=.7).set(title='Self-reported LPC usefulness (= LPC - lipreading alone) in groups 1 and 2')
    plt.show()
    print(df[['group', 'lpc_usefulness']].groupby('group').describe())
    print("The 3 deaf partcipants who have 0 lpc usefulness are implanted deaf CS users who started CS early and use it daily.\
          All scored high on the pretest.")
    
    
    
    # Arrange df for LPC learning ressource
    df['lpc_apprentissage___merged'] = np.nan_to_num(df['lpc_apprentissage___merged'], nan=9) # Because nan cannot be converted to int
    df['lpc_apprentissage___merged'] = [int(x) for x in df['lpc_apprentissage___merged']]
    with pd.option_context('mode.chained_assignment', None): # To desactivate SettingWithCopyWarning
        for response_index in df.index:
                coded_response = str(int(df['lpc_apprentissage___merged'].loc[response_index])).replace('4','0')
                if 'phon' in df['gr1_lpc_apprentissage_precision'].loc[response_index]:
                    coded_response = coded_response + '6'
                if 'CODALI' in df['gr1_lpc_apprentissage_precision'].loc[response_index]:
                    coded_response = coded_response + '7'
                df['lpc_apprentissage___merged'].loc[response_index] = int(coded_response)
    
    dict_lpc_apprentissage = {'0':'Parents', '1':'Ecole', '2':'Stage', '3':'Autodidacte', '5':'Licence LPC', '6':'Orthophoniste', '7':'Service d\'accompagnement'}
    with pd.option_context('mode.chained_assignment', None): # To desactivate SettingWithCopyWarning
        for index in df.loc[df['group'] != '3'].index:
            list_lpc_apprentissage = []
            for num in list(str(int(df.loc[index]['lpc_apprentissage___merged']))):
                txt = dict_lpc_apprentissage[num]
                list_lpc_apprentissage.append(txt)
            list_lpc_apprentissage = list(set(list_lpc_apprentissage)) # Delete duplicates
            list_lpc_apprentissage = (' + ').join(list_lpc_apprentissage)
            df['lpc_apprentissage___merged'].loc[index] = list_lpc_apprentissage
    
    plt.figure(figsize=(15,11))
    sns.histplot(x=df.loc[df['group'] != '3']['group'], hue=df.loc[df['group'] != '3']['lpc_apprentissage___merged'],
                 stat='count', multiple='dodge', palette='pastel', shrink=.7).set(title='LPC learning ressource in groups 1 and 2')
    plt.show()
    
    # Arrange df for LPC usage
    df['lpc_cadre___merged'] = np.nan_to_num(df['lpc_cadre___merged'], nan=9) # Because nan cannot be converted to int
    df['lpc_cadre___merged'] = [int(x) for x in df['lpc_cadre___merged']]
    with pd.option_context('mode.chained_assignment', None): # To desactivate SettingWithCopyWarning
        for response_index in df.index:
            if 'seule' in df['gr1_lpc_cadre_precision'].loc[response_index]:
                coded_response = str(int(df['lpc_cadre___merged'].loc[response_index])).replace('4','')
                coded_response = coded_response + '5'
                df['lpc_cadre___merged'].loc[response_index] = int(coded_response)
            if df['gr2_lpc_cadre_precision'].loc[response_index] != '0':
                coded_response = str(int(df['lpc_cadre___merged'].loc[response_index])).replace('4','')
                if 'formateur' in df['gr2_lpc_cadre_precision'].loc[response_index]:
                    coded_response = coded_response + '3'    
                df['lpc_cadre___merged'].loc[response_index] = int(coded_response)
                
    df['lpc_cadre___merged'] = [str(x) for x in df['lpc_cadre___merged']]
    df['lpc_cadre___merged'] = df['lpc_cadre___merged'].str.replace('4', '', regex=False)
    
    dict_lpc_cadre = {'0':'Famille', '2':'Amis', '3':'Travail/Etudes', '5':'Seul'}
    with pd.option_context('mode.chained_assignment', None): # To desactivate SettingWithCopyWarning
        for index in df.loc[df['group'] != '3'].index:
            list_lpc_cadre = []
            for num in list(str(int(df.loc[index]['lpc_cadre___merged']))):
                txt = dict_lpc_cadre[num]
                list_lpc_cadre.append(txt)
            list_lpc_cadre = list(set(list_lpc_cadre)) # Delete duplicates
            list_lpc_cadre = (' + ').join(list_lpc_cadre)
            df['lpc_cadre___merged'].loc[index] = list_lpc_cadre
    
    plt.figure(figsize=(8,6))
    sns.histplot(x=df.loc[df['group'] != '3']['group'], hue=df.loc[df['group'] != '3']['lpc_cadre___merged'],
                 stat='count', multiple='dodge', palette='pastel', shrink=.7).set(title='LPC usage in groups 1 and 2')
    plt.show()
    
    
    #%% CS in groups 1 and 2
    
    print('\n\n### Group 1 and 2 CS usage ###\n')
    
    # Arrange df
    df['cs_yn'] = df['cs_yn'].replace([0, 1], ['No','Yes'])
    
    plt.figure(figsize=(8,6))
    sns.histplot(x=df.loc[df['group'] != '3']['group'], hue=df.loc[df['group'] != '3']['cs_yn'], hue_order=('Yes','No'),
                 stat='count', multiple='dodge', palette='pastel', shrink=.7).set(title='Knowledge on CS in groups 1 and 2')
    plt.show()
    
    print("\nAll American/UK CS, one participant mentionned notions in Arabic CS.")
    
    plt.figure(figsize=(8,6))
    sns.violinplot(x=df.loc[df['cs_yn'] == 'Yes']['group'], y=df['cs_age'], data=df.loc[df['group'] != '3'],inner='quartile', palette='pastel').set(title='Age of CS learning in groups 1 and 2')
    sns.stripplot(x=df.loc[df['cs_yn'] == 'Yes']['group'], y=df['cs_age'], data=df.loc[df['group'] != '3'], palette='dark')
    plt.show()
    print(df.loc[df['cs_yn'] == 'Yes'][['group','cs_age']].groupby('group').describe())
    
    plt.figure(figsize=(8,6))
    sns.histplot(x=df.loc[df['cs_yn'] == 'Yes']['group'], hue=df.loc[df['group'] != '3']['cs_niv_coder'], hue_order=(1,2,3),
                 stat='count', multiple='dodge', palette='pastel', shrink=.7).set(title='Self-reported level of CS production in groups 1 and 2 (min 0 max 4)')
    plt.show()
    
    plt.figure(figsize=(8,6))
    sns.histplot(x=df.loc[df['cs_yn'] == 'Yes']['group'], hue=df.loc[df['group'] != '3']['cs_niv_comprendre'], hue_order=(0,1,2,3),
                 stat='count', multiple='dodge', palette='pastel', shrink=.7).set(title='Self-reported level of CS comprehension in groups 1 and 2 (min 0 max 4)')
    plt.show()
    
    
    #%% LSF in groups 1 and 2
    
    print('\n\n### Group 1 and 2 LSF usage ###\n')
    
    # Arrange df
    df['lsf_yn'] = df['lsf_yn'].replace([0, 1], ['No','Yes'])
    df['lsf_freq'] = df['lsf_freq'].replace([0, 1, 2, 3, 4, 5, 6],
                                          ['Tous les jours','Plusieurs fois par semaine','1 fois par semaine',
                                           'Plusieurs fois par mois','1 fois par mois',
                                           'Plusieurs fois par an','1 fois par an ou moins'])
    
    plt.figure(figsize=(8,6))
    sns.histplot(x=df.loc[df['group'] != '3']['group'], hue=df.loc[df['group'] != '3']['lsf_yn'], hue_order=('Yes','No'),
                 stat='count', multiple='dodge', palette='pastel', shrink=.7).set(title='Knowledge on LSF in groups 1 and 2')
    plt.show()
    
    plt.figure(figsize=(8,6))
    sns.violinplot(x=df.loc[df['lsf_yn'] == 'Yes']['group'], y=df['lsf_age'], inner='quartile', palette='pastel').set(title='Age of LSF learning in groups 1 and 2')
    sns.stripplot(x=df.loc[df['lsf_yn'] == 'Yes']['group'], y=df['lsf_age'], data=df.loc[df['group'] != '3'], palette='dark')
    plt.show()
    print(df.loc[df['lsf_yn'] == 'Yes'][['group','lsf_age']].groupby('group').describe())
    
    plt.figure(figsize=(12,8))
    sns.histplot(x=df.loc[df['lsf_yn'] == 'Yes']['group'], hue=df.loc[df['group'] != '3']['lsf_freq'],
                 hue_order=('Tous les jours','Plusieurs fois par semaine',
                            'Plusieurs fois par mois','1 fois par mois',
                            'Plusieurs fois par an','1 fois par an ou moins'),
                 stat='count', multiple='dodge', palette='pastel', shrink=.7).set(title='Frequency of LSF use in groups 1 and 2')
    plt.show()
    
    plt.figure(figsize=(8,6))
    sns.histplot(x=df.loc[df['lsf_yn'] == 'Yes']['group'], hue=df.loc[df['group'] != '3']['lsf_niv_exprimer'], hue_order=(0,1,2,3,4),
                 stat='count', multiple='dodge', palette='pastel', shrink=.7).set(title='Self-reported level of LSF production in groups 1 and 2 (min 0 max 4)')
    plt.show()
    
    plt.figure(figsize=(8,6))
    sns.histplot(x=df.loc[df['lsf_yn'] == 'Yes']['group'], hue=df.loc[df['group'] != '3']['lsf_niv_comprendre'], hue_order=(0,1,2,3,4),
                 stat='count', multiple='dodge', palette='pastel', shrink=.7).set(title='Self-reported level of LSF comprehension in groups 1 and 2 (min 0 max 4)')
    plt.show()
    
    # Arrange df for LSF learning ressource
    df['lsf_apprentissage___merged'] = np.nan_to_num(df['lsf_apprentissage___merged'], nan=9.0) # Because nan cannot be converted to int
    with pd.option_context('mode.chained_assignment', None): # To desactivate SettingWithCopyWarning
        for response_index in df.index:
            if df['gr1_lsf_apprentissage_precision'].loc[response_index] != '0':
                coded_response = str(int(df['lsf_apprentissage___merged'].loc[response_index])).replace('4','')
                if 'amis' in df['gr1_lsf_apprentissage_precision'].loc[response_index]:
                    coded_response = coded_response + '6'
                if 'rencontres' in df['gr1_lsf_apprentissage_precision'].loc[response_index]:
                    coded_response = coded_response + '6'
                if 'collegues' in df['gr1_lsf_apprentissage_precision'].loc[response_index]:
                    coded_response = coded_response + '6'
                if 'association' in df['gr1_lsf_apprentissage_precision'].loc[response_index]:
                    coded_response = coded_response + '27'
                if 'formation intensive IVT + compagnon/amis' in df['gr1_lsf_apprentissage_precision'].loc[response_index]:
                    coded_response = coded_response + '26'
                df['lsf_apprentissage___merged'].loc[response_index] = int(coded_response)
            if df['gr2_lsf_apprentissage_precision'].loc[response_index] != '0':
                coded_response = str(int(df['lsf_apprentissage___merged'].loc[response_index])).replace('4','')
                if 'travail' in df['gr2_lsf_apprentissage_precision'].loc[response_index]:
                    coded_response = coded_response + '2'
                if 'professionnel' in df['gr2_lsf_apprentissage_precision'].loc[response_index]:
                    coded_response = coded_response + '2'
                if 'Faculte' in df['gr2_lsf_apprentissage_precision'].loc[response_index]:
                    coded_response = coded_response + '5'
                df['lsf_apprentissage___merged'].loc[response_index] = int(coded_response)
    
    dict_lsf_apprentissage = {'0':'Parents', '1':'Ecole', '2':'Stage', '3':'Autodidacte', '5':'Licence LPC', '6': 'Pairs', '7': 'Structure d\'accueil/association'}
    with pd.option_context('mode.chained_assignment', None): # To desactivate SettingWithCopyWarning
        for index in df.loc[df['group'] != '3'].index:
            list_lsf_apprentissage = []
            for num in list(str(int(df.loc[index]['lsf_apprentissage___merged']))):
                if num != '9':
                    txt = dict_lsf_apprentissage[num]
                    list_lsf_apprentissage.append(txt)
            list_lsf_apprentissage = list(set(list_lsf_apprentissage)) # Delete duplicates
            list_lsf_apprentissage = (' + ').join(list_lsf_apprentissage)
            df['lsf_apprentissage___merged'].loc[index] = list_lsf_apprentissage
    df['lsf_apprentissage___merged'] = df['lsf_apprentissage___merged'].replace(['',9],np.nan) # Turn back 9 to nan
    
    plt.figure(figsize=(20,11))
    sns.histplot(x=df.loc[df['lsf_yn'] == 'Yes']['group'], hue=df.loc[df['group'] != '3']['lsf_apprentissage___merged'],
                 stat='count', multiple='dodge', palette='pastel', shrink=.7).set(title='LSF learning ressource in groups 1 and 2')
    plt.show()
    
    # Arrange df for LSF usage
    df['lsf_cadre___merged'] = np.nan_to_num(df['lsf_cadre___merged'], nan=9.0) # Because nan cannot be converted to int
    df['lsf_cadre___merged'] = [int(x) for x in df['lsf_cadre___merged']]
    with pd.option_context('mode.chained_assignment', None): # To desactivate SettingWithCopyWarning
        for response_index in df.index:
            if df['gr1_lsf_cadre_precision'].loc[response_index] != '0':
                coded_response = str(int(df['lsf_cadre___merged'].loc[response_index])).replace('6','')
                if 'autodidacte' in df['gr1_lsf_cadre_precision'].loc[response_index]:
                    coded_response = coded_response + '5'
                if 'compagnon' in df['gr1_lsf_cadre_precision'].loc[response_index]:
                    coded_response = coded_response + '0'
                df['lsf_cadre___merged'].loc[response_index] = int(coded_response)
    
    df['lsf_cadre___merged'] = [str(x) for x in df['lsf_cadre___merged']]
    df['lsf_cadre___merged'] = df['lsf_cadre___merged'].str.replace('4', '', regex=False)
    
    dict_lsf_cadre = {'0':'Avec ma famille', '2':'Avec mes amis', '3':'Dans le cadre de mon travail/de mes études', '5':'Seul'}
    with pd.option_context('mode.chained_assignment', None): # To desactivate SettingWithCopyWarning
        for index in df.loc[df['group'] != '3'].index:
            list_lsf_cadre = []
            for num in list(df.loc[index]['lsf_cadre___merged']):
                if num != '9':
                    txt = dict_lsf_cadre[num]
                    list_lsf_cadre.append(txt)
            list_lsf_cadre = list(set(list_lsf_cadre)) # Delete duplicates
            list_lsf_cadre = (' + ').join(list_lsf_cadre)
            df['lsf_cadre___merged'].loc[index] = list_lsf_cadre
    df['lsf_cadre___merged'] = df['lsf_cadre___merged'].replace(['',9],np.nan) # Turn back 9 to nan
    
    plt.figure(figsize=(14,10))
    sns.histplot(x=df.loc[df['lsf_yn'] == 'Yes']['group'], hue=df.loc[df['group'] != '3']['lsf_cadre___merged'],
                 stat='count', multiple='dodge', palette='pastel', shrink=.7).set(title='LSF usage in groups 1 and 2')
    plt.show()
    
    
    #%% Other LS in groups 1 and 2
    
    print('\n\n### Group 1 and 2 LS usage ###\n')
    
    # Arrange df
    df['ls_yn'] = df['ls_yn'].replace([0, 1], ['No','Yes'])
    
    print('\n2 group 1 participants and no group 2 participant declared having knowledge on other SL.\n\
    The 2 participants also know LSF, which they have started learning earlier and have an equal level (all excellent) than with LS.\n\
    They both know International sign language, one also knows DSGS and the other ASL and Auslan.')
    
    plt.figure(figsize=(2,6))
    sns.violinplot(y=df.loc[df['ls_yn'] == 'Yes']['ls_age'], inner='quartile', palette='pastel').set(title='Age of other LS learning in group 1')
    sns.stripplot(y=df.loc[df['ls_yn'] == 'Yes']['ls_age'], palette='dark')
    plt.show()

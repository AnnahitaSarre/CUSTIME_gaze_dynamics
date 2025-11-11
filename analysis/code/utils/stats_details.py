#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUSTIME fonctions for detailed statistical analyses (all computed statistics)
"""


#%% Import modules

import os
import sys
import pandas as pd
import pingouin as pg
# import rpy2.robjects as ro
import scipy.stats as stats
from itertools import product
# from rpy2.robjects import pandas2ri
# from rpy2.robjects.packages import importr


#%% 


def proportion_aoi_hits_detailed_analysis(args, subjects, proportion_hits_df):
    pd.set_option('display.max_rows', None, 'display.max_columns', None)
    # results_path = os.path.join(args.path2deriv,f'eye-tracking/statistical_results/{args.stimulus}')
    # from utils.stats import select_stims_for_cond
    # ## Describe and exclude outlier if not excluded during subjects selection
    # if '12' in args.subjects:
    #     for cond in args.cond_dict[args.stimulus]:
    #         df_cond = select_stims_for_cond(args, proportion_hits_df, cond)
    #         print(f"\nStats for s12 in cond {cond}:\n\n{df_cond[df_cond['subject']=='12'].describe()}")
    #     proportion_hits_df = proportion_hits_df[proportion_hits_df['subject'] != '12']
    
    # ## Compute group results for all conditions separately
    # all_aoi_columns = [col for col in list(proportion_hits_df.columns) if col not in ['subject', 'group', 'stim', 'frame',
    #                                                                                   'predictable', 'freq', 'mismatch', 'pseudo']]
    
    # groups_name = "+".join(f"{group}" for group in args.group)
    # file = open(os.path.join(results_path,f'desc_{args.stimulus}_{groups_name}.txt'), 'w')
    # sys.stdout = file
    # for cond in args.cond_dict[args.stimulus]:
    #     print(f'\n\n\n#################### Displaying general results for {cond} ####################\n')
    #     df_cond = select_stims_for_cond(args, proportion_hits_df, cond)
    #     print(df_cond.describe())
    # sys.stdout = sys.__stdout__
    # file.close()


    # ## Initiate analyses
    # pandas2ri.activate()
    # base = importr('base')
    # stats = importr('stats')
    # jmv = importr('jmv')
    # aois_list = ['proportion_hits_all_face','proportion_hits_face_lips_no_other',
    #               'proportion_lat_face_left','proportion_lat_lips_left','proportion_lat_upper_face_left']
    # aois_list_last_word = [f"{aoi}_last_word" for aoi in aois_list]
    
    # groups_name = "+".join(f"{group}" for group in args.group)
    # file = open(os.path.join(results_path,f'anova_{args.stimulus}_{groups_name}_thr{args.alpha}.txt'), 'w')
    # sys.stdout = file

    # if args.stimulus == 'sentences':
    #     ## Compute and save full analysis table
    #     proportion_hits_df_save = pd.melt(proportion_hits_df,
    #                                       id_vars=[col for col in proportion_hits_df.columns if col not in all_aoi_columns],
    #                                       value_vars=all_aoi_columns, var_name='aoi', value_name='proportion')
    #     proportion_hits_df_save['proportion'] = proportion_hits_df_save['proportion']/100
    #     proportion_hits_df_save['proportion'] = proportion_hits_df_save['proportion'].replace(0,0.001)
    #     proportion_hits_df_save['proportion'] = proportion_hits_df_save['proportion'].replace(1,0.999)
    #     proportion_hits_df_save['proportion'] = logit(proportion_hits_df_save['proportion'])
    #     proportion_hits_df_save = proportion_hits_df_save.pivot_table(index=['subject','group'], columns=['predictable','aoi'], 
    #                                                                   values='proportion')
    #     proportion_hits_df_save.columns = ['subject','group'] + [f'{col[0]}' for col in proportion_hits_df_save.columns[2:]]
    #     proportion_hits_df_save.to_csv('utils/sentences_analysis_table.csv')
    #     ## Whole sentences and last words in-place analyses
    #     for aoi in aois_list + aois_list_last_word:
    #         proportion_hits_df_aoi = proportion_hits_df.drop([col for col in all_aoi_columns if col != aoi], axis=1)
    #         proportion_hits_df_aoi = pd.melt(proportion_hits_df_aoi,
    #                               id_vars=[col for col in proportion_hits_df_aoi.columns if col not in all_aoi_columns],
    #                               value_name='proportion')
    #         proportion_hits_df_aoi['proportion'] = proportion_hits_df_aoi['proportion']/100
    #         proportion_hits_df_aoi['proportion'] = proportion_hits_df_aoi['proportion'].replace(0,0.001)
    #         proportion_hits_df_aoi['proportion'] = proportion_hits_df_aoi['proportion'].replace(1,0.999)
    #         proportion_hits_df_aoi['proportion'] = logit(proportion_hits_df_aoi['proportion'])
    #         # Effect of groups and conditions
    #         proportion_hits_df_aoi = proportion_hits_df_aoi.pivot_table(index=['subject','group'], columns=['predictable'], 
    #                                                                     values='proportion', aggfunc='mean').reset_index()
    #         proportion_hits_df_aoi.columns = ['subject','group'] + [f'predictable_{col[0]}' for col in proportion_hits_df_aoi.columns[2:]]
    #         proportion_hits_df_aoi_r = pandas2ri.py2rpy(proportion_hits_df_aoi)
    #         ro.globalenv['proportion_hits_df_aoi_r'] =  proportion_hits_df_aoi_r
    #         if len(args.group) > 1:
    #             jmv_formula = "anovaRM(data=proportion_hits_df_aoi_r, rm=list(list(label='predictable', levels=c('0','1')))," + \
    #                 "rmCells = list(list(measure='predictable_0', cell=c('0')), list(measure='predictable_1', cell=c('1')))," + \
    #                 "bs = group, effectSize = 'eta', rmTerms = ~ predictable, bsTerms = ~ group," +\
    #                 "postHoc = list(c('group'), c('predictable'), c('predictable','group'))," +\
    #                 f"postHocCorr = c('tukey'), emMeans = ~predictable, emmTables=TRUE, ciWidthEmm={args.alpha})"
    #         else:
    #             jmv_formula = "anovaRM(data=proportion_hits_df_aoi_r, rm=list(list(label='predictable', levels=c('0', '1'))),"  + \
    #                 "rmCells = list(list(measure='predictable_0', cell=c('0')), list(measure='predictable_1', cell=c('1')))," + \
    #                 "effectSize = 'eta', rmTerms = ~ predictable,"  +\
    #                 "postHoc = list(c('predictable'))," +\
    #                 f"postHocCorr = c('tukey'), emMeans = ~ predictable, emmTables=TRUE, ciWidthEmm={args.alpha})"
    #         anova_result = ro.r(jmv_formula)
    #         aoi_text = 'last words of ' if 'last' in aoi else ''
    #         print(f'#################### General ANOVA results for {aoi_text}sentences for {aoi} ####################\n\n\n\n{anova_result}\n')
    #     sys.stdout = sys.__stdout__
    #     file.close()
    
    # elif args.stimulus == 'words':
    #     ## Compute and save full analysis table
    #     proportion_hits_df_save = pd.melt(proportion_hits_df,
    #                                       id_vars=[col for col in proportion_hits_df.columns if col not in all_aoi_columns],
    #                                       value_vars=all_aoi_columns, var_name='aoi', value_name='proportion')
    #     proportion_hits_df_save['proportion'] = proportion_hits_df_save['proportion']/100
    #     proportion_hits_df_save['proportion'] = proportion_hits_df_save['proportion'].replace(0,0.001)
    #     proportion_hits_df_save['proportion'] = proportion_hits_df_save['proportion'].replace(1,0.999)
    #     proportion_hits_df_save['proportion'] = logit(proportion_hits_df_save['proportion'])
    #     proportion_hits_df_save = proportion_hits_df_save.pivot_table(index=['subject','group'],
    #                                                                   columns=['aoi','pseudo','freq','mismatch'], 
    #                                                                   values='proportion', aggfunc='mean').reset_index()
    #     proportion_hits_df_save.columns = ['subject','group'] + [f'{col[0]}_pseudo-{col[1]}_freq-{col[2]}_mismatch-{col[3]}' for col in proportion_hits_df_save.columns[2:]]
    #     proportion_hits_df_save.to_csv('utils/words_analysis_table.csv')
    #     ## Real words in-place analyses
    #     proportion_hits_df_real = select_stims_for_cond(args, proportion_hits_df, 'pseudo_0')
    #     for aoi in aois_list:
    #         proportion_hits_df_aoi = proportion_hits_df_real.drop([col for col in all_aoi_columns if col != aoi], axis=1)
    #         proportion_hits_df_aoi = pd.melt(proportion_hits_df_aoi,
    #                               id_vars=[col for col in proportion_hits_df_aoi.columns if col not in all_aoi_columns],
    #                               value_name='proportion')
    #         proportion_hits_df_aoi['proportion'] = proportion_hits_df_aoi['proportion']/100
    #         proportion_hits_df_aoi['proportion'] = proportion_hits_df_aoi['proportion'].replace(0,0.001)
    #         proportion_hits_df_aoi['proportion'] = proportion_hits_df_aoi['proportion'].replace(1,0.999)
    #         proportion_hits_df_aoi['proportion'] = logit(proportion_hits_df_aoi['proportion'])
    #         # Effect of groups and conditions
    #         proportion_hits_df_aoi = proportion_hits_df_aoi.pivot_table(index=['subject','group'], columns=['freq','mismatch'], 
    #                                                                       values='proportion', aggfunc='mean').reset_index()
    #         proportion_hits_df_aoi.columns = ['subject', 'group'] + [f'freq-{col[0]}_mismatch-{int(col[1])}' for col in proportion_hits_df_aoi.columns[2:]]
    #         proportion_hits_df_aoi_r = pandas2ri.py2rpy(proportion_hits_df_aoi)
    #         ro.globalenv['proportion_hits_df_aoi_r'] = proportion_hits_df_aoi_r
    #         if len(args.group) > 1:
    #             jmv_formula = "anovaRM(data=proportion_hits_df_aoi_r, rm = list(list(label='freq', levels=c('l', 'h')), list(label='mismatch', levels=c('0', '1', '2')))," + \
    #                 "rmCells = list(list(measure='freq-h_mismatch-0', cell=c('h', '0')), list(measure='freq-h_mismatch-1', cell=c('h', '1')), list(measure='freq-h_mismatch-2', cell=c('h', '2'))," + \
    #                     "list(measure='freq-l_mismatch-0', cell=c('l', '0')), list(measure='freq-l_mismatch-1', cell=c('l', '1')), list(measure='freq-l_mismatch-2', cell=c('l', '2'))),"  + \
    #                 "bs = group, effectSize = 'eta', rmTerms = ~ freq + mismatch + freq:mismatch, bsTerms = ~ group," +\
    #                 "postHoc = list(c('group'), c('freq'), c('mismatch'), c('group', 'freq'),  c('group', 'mismatch'), c('freq','mismatch'))," +\
    #                 f"postHocCorr = c('tukey'), emMeans = ~group, emmTables=TRUE, ciWidthEmm={args.alpha})" 
    #         else:
    #             jmv_formula = "anovaRM(data=proportion_hits_df_aoi_r, rm = list(list(label='freq', levels=c('l', 'h')), list(label='mismatch', levels=c('0', '1', '2')))," + \
    #                 "rmCells = list(list(measure='freq-h_mismatch-0', cell=c('h', '0')), list(measure='freq-h_mismatch-1', cell=c('h', '1')), list(measure='freq-h_mismatch-2', cell=c('h', '2'))," + \
    #                     "list(measure='freq-l_mismatch-0', cell=c('l', '0')), list(measure='freq-l_mismatch-1', cell=c('l', '1')), list(measure='freq-l_mismatch-2', cell=c('l', '2'))),"  + \
    #                 "effectSize = 'eta', rmTerms = ~ freq + mismatch + freq:mismatch," +\
    #                 "postHoc = list(c('freq'), c('mismatch'), c('freq', 'mismatch'))," +\
    #                 f"postHocCorr = c('tukey'), emMeans = ~mismatch, emmTables=TRUE, ciWidthEmm={args.alpha})"
    #         anova_result = ro.r(jmv_formula)
    #         print(f'#################### General ANOVA results for words for {aoi} ####################\n\n\n\n{anova_result}\n')
    #     ## Pseudowords in-place analyses
    #     proportion_hits_df_pseudo = select_stims_for_cond(args, proportion_hits_df, 'pseudo_1')
    #     for aoi in aois_list:
    #         proportion_hits_df_aoi = proportion_hits_df_pseudo.drop([col for col in all_aoi_columns if col != aoi], axis=1)
    #         proportion_hits_df_aoi = pd.melt(proportion_hits_df_aoi,
    #                               id_vars=[col for col in proportion_hits_df_aoi.columns if col not in all_aoi_columns],
    #                               value_name='proportion')
    #         proportion_hits_df_aoi['proportion'] = proportion_hits_df_aoi['proportion']/100
    #         proportion_hits_df_aoi['proportion'] = proportion_hits_df_aoi['proportion'].replace(0,0.001)
    #         proportion_hits_df_aoi['proportion'] = proportion_hits_df_aoi['proportion'].replace(1,0.999)
    #         proportion_hits_df_aoi['proportion'] = logit(proportion_hits_df_aoi['proportion'])
    #         # Effect of groups and conditions
    #         proportion_hits_df_aoi = proportion_hits_df_aoi.pivot_table(index=['subject','group'], columns=['freq','mismatch'], 
    #                                                                       values='proportion', aggfunc='mean').reset_index()
    #         proportion_hits_df_aoi.columns = ['subject', 'group'] + [f'freq-{col[0]}_mismatch-{int(col[1])}' for col in proportion_hits_df_aoi.columns[2:]]
    #         proportion_hits_df_aoi_r = pandas2ri.py2rpy(proportion_hits_df_aoi)
    #         ro.globalenv['proportion_hits_df_aoi_r'] = proportion_hits_df_aoi_r
    #         if len(args.group) > 1:
    #             jmv_formula = "anovaRM(data=proportion_hits_df_aoi_r, rm = list(list(label='freq', levels=c('l', 'h')), list(label='mismatch', levels=c('0', '1', '2')))," + \
    #                 "rmCells = list(list(measure='freq-h_mismatch-0', cell=c('h', '0')), list(measure='freq-h_mismatch-1', cell=c('h', '1')), list(measure='freq-h_mismatch-2', cell=c('h', '2'))," + \
    #                     "list(measure='freq-l_mismatch-0', cell=c('l', '0')), list(measure='freq-l_mismatch-1', cell=c('l', '1')), list(measure='freq-l_mismatch-2', cell=c('l', '2'))),"  + \
    #                 "bs = group, effectSize = 'eta', rmTerms = ~ freq + mismatch + freq:mismatch, bsTerms = ~ group," +\
    #                 "postHoc = list(c('group'), c('freq'), c('mismatch'), c('group', 'freq'),  c('group', 'mismatch'), c('freq','mismatch'))," +\
    #                 f"postHocCorr = c('tukey'), emMeans = ~group, emmTables=TRUE, ciWidthEmm={args.alpha})"
    #         else:
    #             jmv_formula = "anovaRM(data=proportion_hits_df_aoi_r, rm = list(list(label='freq', levels=c('l', 'h')), list(label='mismatch', levels=c('0', '1', '2')))," + \
    #                 "rmCells = list(list(measure='freq-h_mismatch-0', cell=c('h', '0')), list(measure='freq-h_mismatch-1', cell=c('h', '1')), list(measure='freq-h_mismatch-2', cell=c('h', '2'))," + \
    #                     "list(measure='freq-l_mismatch-0', cell=c('l', '0')), list(measure='freq-l_mismatch-1', cell=c('l', '1')), list(measure='freq-l_mismatch-2', cell=c('l', '2'))),"  + \
    #                 "effectSize = 'eta', rmTerms = ~ freq + mismatch + freq:mismatch," +\
    #                 "postHoc = list(c('freq'), c('mismatch'), c('freq', 'mismatch'))," +\
    #                 f"postHocCorr = c('tukey'), emMeans = ~freq, emmTables=TRUE, ciWidthEmm={args.alpha})"
    #         anova_result = ro.r(jmv_formula) # Same model as for real words, but need to recompute the formula to reboot the aoi loop
    #         print(f'#################### General ANOVA results for pseudowords for {aoi} ####################\n\n\n\n{anova_result}\n')
        

    #     if any([sub.startswith('3') for sub in subjects]):
    #         sys.stdout = sys.__stdout__
    #         file.close()
    #     ## Real words vs pseudowords in the deaf and hearing cs users model in-place analyses
    #     if not any(sub.startswith('3') for sub in subjects):
    #         pass
    #         proportion_hits_df_real_and_pseudo = proportion_hits_df
    #         for aoi in aois_list:
    #             proportion_hits_df_aoi = proportion_hits_df_real_and_pseudo.drop([col for col in all_aoi_columns if col != aoi], axis=1)
    #             proportion_hits_df_aoi = pd.melt(proportion_hits_df_aoi,
    #                                               id_vars=[col for col in proportion_hits_df_aoi.columns if col not in all_aoi_columns],
    #                                               value_name='proportion')
    #             proportion_hits_df_aoi['proportion'] = proportion_hits_df_aoi['proportion']/100
    #             proportion_hits_df_aoi['proportion'] = proportion_hits_df_aoi['proportion'].replace(0,0.001)
    #             proportion_hits_df_aoi['proportion'] = proportion_hits_df_aoi['proportion'].replace(1,0.999)
    #             proportion_hits_df_aoi['proportion'] = logit(proportion_hits_df_aoi['proportion'])
    #             # Effect of groups and conditions
    #             proportion_hits_df_aoi = proportion_hits_df_aoi.pivot_table(index=['subject', 'group'], columns=['pseudo','freq','mismatch'], 
    #                                                                         values='proportion', aggfunc='mean').reset_index()
    #             proportion_hits_df_aoi.columns = ['subject', 'group'] + [f'pseudo-{col[0]}_freq-{col[1]}_mismatch-{int(col[2])}' for col in proportion_hits_df_aoi.columns[2:]]
    #             proportion_hits_df_aoi_r = pandas2ri.py2rpy(proportion_hits_df_aoi)
    #             ro.globalenv['proportion_hits_df_aoi_r'] = proportion_hits_df_aoi_r
    #             jmv_formula = "anovaRM(data=proportion_hits_df_aoi_r, rm = list(list(label='pseudo', levels=c('0','1')), list(label='freq',levels=c('l', 'h')), list(label='mismatch', levels=c('0','1','2')))," + \
    #                 "rmCells = list(list(measure='pseudo-0_freq-h_mismatch-0', cell=c('0','h','0')), list(measure='pseudo-0_freq-h_mismatch-1', cell=c('0','h','1')), list(measure='pseudo-0_freq-h_mismatch-2', cell=c('0','h','2'))," + \
    #                     "list(measure='pseudo-0_freq-l_mismatch-0', cell=c('0','l','0')), list(measure='pseudo-0_freq-l_mismatch-1', cell=c('0','l','1')), list(measure='pseudo-0_freq-l_mismatch-2', cell=c('0','l','2'))," + \
    #                     "list(measure='pseudo-1_freq-h_mismatch-0', cell=c('1','h','0')), list(measure='pseudo-1_freq-h_mismatch-1', cell=c('1','h','1')), list(measure='pseudo-1_freq-h_mismatch-2', cell=c('1','h','2'))," + \
    #                         "list(measure='pseudo-1_freq-l_mismatch-0', cell=c('1','l','0')), list(measure='pseudo-1_freq-l_mismatch-1', cell=c('1','l','1')), list(measure='pseudo-1_freq-l_mismatch-2', cell=c('1','l','2'))),"  + \
    #                 "bs = group, effectSize = 'eta', rmTerms = ~ freq + mismatch + pseudo + freq:mismatch + freq:pseudo + mismatch:pseudo, bsTerms = ~ group," +\
    #                 "postHoc = list(c('group'), c('freq'), c('mismatch'), c('pseudo')," +\
    #                 "c('group', 'freq'),  c('group', 'mismatch'), c('group', 'pseudo'), c('freq','mismatch'), c('pseudo', 'mismatch'), c('pseudo', 'freq'), c('group', 'pseudo', 'mismatch'))," +\
    #                 f"postHocCorr = c('tukey'), emMeans = ~pseudo, emmTables=TRUE, ciWidthEmm={args.alpha})"
    #             anova_result = ro.r(jmv_formula)
    #             print(f'#################### General ANOVA results for words and pseudowords for {aoi} ####################\n\n\n\n{anova_result}\n') 
    #         sys.stdout = sys.__stdout__
    #         file.close()
            
            
def proportion_correlation(args, proportion_hits_df):
    df_cov = pd.read_csv(os.path.join(args.path2deriv,'behavioral/custime_covariates.csv'))
    df_cov['subject'] = df_cov['subject'].astype('str')
    proportion_hits_df = proportion_hits_df.fillna(0)
    num_cols = proportion_hits_df.select_dtypes(include='number').columns
    df_aoi = proportion_hits_df.groupby("subject", as_index=False)[num_cols].agg('median')
    df = pd.merge(df_aoi, df_cov, on="subject", how="inner")
    df = df[df['subject'].astype(str).str[0] == '2']
    # df_corr = df_corr[(df_corr['pseudo']=='0') & (df_corr['freq_x']=='l')]
    aoi_list = ['proportion_hits_face_lips_no_other','proportion_lat_face_left',
                'proportion_lat_lips_left','proportion_lat_upper_face_left']
    cov_list = [
        # 'sentences_all_scores','sentences_hard_scores','lpc_age','syllables_scores','dprime',
                'lsf_age', 'lsf_niv_comprendre', 'lsf_niv_exprimer']
    for cov, aoi in product(cov_list,aoi_list):
        df_corr_aoi = df[df[aoi].notna()]
        res = stats.pearsonr(df[aoi], df[cov])
        print(f'Correlation of {aoi} with {cov}: r={res[0]:.02f}, p={res[1]:.04f}')
    print()
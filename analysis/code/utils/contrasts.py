#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CUSTIME contrasts listing script: a list of all desired contrasts, each as a dictionary,
which contains the queries (e.g., "shape=='one_finger'", "shape=='five_fingers'"),
condition names ("one finger", "five fingers"), colors for plotting ("r", "b"), etc.
Cross-decoding, intra-condition decoding and across-condition decoding are
handled programatically (see at the end of this function)
"""


def load_contrasts(args):
    
    contrasts = {}
    
    color_palette = [
    '#332288',  # dark blue
    '#88CCEE',  # light blue
    '#117733',  # dark green
    '#44AA99',  # teal
    '#999933',  # olive
    '#DDCC77',  # sand
    '#CC6677',  # rose
    '#882255',  # wine
    '#AA4499',  # purple
    '#661100',  # dark brown
    '#6699CC',  # steel blue
    '#888888',  # gray
    '#E69F00',  # orange
    '#56B4E9',  # sky blue
    '#009E73',  # bluish green
    '#F0E442',  # yellow
    ]

    lines = [
    'solid',            # ────
    'dashed',           # ─ ─ ─
    'dotted',           # ⋅⋅⋅⋅⋅
    'dashdot',          # ─ ⋅ ─ ⋅
    (0, (1, 1)),        # very fine dotted
    (0, (5, 5)),        # long dash
    (0, (3, 1, 1, 1)),  # long-short-short-short
    (0, (5, 2, 2, 2)),  # long-short pattern
    (0, (4, 4, 1, 4)),  # long-short-alt
    (0, (2, 1)),        # short dash
    (0, (6, 2)),        # longer dash
    (0, (1, 2)),        # fine sparse dots
    (0, (7, 1)),        # long dash, tight
    (0, (4, 1, 2, 1)),  # varied dash-dot
    (0, (6, 1, 1, 1)),  # long dash + dots
    (0, (3, 2, 1, 2)),  # custom rhythmic
    ]
    
    #%% Contrast to use for all ET roi analyses
    # Just to be able to use data.epoch_data()
    # Always start at 0 to synchronize gaze and video, long interval here because the analysis will anyway stop at the end of the video/table
    # video durations between 1.47 and 3.17 for Words and 4.41 and 11.58 for Sentences
    
    contrasts["eye_tracking"] = {}
    if args.stimulus == 'words':
        contrasts["eye_tracking"]["epoch_times"] = [0, 3.5]
    elif args.stimulus == 'sentences':
        contrasts["eye_tracking"]["epoch_times"] = [0, 12]
    else:
        contrasts["eye_tracking"]["epoch_times"] = [-0.3, 1.8]
    
    #%% Contrasts common to all custime experiments
    
    contrasts["button_press_fixation"] = {}
    contrasts["button_press_fixation"]["queries"] = ["event_type=='button_press'",
                                                     "event_type=='fixation_cross'"]
    contrasts["button_press_fixation"]["labels"] = ["Button press",
                                                    "Fixation"]
    contrasts["button_press_fixation"]["colors"] = ["#FB9A99", "#A6CEE3"]
    contrasts["button_press_fixation"]["lines"] = ['solid', 'dashed']
    contrasts["button_press_fixation"]["epoch_times"] = [-0.3, 1.2]
    
    ##
    
    contrasts["response_validity"] = {}
    contrasts["response_validity"]["queries"] = ["response_validity==1",
                                                 "response_validity==0"]
    contrasts["response_validity"]["labels"] = ["Valid response",
                                                "Invalid response"]
    contrasts["response_validity"]["colors"] = ["#FB9A99", "#A6CEE3"]
    contrasts["response_validity"]["lines"] = ['solid', 'dashed']
    contrasts["response_validity"]["epoch_times"] = [-0.3, 1.2]
    
    ##
    
    contrasts["response_type"] = {}
    contrasts["response_type"]["queries"] = ["response_type=='CR'",
                                             "response_type=='HIT'"
                                             "response_type=='FA'",
                                             "response_type=='MISS'"]
    contrasts["response_type"]["labels"] = ["CR",
                                            "HIT",
                                            "FA",
                                            "MISS"]
    contrasts["response_type"]["colors"] = ['#A6CEE3', '#B2DF8A', '#FB9A99', '#FDBF6F']
    contrasts["response_type"]["lines"] = ['solid', 'dashed', 'dotted', 'dashdot']
    contrasts["response_type"]["epoch_times"] = [-0.3, 1.2]
    
    ##
    
    contrasts["valid_response"] = {}
    contrasts["valid_response"]["queries"] = ["response_validity==1"]
    contrasts["valid_response"]["labels"] = ["Valid response"]
    contrasts["valid_response"]["colors"] = ["b"]
    contrasts["valid_response"]["lines"] = ['solid']
    contrasts["valid_response"]["epoch_times"] = [-0.3, 1.2]
    
    ##
    
    contrasts["invalid_response"] = {}
    contrasts["invalid_response"]["queries"] = ["response_validity==0"]
    contrasts["invalid_response"]["labels"] = ["Invalid response"]
    contrasts["invalid_response"]["colors"] = ["b"]
    contrasts["invalid_response"]["lines"] = ['solid']
    contrasts["invalid_response"]["epoch_times"] = [-0.3, 1.2]
    
    ##
    
    contrasts["fixation"] = {}
    contrasts["fixation"]["queries"] = ["event_type=='fixation_cross'"]
    contrasts["fixation"]["labels"] = ["Fixation cross"]
    contrasts["fixation"]["colors"] = ["b"]
    contrasts["fixation"]["lines"] = ['solid']
    contrasts["fixation"]["epoch_times"] = [0, 0.9]
    
    
    
    #%% Contrasts for the "words" experiment
    
    contrasts["word"] = {}
    contrasts["word"]["queries"] = ["event_type=='word_onset'"]
    contrasts["word"]["labels"] = ["Word"]
    contrasts["word"]["colors"] = ["b"]
    contrasts["word"]["lines"] = ['solid']
    contrasts["word"]["epoch_times"] = [-0.1, 12]
    
    ##
    
    contrasts["word_fixation"] = {}
    contrasts["word_fixation"]["queries"] = ["event_type=='word_onset'",
                                             "event_type=='fixation_cross'"]
    contrasts["word_fixation"]["labels"] = ["Word",
                                            "Fixation"]
    contrasts["word_fixation"]["colors"] = ["#FB9A99", "#A6CEE3"]
    contrasts["word_fixation"]["lines"] = ['solid', 'dashed']
    contrasts["word_fixation"]["epoch_times"] = [-0.1, 12]
    
    ##
    
    contrasts["pseudo"] = {}
    contrasts["pseudo"]["queries"] = ["event_type=='word_onset' and pseudo==1"]
    contrasts["pseudo"]["labels"] = ["Pseudoword"]
    contrasts["pseudo"]["colors"] = ["b"]
    contrasts["pseudo"]["lines"] = ['solid']
    contrasts["pseudo"]["epoch_times"] = [-0.3, 3.5]
    
    ##
    
    contrasts["real_word"] = {}
    contrasts["real_word"]["queries"] = ["event_type=='word_onset' and pseudo==0"]
    contrasts["real_word"]["labels"] = ["Real word"]
    contrasts["real_word"]["colors"] = ["b"]
    contrasts["real_word"]["lines"] = ['solid']
    contrasts["real_word"]["epoch_times"] = [-0.3, 3.5]
    
    ##
    
    contrasts["pseudo_real"] = {}
    contrasts["pseudo_real"]["queries"] = ["event_type=='word_onset' and pseudo==1",
                                           "event_type=='word_onset' and pseudo==0"]
    contrasts["pseudo_real"]["labels"] = ["Pseudoword",
                                          "Real word"]
    contrasts["pseudo_real"]["colors"] = ["#FB9A99", "#A6CEE3"]
    contrasts["pseudo_real"]["lines"] = ['solid', 'dashed']
    contrasts["pseudo_real"]["epoch_times"] = [-0.3, 3.5]
    
    ##
    
    contrasts["mismatch"] = {}
    contrasts["mismatch"]["queries"] = ["event_type=='word_onset' and pseudo==0 and mismatch==0",
                                        "event_type=='word_onset' and pseudo==0 and mismatch==1",
                                        "event_type=='word_onset' and pseudo==0 and mismatch==2"]
    contrasts["mismatch"]["labels"] = ["0 mismatch",
                                       "1 mismatch",
                                       "2 mismatch"]
    contrasts["mismatch"]["colors"] = ['#A6CEE3', '#B2DF8A', '#FB9A99']
    contrasts["mismatch"]["lines"] = ['solid', 'dashed', 'dotted']
    contrasts["mismatch"]["epoch_times"] = [-0.3, 3.5]
    
    ##
    
    contrasts["frequency"] = {}
    contrasts["frequency"]["queries"] = ["event_type=='word_onset' and pseudo==0 and freq=='l'",
                                         "event_type=='word_onset' and pseudo==0 and freq=='h'"]
    contrasts["frequency"]["labels"] = ["Low frequency",
                                        "High frequency"]
    contrasts["frequency"]["colors"] = ["#FB9A99", "#A6CEE3"]
    contrasts["frequency"]["lines"] = ['solid', 'dashed']
    contrasts["frequency"]["epoch_times"] = [-0.3, 3.5]
    
    ##
    
    contrasts["frequent"] = {}
    contrasts["frequent"]["queries"] = ["event_type=='word_onset' and pseudo==0 and freq=='h'"]
    contrasts["frequent"]["labels"] = ["High frequency"]
    contrasts["frequent"]["colors"] = ["b"]
    contrasts["frequent"]["lines"] = ['solid']
    contrasts["frequent"]["epoch_times"] = [-0.3, 3.5]
    
    ##
    
    contrasts["unfrequent"] = {}
    contrasts["unfrequent"]["queries"] = ["event_type=='word_onset' and pseudo==0 and freq=='l'"]
    contrasts["unfrequent"]["labels"] = ["Low frequency"]
    contrasts["unfrequent"]["colors"] = ["b"]
    contrasts["unfrequent"]["lines"] = ['solid']
    contrasts["unfrequent"]["epoch_times"] = [-0.3, 3.5]
    
    
    ##
    
    contrasts["word_cats"] = {}
    contrasts["word_cats"]["queries"] = ["event_type=='word_onset' and pseudo==0 and mismatch==0 and freq=='l'",
                                         "event_type=='word_onset' and pseudo==0 and mismatch==0 and freq=='h'",
                                         "event_type=='word_onset' and pseudo==0 and mismatch==1 and freq=='l'",
                                         "event_type=='word_onset' and pseudo==0 and mismatch==1 and freq=='h'",
                                         "event_type=='word_onset' and pseudo==0 and mismatch==2 and freq=='l'",
                                         "event_type=='word_onset' and pseudo==0 and mismatch==2 and freq=='h'"]
    contrasts["word_cats"]["labels"] = ["Word l0",
                                        "Word h0",
                                        "Word l1",
                                        "Word h1",
                                        "Word l2",
                                        "Word h2"]
    contrasts["word_cats"]["colors"] = ["#A6CEE3", "#B2DF8A", "#FB9A99",
                                        "#FDBF6F", "#CAB2D6", "#FFFF99"]
    contrasts["word_cats"]["lines"] = ['solid', 'dashed', 'dotted',
                                            'dashdot', (0, (3, 1, 1, 1)), (0, (5, 2))]
    contrasts["word_cats"]["epoch_times"] = [-0.3, 3.5]
    
    
    #%% Contrasts for the "sentences" experiment
    
    contrasts["sentence"] = {}
    contrasts["sentence"]["queries"] = ["event_type=='sentence_onset'"]
    contrasts["sentence"]["labels"] = ["sentence"]
    contrasts["sentence"]["colors"] = ["b"]
    contrasts["sentence"]["lines"] = ['solid']
    contrasts["sentence"]["epoch_times"] = [-0.1, 12]
    
    ##
    
    contrasts["sentence_fixation"] = {}
    contrasts["sentence_fixation"]["queries"] = ["event_type=='sentence_onset'",
                                                 "event_type=='fixation_cross'"]
    contrasts["sentence_fixation"]["labels"] = ["sentence",
                                                "fixation"]
    contrasts["sentence_fixation"]["colors"] = ["#FB9A99", "#A6CEE3"]
    contrasts["sentence_fixation"]["lines"] = ['solid', 'dashed']
    contrasts["sentence_fixation"]["epoch_times"] = [-0.1, 12]
    
    ##
    
    contrasts["predictability"] = {}
    contrasts["predictability"]["queries"] = ["event_type=='sentence_onset' and predictable==1",
                                              "event_type=='sentence_onset' and predictable==0"]
    contrasts["predictability"]["labels"] = ["predictable",
                                             "unpredictable"]
    contrasts["predictability"]["colors"] = ["#FB9A99", "#A6CEE3"]
    contrasts["predictability"]["lines"] = ['solid', 'dashed']
    contrasts["predictability"]["epoch_times"] = [-0.1, 12]
    
    ## 
    
    contrasts["predictable"] = {}
    contrasts["predictable"]["queries"] = ["event_type=='sentence_onset' and predictable==1"]
    contrasts["predictable"]["labels"] = ["predictable"]
    contrasts["predictable"]["colors"] = ["b"]
    contrasts["predictable"]["lines"] = ['solid']
    contrasts["predictable"]["epoch_times"] = [-0.1, 12]
    
    ## 
    
    contrasts["unpredictable"] = {}
    contrasts["unpredictable"]["queries"] = ["event_type=='sentence_onset' and predictable==0"]
    contrasts["unpredictable"]["labels"] = ["predictable"]
    contrasts["unpredictable"]["colors"] = ["b"]
    contrasts["unpredictable"]["lines"] = ['solid']
    contrasts["unpredictable"]["epoch_times"] = [-0.1, 12]
    
    return contrasts

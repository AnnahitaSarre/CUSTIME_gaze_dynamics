%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% CUSTIME stimulation script: Displays 270 (pseudo)words in the LfPC form %%%%%%%%%
%%%%%%%%       Sends triggers to an EEG and an Eyelink 1000 eye-tracking         %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% PsychtoolboxVersion: In CENIR: '3.0.18 - Flavor: beta - Corresponds to SVN Revision 12737' /
% In ICM-COHEN-LF003: '3.0.17 - Flavor: Manual Install, 20-fevrier.-2021 07:52:56'

% Gstreamer: In CENIR: 1.16.2 /In ICM-COHEN-LF003: 1.18.2

% screen_distance = 80;
% screen_width = 2560; % In px ICM-COHEN-LF003: 1920
% screen_height = 1440; % In px ICM-COHEN-LF003: 1080

% Nb of runs: 2
% To be displayed to LfPC proficient users

clear
clc


%% Parameters to modify if needed

% Stimuli parameters
nb_words_in_exp = 270; % Nb of words displayed during the entire experiment (to be splitted into several runs) % Default = 270 (all words)
spacing_of_rests = 0; % Nb of word displays between rest periods % Default = 0
nb_runs = 2; % Nb of runs that will be runned in the subject's session, with the words to be distributed among them % Default = 2

% Timing parameters
% One constraint: response_interval_max <= video_duration + ISI
fixation_duration_min = 0.3; % In seconds % Default: 0.3
fixation_duration_max = 0.9; % In seconds % Default: 0.9
isi = 0.8; % In seconds; Interstimulus interval, aka the blank interval between two trials % Default: 0.8
response_interval_min = 0.2; % In seconds; Min response interval after the onset of the video; If a mouse button is clicked after the pseudo but before this delay , the response is considered invalid
response_interval_max = 0.8; % In seconds; Max response interval after the end of the video; Should not be longer than the ISI!
rest_duration = 3; % In seconds % Default = 3

% Fixation cross display parameters
fixation_dim = 20; % In px; Here we set the size of the arms of the fixation cross % Default: 20
fixation_width = 4; % In px; Set the line width for the fixation cross
fixation_line_x = [-fixation_dim fixation_dim 0 0]; % Set the coordinates of the two lines (these are all relative to zero we will let the drawing routine center the cross in the center of our monitor for us)
fixation_line_y = [0 0 -fixation_dim fixation_dim];
fixation_color = [128 128 128];
fixation_position_x = -10; % Position relative to the screen center; Should be between the two eyes
fixation_position_y = -200; % Position relative to the screen center; Should be between the two eyes

% Video display parameters
video_width = 2560; % In px
video_height = 1440; % In px

% Other parameters
dummymode = 0; % set to 1 to run eye-tracking in dummymode (using mouse as pseudo-eyetracker)


%% Collect basic info

if strcmp(getenv('COMPUTERNAME'), 'STIM-EEG2')
    screen_id = 2; % Default = 2
else
    screen_id = max(Screen('Screens')); % Open onscreen window. We use the display with the highest number on multi-display setups
end

main_path = pwd; % current path
date_of_day = datestr(now, 'yyyymmdd'); % date of the day

subject_id = input('Numero de sujet : ');
num_run = input('Numero de run : ');

fprintf('##### Subject ID: %d; Run number: %d\n', subject_id, num_run);


%% Create subject's directories and eye-tracking file

results_path_beh = fullfile(main_path, sprintf('/custime_results/sub-S%d/ses-%08s/beh', subject_id, date_of_day));
mkdir(results_path_beh);

results_path_et = fullfile(main_path, sprintf('/custime_results/sub-S%d/ses-%08s/beh', subject_id, date_of_day));
mkdir(results_path_et);
file_name_et = fullfile(results_path_et, sprintf('sub-S%d_ses-%014s_task-WordsUsers_run-%01d.edf', subject_id, date_of_day, num_run));


%% Keyboard and screen parameters

AssertOpenGL; % Childdgfd ethy protection: Make sure we run on the OSX / OpenGL Psychtoolbox. Abort if we don't

KbName('UnifyKeyNames'); % Switch KbName into unified mode: It will use the names of the OS-X platform on all platforms in order to make this script portable
esc = KbName('ESCAPE'); % Query keycode for escape key
t = KbName('t'); % Query keycode for 't' key

exp_term = 0; % Flag for exiting the script

screen_background = [177, 156, 120];

Screen('Preference', 'SkipSyncTests', 1);
Screen('Preference', 'SuppressAllWarnings', 1);

[win, window_rect] = PsychImaging('OpenWindow', screen_id, screen_background); % Open an on screen window with background color 'screen_background'

[screen_center_x, screen_center_y] = RectCenter(window_rect); % In px % Get the centre coordinate of the window

video_x_left = screen_center_x-ceil(video_width/2);
video_y_left = screen_center_y-ceil(video_height/2);
video_x_right = screen_center_x+ceil(video_width/2);
video_y_right = screen_center_y+ceil(video_height/2);
video_rect = [video_x_left video_y_left video_x_right video_y_right]; % In px

% HideCursor; % Comment it when testing

% Priority(MaxPriority(win)); % Comment it when testing


%% Define the list of words to be displayed in each of the subject's runs

rng_struct = rng('shuffle'); % Initialize the random generator

file_custime_words_stim = fullfile(main_path, '/custime_stim_lists/custime_words_stim_users.mat');
file_custime_words_stim_subject = fullfile(main_path, sprintf('/custime_stim_lists/custime_words_stim_subjects/custime_words_stim_%d.mat', subject_id));
if ~isfile(file_custime_words_stim_subject) % Create a stim mat only if this hasn't be done yet for the considered subject
    nb_words_per_cat_in_run_pseudo = floor(nb_words_in_exp/6*0.1/nb_runs); % 6 = nb of mismatch-frequency categories ; 0.1 = proportion of pseudowords per mismatch-frequency category
    nb_words_per_cat_in_run_nonpseudo = floor((nb_words_in_exp - nb_words_in_exp*0.1)/6/nb_runs);
    copyfile(file_custime_words_stim, file_custime_words_stim_subject);
    load(file_custime_words_stim_subject);
    custime_words_stim = custime_words_stim_users(randperm(nb_words_in_exp));
    clear('custime_words_stim_users');
    for i_run=1:nb_runs
        nb_in_cat_nonpseudo = 0;
        nb_in_cat_pseudo = 0;
        for i_word=1:nb_words_in_exp % Choose the words for the considered category, for pseudo and nonpseudo versions
            if isempty(custime_words_stim(i_word).run) && custime_words_stim(i_word).freq == 'h' && custime_words_stim(i_word).mismatch == 0
                if nb_in_cat_nonpseudo < nb_words_per_cat_in_run_nonpseudo && ~custime_words_stim(i_word).pseudo
                    custime_words_stim(i_word).run = i_run;
                    nb_in_cat_nonpseudo = nb_in_cat_nonpseudo + 1;
                end
                if nb_in_cat_pseudo < nb_words_per_cat_in_run_pseudo && custime_words_stim(i_word).pseudo
                    custime_words_stim(i_word).run = i_run;
                    nb_in_cat_pseudo = nb_in_cat_pseudo + 1;
                end
            end
        end
        nb_in_cat_nonpseudo = 0;
        nb_in_cat_pseudo = 0;
        for i_word=1:nb_words_in_exp % Choose the words for the considered category, for pseudo and nonpseudo versions
            if isempty(custime_words_stim(i_word).run) && custime_words_stim(i_word).freq == 'h' && custime_words_stim(i_word).mismatch == 1
                if nb_in_cat_nonpseudo < nb_words_per_cat_in_run_nonpseudo && ~custime_words_stim(i_word).pseudo
                    custime_words_stim(i_word).run = i_run;
                    nb_in_cat_nonpseudo = nb_in_cat_nonpseudo + 1;
                end
                if nb_in_cat_pseudo < nb_words_per_cat_in_run_pseudo && custime_words_stim(i_word).pseudo
                    custime_words_stim(i_word).run = i_run;
                    nb_in_cat_pseudo = nb_in_cat_pseudo + 1;
                end
            end
        end
        nb_in_cat_nonpseudo = 0;
        nb_in_cat_pseudo = 0;
        for i_word=1:nb_words_in_exp % Choose the words for the considered category, for pseudo and nonpseudo versions
            if isempty(custime_words_stim(i_word).run) && custime_words_stim(i_word).freq == 'h' && custime_words_stim(i_word).mismatch == 2
                if nb_in_cat_nonpseudo < nb_words_per_cat_in_run_nonpseudo && ~custime_words_stim(i_word).pseudo
                    custime_words_stim(i_word).run = i_run;
                    nb_in_cat_nonpseudo = nb_in_cat_nonpseudo + 1;
                end
                if nb_in_cat_pseudo < nb_words_per_cat_in_run_pseudo && custime_words_stim(i_word).pseudo
                    custime_words_stim(i_word).run = i_run;
                    nb_in_cat_pseudo = nb_in_cat_pseudo + 1;
                end
            end
        end
        nb_in_cat_nonpseudo = 0;
        nb_in_cat_pseudo = 0;
        for i_word=1:nb_words_in_exp % Choose the words for the considered category, for pseudo and nonpseudo versions
            if isempty(custime_words_stim(i_word).run) && custime_words_stim(i_word).freq == 'l' && custime_words_stim(i_word).mismatch == 0
                if nb_in_cat_nonpseudo < nb_words_per_cat_in_run_nonpseudo && ~custime_words_stim(i_word).pseudo
                    custime_words_stim(i_word).run = i_run;
                    nb_in_cat_nonpseudo = nb_in_cat_nonpseudo + 1;
                end
                if nb_in_cat_pseudo < nb_words_per_cat_in_run_pseudo && custime_words_stim(i_word).pseudo
                    custime_words_stim(i_word).run = i_run;
                    nb_in_cat_pseudo = nb_in_cat_pseudo + 1;
                end
            end
        end
        nb_in_cat_nonpseudo = 0;
        nb_in_cat_pseudo = 0;
        for i_word=1:nb_words_in_exp % Choose the words for the considered category, for pseudo and nonpseudo versions
            if isempty(custime_words_stim(i_word).run) && custime_words_stim(i_word).freq == 'l' && custime_words_stim(i_word).mismatch == 1
                if nb_in_cat_nonpseudo < nb_words_per_cat_in_run_nonpseudo && ~custime_words_stim(i_word).pseudo
                    custime_words_stim(i_word).run = i_run;
                    nb_in_cat_nonpseudo = nb_in_cat_nonpseudo + 1;
                end
                if nb_in_cat_pseudo < nb_words_per_cat_in_run_pseudo && custime_words_stim(i_word).pseudo
                    custime_words_stim(i_word).run = i_run;
                    nb_in_cat_pseudo = nb_in_cat_pseudo + 1;
                end
            end
        end
        nb_in_cat_nonpseudo = 0;
        nb_in_cat_pseudo = 0;
        for i_word=1:nb_words_in_exp % Choose the words for the considered category, for pseudo and nonpseudo versions
            if isempty(custime_words_stim(i_word).run) && custime_words_stim(i_word).freq == 'l' && custime_words_stim(i_word).mismatch == 2
                if nb_in_cat_nonpseudo < nb_words_per_cat_in_run_nonpseudo && ~custime_words_stim(i_word).pseudo
                    custime_words_stim(i_word).run = i_run;
                    nb_in_cat_nonpseudo = nb_in_cat_nonpseudo + 1;
                end
                if nb_in_cat_pseudo < nb_words_per_cat_in_run_pseudo && custime_words_stim(i_word).pseudo
                    custime_words_stim(i_word).run = i_run;
                    nb_in_cat_pseudo = nb_in_cat_pseudo + 1;
                end
            end
        end
    end
    for i_word=1:nb_words_in_exp % If a word is not assigned a run, assign it a run randomly
        if isempty(custime_words_stim(i_word).run)
            custime_words_stim(i_word).run = randi(nb_runs);
        end
    end
    save(file_custime_words_stim_subject, 'custime_words_stim');
end


%% Generate the order of trials and create the log structure

% Load the subject's list of stimuli and count the nb of words to be displayed during this run
load(file_custime_words_stim_subject);
nb_words_in_run = 0;
for i_word=1:size(custime_words_stim,1)
    if custime_words_stim(i_word).run == num_run
        nb_words_in_run = nb_words_in_run + 1;
    end
end

% Initialize the struct log
log_structure(nb_words_in_run) = struct('subject', [], 'run', [], 'word', [], 'num_in_cat', [], 'freq', [], 'mismatch', [], 'pseudo', [], ...
    'video', [], 'word_onset_time', [], 'word_marker', [], ...
    'click_time', [], 'click_marker', [], 'response_time', [], 'response_validity', [], 'response_type', [], ...
    'fixation_onset_time', [], 'fixation_duration', [], 'fixation_marker', []);

% Append the words and their information to the structure log for the considered run
num_row_struct_exp = 1;
i_trial = 1;
while num_row_struct_exp <= nb_words_in_exp && i_trial <= nb_words_in_run
    if  isequal(custime_words_stim(num_row_struct_exp).run,num_run)
        log_structure(i_trial).run = custime_words_stim(num_row_struct_exp).run;
        log_structure(i_trial).word = custime_words_stim(num_row_struct_exp).word;
        log_structure(i_trial).num_in_cat = custime_words_stim(num_row_struct_exp).num_in_cat;
        log_structure(i_trial).freq = custime_words_stim(num_row_struct_exp).freq;
        log_structure(i_trial).mismatch = custime_words_stim(num_row_struct_exp).mismatch;
        log_structure(i_trial).pseudo = custime_words_stim(num_row_struct_exp).pseudo;
        i_trial = i_trial + 1;
    end
    num_row_struct_exp = num_row_struct_exp + 1;
end
clear('custime_words_stim');

% Randomize trials, making sure two pseudowords are not displayed in a row
while 1
    invalid_target = 0; % Two pseudowords in a row
    log_structure = log_structure(randperm(nb_words_in_run));
    for i_trial=2:nb_words_in_run
        if log_structure(i_trial-1).pseudo && log_structure(i_trial).pseudo
            invalid_target = 1;
            break;
        end
    end
    if ~invalid_target
        break;
    end
end

% Generate a list of the videos to be displayed
list_videos = {};
for i_trial=1:nb_words_in_run
        list_videos(i_trial) = {fullfile(main_path,'custime_videos','custime_videos_words',sprintf('word_%s%d_%02d',log_structure(i_trial).freq,log_structure(i_trial).mismatch,log_structure(i_trial).num_in_cat))};
        if log_structure(i_trial).pseudo
            list_videos(i_trial) = strcat(list_videos(i_trial), '_pseudo.mp4');
        else
            list_videos(i_trial) = strcat(list_videos(i_trial), '.mp4');
        end
end

% Create a fixation duration for each trial
fixation_duration_list = linspace(fixation_duration_min, fixation_duration_max, nb_words_in_run);
fixation_duration_list = fixation_duration_list(randperm(nb_words_in_run));

% Determine the trials after which a rest period will happen
if spacing_of_rests > 0
    list_rests = [];
    rest_range = 1;
    while spacing_of_rests * rest_range < nb_words_in_run
        list_rests = [list_rests, spacing_of_rests * rest_range];
        rest_range = rest_range + 1;
    end
end

% Append other trial info to the struct log
for i_trial=1:nb_words_in_run
    log_structure(i_trial).subject = subject_id;
    log_structure(i_trial).video = list_videos(i_trial);
    log_structure(i_trial).fixation_duration = fixation_duration_list(i_trial);
end


%% EEG and eye-tracking initialization

%OpenParPort(64); % Open the port for the EEG triggers

if strcmp(getenv('COMPUTERNAME'), 'STIM-EEG2')
    if ~EyelinkInit(dummymode, 1)
     fprintf('####### Eyelink Init aborted.\n');
     return
    end
    [v, vs]=Eyelink('GetTrackerVersion');
    fprintf('##### Running experiment on a ''%s'' tracker.\n', vs);
    el = EyelinkInitDefaults(win);
    el.backgroundcolour = screen_background;
    EyelinkUpdateDefaults(el);
    delete('custime.edf');
end


%% Show eye-tracking instructions and launch the eye-tracking calibration task

eye_instructions_size = 45;
Screen('TextSize', win, eye_instructions_size);
eye_instructions_text = 'Calibration pour le suivi des mouvements occulaires : Suivez le point du regard.';
eye_instructions_width = RectWidth(Screen('TextBounds',win,eye_instructions_text));
eye_instructions_height = RectHeight(Screen('TextBounds',win,eye_instructions_text));
eye_instructions_x = screen_center_x-ceil(eye_instructions_width/2);
eye_instructions_y = screen_center_y-ceil(eye_instructions_height);
Screen('DrawText', win, eye_instructions_text, eye_instructions_x, eye_instructions_y);

eye_continue_size = 25;
Screen('TextSize', win, eye_continue_size);
eye_continue_text = 'Cliquez sur la souris pour continuer';
eye_continue_width = RectWidth(Screen('TextBounds',win,eye_continue_text));
eye_continue_height = RectHeight(Screen('TextBounds',win,eye_continue_text));
eye_continue_x = screen_center_x-ceil(eye_continue_width/2);
eye_continue_y = screen_center_y-ceil(eye_continue_height/2)+50;
Screen('DrawText', win, eye_continue_text, eye_continue_x, eye_continue_y);

Screen('Flip',win);

% Wait for mouse (or escape key) press
fprintf('##### Showing eye-tracking instructions, waiting for mouse press... \n');  
while 1
    [keyIsDown, secs, keyCode] = KbCheck; % Keyboard check
    [x,y,buttons] = GetMouse; % Mouse check
    if any(buttons)
        fprintf('## Mouse was pressed \n');
        break;
    end
    if keyIsDown
        if keyCode(esc)
            exp_term = 1;
            fprintf('####### Escape key was pressed \n');
            break;
        end
    end
end

if exp_term
    Priority(0);
    ShowCursor;
    fprintf('##### Saving struct log file... \n');
    date_of_run = datestr(now, 'yyyymmddHHMMSS');
    file_name_beh = fullfile(results_path_beh, sprintf('sub-S%d_ses-%014s_task-WordsUsers_run-%01d_beh.csv', subject_id, date_of_run, num_run));
    writetable(struct2table(log_structure), file_name_beh);
    fprintf('##### Struct log file saved \n');
    Screen('CloseAll');
    return;
end

if strcmp(getenv('COMPUTERNAME'), 'STIM-EEG2')
    EyelinkDoTrackerSetup(el); % Eye-tracking calibration task
    WaitSecs(.5); % Wait so that it doesn't escape the program if the esc key was pressed in EyelinkDoTrackerSetup
end


%% Show task instructions

task_instructions_size = 45;
Screen('TextSize', win, task_instructions_size);
task_instructions_text = 'Cliquez sur la souris quand le mot que vous percevez n''existe pas';
task_instructions_width = RectWidth(Screen('TextBounds',win,task_instructions_text));
task_instructions_height = RectHeight(Screen('TextBounds',win,task_instructions_text));
task_instructions_x = screen_center_x-ceil(task_instructions_width/2);
task_instructions_y = screen_center_y-ceil(task_instructions_height);
Screen('DrawText', win, task_instructions_text, task_instructions_x, task_instructions_y);

task_continue_size = 25;
Screen('TextSize', win, task_continue_size);
task_continue_text = 'Demarrage de l''enregistrement EEG...';
task_continue_width = RectWidth(Screen('TextBounds',win,task_continue_text));
task_continue_height = RectHeight(Screen('TextBounds',win,task_continue_text));
task_continue_x = screen_center_x-ceil(task_continue_width/2);
task_continue_y = screen_center_y-ceil(task_continue_height/2)+50;
Screen('DrawText', win, task_continue_text, task_continue_x, task_continue_y);

Screen('Flip',win);

% Wait for 't' button (or escape key) press
fprintf('##### Showing task instructions.\n');
fprintf('##### Start the EEG recording, wait for its stabilization and press ''t'' to continue.\n');
while 1
    [keyIsDown, secs, keyCode] = KbCheck; % Keyboard check
    if keyIsDown
        if keyCode(esc)
            exp_term = 1;
            fprintf('####### Escape key was pressed \n');
            break;
        end
        if keyCode(t)
            fprintf('## ''t'' button pressed \n');
            break;
        end
    end
end

if exp_term
    Priority(0);
    ShowCursor;
    fprintf('##### Saving struct log file... \n');
    date_of_run = datestr(now, 'yyyymmddHHMMSS');
    file_name_beh = fullfile(results_path_beh, sprintf('sub-S%d_ses-%014s_task-WordsUsers_run-%01d_beh.csv', subject_id, date_of_run, num_run));
    writetable(struct2table(log_structure), file_name_beh);
    fprintf('##### Struct log file saved \n');
    Screen('CloseAll');
    return;
end

if strcmp(getenv('COMPUTERNAME'), 'STIM-EEG2')
    Eyelink('Openfile', 'custime.edf');
    fprintf('##### Eye-tracking file opened. \n');
    Eyelink('StartRecording');
    fprintf('##### Eye-tracking started recording. \n');
end

Screen('Flip',win);


%% Display loop

date_of_run = datestr(now, 'yyyymmddHHMMSS');
t0 = GetSecs;
last_time = t0;

fprintf('##### Entering trial loop... \n');
    
for i_trial=1:nb_words_in_run

    % First fixation cross preparation while blank interval display
    last_time = Screen('Flip',win); % Display the blank interval
    Screen('DrawLines', win, [fixation_line_x; fixation_line_y], fixation_width, fixation_color, [screen_center_x+fixation_position_x screen_center_y+fixation_position_y]);
    while GetSecs < last_time + isi
        [keyIsDown, secs, keyCode] = KbCheck; % Keyboard check
        [x,y,buttons] = GetMouse; % Mouse check
        if any(buttons) && (i_trial~=1) && isempty(log_structure(i_trial-1).click_time) % Will not take buttons in the first trial (avoids problems with non existant log_structure(i_trial-1))
            last_mouse_click = GetSecs - t0;
            fprintf('   ## Mouse was pressed \n');
            if (log_structure(i_trial-1).pseudo) && (last_mouse_click > log_structure(i_trial-1).word_onset_time + response_interval_min) && (last_mouse_click < video_end + response_interval_max)
                SendTrigger(trigger_resp_val);
                Eyelink('Message', sprintf('S%3d', trigger_resp_val));
                log_structure(i_trial-1).click_time = last_mouse_click; % This response is for the word that was just displayed, aka the previous trial
                log_structure(i_trial-1).click_marker = trigger_resp_val;
                log_structure(i_trial-1).response_time = log_structure(i_trial-1).click_time - log_structure(i_trial-1).word_onset_time;
                log_structure(i_trial-1).response_validity = 1;
                log_structure(i_trial-1).response_type = 'HIT'; % Hit
                fprintf('   Hit on trial %d/%d! Response time: %ds\n', i_trial-1, nb_words_in_run, log_structure(i_trial-1).response_time);
            else
                SendTrigger(trigger_resp_val);
                Eyelink('Message', sprintf('S%3d', trigger_resp_val));
                log_structure(i_trial-1).click_time = last_mouse_click;
                log_structure(i_trial-1).click_marker = trigger_resp_val;
                log_structure(i_trial-1).response_time = log_structure(i_trial-1).click_time - log_structure(i_trial-1).word_onset_time;
                log_structure(i_trial-1).response_validity = 0;
                log_structure(i_trial-1).response_type = 'FA'; % False alarm
                fprintf('   False alarm on trial %d/%d...\n', i_trial-1, nb_words_in_run);
            end
        end
        if keyIsDown
            if keyCode(esc)
                exp_term = 1;
                fprintf('####### Escape key was pressed \n');
                break;
            end
        end
    end
    if exp_term
        break;
    end
    
    % First fixation cross info saving while first fixation cross display
    last_time = Screen('Flip', win);
    SendTrigger(1);
    Eyelink('Message', 'S  1');
    log_structure(i_trial).fixation_onset_time = last_time - t0;
    log_structure(i_trial).fixation_marker = 1;
    
    % Video and trigger preparation while first fixation cross display
    video_name = list_videos{i_trial};
    word_video = Screen('OpenMovie', win, video_name);
    Screen('PlayMovie', word_video, 1);
    first_image = 1;
    trigger_mismatch = log_structure(i_trial).mismatch;
    if log_structure(i_trial).freq == 'l'
        trigger_freq = 4;
    elseif log_structure(i_trial).freq == 'h'
        trigger_freq = 8;
    end
    if log_structure(i_trial).pseudo
        trigger_pseudo = 64;
    else
        trigger_pseudo = 0;
    end
    trigger_val =  trigger_mismatch + trigger_freq + trigger_pseudo;
    trigger_resp_val = trigger_val + 128;
    while GetSecs < last_time + log_structure(i_trial).fixation_duration
        [keyIsDown, secs, keyCode] = KbCheck; % Keyboard check
        if keyIsDown
            if keyCode(esc)
                exp_term = 1;
                fprintf('####### Escape key was pressed \n');
                break;
            end
        end
    end
    if exp_term
        break;
    end
    if (i_trial~=1) && (GetSecs - t0 > video_end + response_interval_max) % Take care one last time of response concerning for the previous trial, before moving on to the next word
        if isempty(log_structure(i_trial-1).click_time)
            if log_structure(i_trial-1).pseudo
                log_structure(i_trial-1).response_validity = 0;
                log_structure(i_trial-1).response_type = 'MISS'; % Miss
                fprintf('   Miss on trial %d/%d...\n', i_trial-1, nb_words_in_run);
            else
                log_structure(i_trial-1).response_validity = 1;
                log_structure(i_trial-1).response_type = 'CR'; % Correct rejection
            end
        end
    elseif (i_trial~=1) && (GetSecs - t0 < video_end + response_interval_max)
        fprintf('##### Warning: Word displayed before the end of response interval of the previous one! #####\n');
    end
    
    % Word info saving while word video display
    while 1
        tex = Screen('GetMovieImage', win, word_video); % Wait for next movie frame, retrieve texture handle to it
        if tex<=0 % Valid texture returned? A negative value means end of movie reached
            break;
        end
        [keyIsDown, secs, keyCode] = KbCheck;
        [x,y,buttons] = GetMouse;
        if any(buttons) && (i_trial~=1) && isempty(log_structure(i_trial).click_time) % This time, this concerns the word that is displayed at this very moment
            last_mouse_click = GetSecs - t0;
            fprintf('   ## Mouse was pressed \n');
            if (log_structure(i_trial).pseudo) && (last_mouse_click > log_structure(i_trial).word_onset_time + response_interval_min)
                SendTrigger(trigger_resp_val);
                Eyelink('Message', sprintf('S%3d', trigger_resp_val));
                log_structure(i_trial).click_time = last_mouse_click;
                log_structure(i_trial).click_marker = trigger_resp_val;
                log_structure(i_trial).response_time = log_structure(i_trial).click_time - log_structure(i_trial).word_onset_time;
                log_structure(i_trial).response_validity = 1;
                log_structure(i_trial).response_type = 'HIT'; % Hit
                fprintf('   Hit on trial %d/%d! Response time: %ds\n', i_trial, nb_words_in_run, log_structure(i_trial).response_time);
            else
                SendTrigger(trigger_resp_val);
                Eyelink('Message', sprintf('S%3d', trigger_resp_val));
                log_structure(i_trial).click_time = last_mouse_click;
                log_structure(i_trial).click_marker = trigger_resp_val;
                log_structure(i_trial).response_time = log_structure(i_trial).click_time - log_structure(i_trial).word_onset_time;
                log_structure(i_trial).response_validity = 0;
                log_structure(i_trial).response_type = 'FA'; % False alarm
                fprintf('   False alarm on trial %d/%d...\n', i_trial, nb_words_in_run);
            end
        end
        if keyIsDown
            if keyCode(esc)
                exp_term = 1;
                fprintf('####### Escape key was pressed \n');
                break;
            end
        end
        Screen('DrawTexture', win, tex, [], video_rect); % Draw the new texture immediately to screen
        last_time = Screen('Flip', win); % Update display
        if first_image
            SendTrigger(trigger_val);
            Eyelink('Message', sprintf('S%3d', trigger_val));
            first_image = 0;
            log_structure(i_trial).word_onset_time = last_time - t0; % We need to put this info before looking for responses...
            log_structure(i_trial).word_marker = trigger_val;
            fprintf('# Displaying trial %d/%d\n', i_trial, nb_words_in_run);
        end
        Screen('Close', tex); % Release texture
    end
    Screen('PlayMovie', word_video, 0); % Stop playback
    Screen('CloseMovie', word_video); % Close movie
    video_end = GetSecs - t0;
    
    % Second fixation cross preparation and display
    Screen('DrawLines', win, [fixation_line_x; fixation_line_y], fixation_width, fixation_color, [screen_center_x+fixation_position_x screen_center_y+fixation_position_y]);
    last_time = Screen('Flip', win);
    while GetSecs < last_time + log_structure(i_trial).fixation_duration
        [keyIsDown, secs, keyCode] = KbCheck; % Keyboard check
        [x,y,buttons] = GetMouse; % Mouse check
        if any(buttons) && isempty(log_structure(i_trial).click_time)
            last_mouse_click = GetSecs - t0;
            fprintf('   ## Mouse was pressed \n');
            if (log_structure(i_trial).pseudo) && (last_mouse_click > log_structure(i_trial).word_onset_time + response_interval_min) && (last_mouse_click < video_end + response_interval_max)
                SendTrigger(trigger_resp_val);
                Eyelink('Message', sprintf('S%3d', trigger_resp_val));
                log_structure(i_trial).click_time = last_mouse_click; % This response is for the word that was just displayed, aka the previous trial
                log_structure(i_trial).click_marker = trigger_resp_val;
                log_structure(i_trial).response_time = log_structure(i_trial).click_time - log_structure(i_trial).word_onset_time;
                log_structure(i_trial).response_validity = 1;
                log_structure(i_trial).response_type = 'HIT'; % Hit
                fprintf('   Hit on trial %d/%d! Response time: %ds\n', i_trial, nb_words_in_run, log_structure(i_trial).response_time);
            else
                SendTrigger(trigger_resp_val);
                Eyelink('Message', sprintf('S%3d', trigger_resp_val));
                log_structure(i_trial).click_time = last_mouse_click;
                log_structure(i_trial).click_marker = trigger_resp_val;
                log_structure(i_trial).response_time = log_structure(i_trial).click_time - log_structure(i_trial).word_onset_time;
                log_structure(i_trial).response_validity = 0;
                log_structure(i_trial).response_type = 'FA'; % False alarm
                fprintf('   False alarm on trial %d/%d...\n', i_trial, nb_words_in_run);
            end
        end
        if keyIsDown
            if keyCode(esc)
                exp_term = 1;
                fprintf('####### Escape key was pressed \n');
                break;
            end
        end
    end
    if exp_term
        break;
    end
    if spacing_of_rests > 0 % Implement the rest periods
        if ismember(i_trial, list_rests)
            fprintf('## Entering rest period... \n');
            WaitSecs(rest_duration);
            fprintf('## Exiting rest period. \n');
        end
    end
end

% After the last word display (and if the exp was not aborted), display the blank one last time and collect last trial info
if ~exp_term
    last_time = Screen('Flip',win);
    while GetSecs < last_time + isi
        [keyIsDown, secs, keyCode] = KbCheck; % Keyboard check
        [x,y,buttons] = GetMouse; % Mouse check
        if any(buttons) && isempty(log_structure(i_trial).click_time)
            last_mouse_click = GetSecs - t0;
            fprintf('   ## Mouse was pressed \n');
            if (log_structure(i_trial).pseudo) && (last_mouse_click > log_structure(i_trial).word_onset_time + response_interval_min) && (last_mouse_click < end_video + response_interval_max)
                SendTrigger(trigger_resp_val);
                Eyelink('Message', sprintf('S%3d', trigger_resp_val));
                log_structure(i_trial).click_time = last_mouse_click;
                log_structure(i_trial).click_marker = trigger_resp_val;
                log_structure(i_trial).response_time = log_structure(i_trial).click_time - log_structure(i_trial).word_onset_time;
                log_structure(i_trial).response_validity = 1;
                log_structure(i_trial).response_type = 'HIT'; % Hit
                fprintf('   Hit on trial %d/%d! Response time: %ds\n', i_trial, nb_words_in_run, log_structure(i_trial).response_time);
            else
                SendTrigger(trigger_resp_val);
                Eyelink('Message', sprintf('S%3d', trigger_resp_val));
                log_structure(i_trial).click_time = last_mouse_click;
                log_structure(i_trial).click_marker = trigger_resp_val;
                log_structure(i_trial).response_time = log_structure(i_trial).click_time - log_structure(i_trial).word_onset_time;
                log_structure(i_trial).response_validity = 0;
                log_structure(i_trial).response_type = 'FA'; % False alarm
                fprintf('  False alarm on trial %d/%d...\n', i_trial, nb_words_in_run);
            end
        end
        if keyIsDown
            if keyCode(esc)
                exp_term = 1;
                fprintf('####### Escape key was pressed \n');
                break;
            end
        end
    end
    if isempty(log_structure(i_trial).click_time)
        if log_structure(i_trial).pseudo
            log_structure(i_trial).response_validity = 0;
            log_structure(i_trial).response_type = 'MISS'; % Miss
            fprintf('   Miss on trial %d/%d...\n', i_trial, nb_words_in_run);
        else
            log_structure(i_trial).response_validity = 1;
            log_structure(i_trial).response_type = 'CR'; % Correct rejection
        end
    end

    fprintf('##### All trials displayed \n');
end

end_of_run_timestamp = GetSecs;
run_duration = end_of_run_timestamp - date_of_run;


%% Save struct log and eye-tracking files

fprintf('##### Saving struct log file... \n');
file_name_beh = fullfile(results_path_beh, sprintf('sub-S%d_ses-%014s_task-WordsUsers_run-%01d_beh.csv', subject_id, date_of_run, num_run));
writetable(struct2table(log_structure), file_name_beh);
fprintf('##### Struct log file saved \n');

if strcmp(getenv('COMPUTERNAME'), 'STIM-EEG2')
    fprintf('##### Stopping the eye-tracking recording and closing the edf file... \n');
    Eyelink('StopRecording');
    Eyelink('CloseFile');
    for i = 1:10
        status = Eyelink('ReceiveFile');
        if status >= 0
            fprintf('##### Eye-tracking recording stopped and edf file closed.\n');
            movefile('custime.edf', file_name_et);
            break
        end
    end
    if status < 0
        warning('Could not retrieve eyetracker file');
    end
end


%% Close onscreen window and finish

if strcmp(getenv('COMPUTERNAME'), 'STIM-EEG2')
    fprintf('##### Shuting down the eye-tracking... \n');
    Eyelink('ShutDown');
    fprintf('##### Eye-tracking shut down. \n');
end

ShowCursor;
Screen('CloseAll');
sca;
fprintf('##### Script execution has finished correctly \n')
    
return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% CUSTIME stimulation script: Displays 100 sentences in the LfPC form     %%%%%%%%%
%%%%%%%%        Sends triggers to an EEG and an Eyelink 1000 eye-tracking        %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% PsychtoolboxVersion: In CENIR: '3.0.18 - Flavor: beta - Corresponds to SVN Revision 12737' /
% In ICM-COHEN-LF003: '3.0.17 - Flavor: Manual Install, 20-fevrier.-2021 07:52:56'

% Gstreamer: In CENIR: 1.16.2 /In ICM-COHEN-LF003: 1.18.2

% screen_distance = 80; % In cm CHANGE THAT
% screen_width = 2560; % In px ICM-COHEN-LF003: 1920
% screen_height = 1440; % In px ICM-COHEN-LF003: 1080

% Nb of runs: 2
% To be displayed to LfPC proficient users

clear
clc


%% Parameters to modify if needed

% Stimuli parameters
nb_sentences_in_exp = 100; % Nb of sentences displayed during the entire experiment (to be splitted into several runs % Default = 100 (all sentences)
nb_questions_in_exp = 8; % Nb of questions asked during the entire experiment % Default = 8 (must be an even number !)
spacing_of_rests = 0; % Nb of sentence displays between rest periods % Default = 0
nb_runs = 2; % Nb of runs that will be runned in the subject's session, with the sentences to be distributed among them % Default = 2 (doesn't run with a different number)

% Timing parameters
% One constraint: response_interval_max <= video_duration + ISI
fixation_duration_min = 0.3; % In seconds % Default: 0.3
fixation_duration_max = 0.9; % In seconds % Default: 0.9
isi = 0.8; % In seconds; Interstimulus interval, aka the blank interval between two trials % Default: 0.8
rest_duration = 3; % In seconds % Default = 3

% Fixation cross display parameters
fixation_dim = 20; % In px; Here we set the size of the arms of the fixation cross % Default: 20
fixation_width = 4; % In px; Set the line width for the fixation cross
fixation_line_x = [-fixation_dim fixation_dim 0 0]; % Set the coordinates of the two lines (these are all relative to zero we will let the drawing routine center the cross in the center of our monitor for us)
fixation_line_y = [0 0 -fixation_dim fixation_dim];
fixation_color = [128 128 128];
fixation_position_x = 0; % Position relative to the screen center; Should be between the two eyes
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

fprintf('\n##### Subject ID: %d; Run number: %d\n', subject_id, num_run);


%% Create subject's directories and eye-tracking file

results_path_beh = fullfile(main_path, sprintf('/custime_results/sub-S%d/ses-%08s/beh', subject_id, date_of_day));
mkdir(results_path_beh);

results_path_et = fullfile(main_path, sprintf('/custime_results/sub-S%d/ses-%08s/beh', subject_id, date_of_day));
mkdir(results_path_et);
file_name_et = fullfile(results_path_et, sprintf('sub-S%d_ses-%014s_task-SentencesUsers_run-%01d.edf', subject_id, date_of_day, num_run));


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


%% Generate the order of trials and define the list of sentences to be displayed in each of the subject's runs

rng_struct = rng('shuffle'); % Initialize the random generator

file_custime_sentences_stim = fullfile(main_path, '/custime_stim_lists/custime_sentences_stim_users.mat');
file_custime_sentences_stim_subject = fullfile(main_path, sprintf('/custime_stim_lists/custime_sentences_stim_subjects/custime_sentences_stim_%d.mat', subject_id));
if ~isfile(file_custime_sentences_stim_subject) % Create a stim mat only if this hasn't be done yet for the considered subject
    copyfile(file_custime_sentences_stim, file_custime_sentences_stim_subject);
    load(file_custime_sentences_stim_subject);
    % Initialize a temporary struct log
    structure_temp(nb_sentences_in_exp) = struct('subject', [], 'run', [], ...
        'num_pair', [], 'sentence', [], 'num_sentence', [], 'predictable', [], 'diff_of_prob', [], ...
        'video', [], 'sentence_onset_time', [], 'sentence_offset_time', [], 'sentence_marker', [], ...
        'is_question', [], 'is_odd', [], 'click_time', [], 'click_marker', [], 'response_time', [], 'response_validity', [], 'response_type', [], ...
        'fixation_onset_time', [], 'fixation_duration', [], 'fixation_marker', []);

    % Append the sentences and their information to the structure log
    load('custime_sentences_stim_users');
    for i_trial=1:nb_sentences_in_exp
        structure_temp(i_trial).num_pair = custime_sentences_stim_users(i_trial).num_pair;
        structure_temp(i_trial).sentence = custime_sentences_stim_users(i_trial).sentence;
        structure_temp(i_trial).num_sentence = custime_sentences_stim_users(i_trial).num_sent;
        structure_temp(i_trial).predictable = custime_sentences_stim_users(i_trial).predictable;
        structure_temp(i_trial).diff_of_prob = custime_sentences_stim_users(i_trial).diff_of_prob;
        structure_temp(i_trial).is_question = 0;
    end
    clear('custime_sentences_stim_users');

    % Generate the order of trials
    structure_temp = structure_temp(randperm(nb_sentences_in_exp));
    num_row_substructure = 1;
    pair_in_substructure = [];
    % Append half of the predictable sentences to a substructure
    for i_trial=1:size(structure_temp,2)
        if num_row_substructure <= nb_sentences_in_exp/4 && structure_temp(i_trial).predictable
            substructure(num_row_substructure) = structure_temp(i_trial);
            structure_temp(i_trial) = [];
            pair_in_substructure = [pair_in_substructure, substructure(num_row_substructure).num_pair];
            num_row_substructure = num_row_substructure + 1;
        end
    end
    % Append the unpredictable sentences from the other pairs to the substructure
    i_trial = 1;
    while i_trial <= size(structure_temp,2)
        if ~structure_temp(i_trial).predictable && ~ismember(structure_temp(i_trial).num_pair, pair_in_substructure)
            substructure(num_row_substructure) = structure_temp(i_trial);
            structure_temp(i_trial) = [];
            pair_in_substructure = [pair_in_substructure, substructure(num_row_substructure).num_pair];
            num_row_substructure = num_row_substructure + 1;
        else
            i_trial = i_trial + 1;
        end
    end
    % Randomize the two substructures and add them together to create the final log structure (without repetition of sentences)
    structure_temp = structure_temp(randperm(nb_sentences_in_exp/2));
    substructure = substructure(randperm(nb_sentences_in_exp/2));
    log_structure_exp = [structure_temp, substructure];

    % Generate a list of the videos to be displayed
    list_videos = {};
    for i_trial=1:nb_sentences_in_exp
        list_videos(i_trial) = {fullfile(main_path,'custime_videos','custime_videos_sentences',sprintf('sent_%02d.mp4',log_structure_exp(i_trial).num_sentence))};
    end

    % Create a fixation duration for each trial
    fixation_duration_list = linspace(fixation_duration_min, fixation_duration_max, nb_sentences_in_exp);
    fixation_duration_list = fixation_duration_list(randperm(nb_sentences_in_exp));

    % Determine the trials after which a rest period will happen
    if spacing_of_rests > 0
        list_rests = [];
        rest_range = 1;
        while spacing_of_rests * rest_range < nb_sentences_in_exp
            list_rests = [list_rests, spacing_of_rests * rest_range];
            rest_range = rest_range + 1;
        end
    end

    % Append other trial info to the struct log
    for i_trial=1:nb_sentences_in_exp
        log_structure_exp(i_trial).subject = subject_id;
        log_structure_exp(i_trial).video = char(list_videos(i_trial));
        log_structure_exp(i_trial).fixation_duration = fixation_duration_list(i_trial);
    end


    %% Generate the questions

    if nb_questions_in_exp > 0
        list_possible_odd_sentences = ["Je suis sure que ca va bien se passer." "Le train aura un peu de retard." "Le garcon refuse de lire un livre si ennuyeux." ...
        "Elle a oublie d'appeler un taxi." "On s'est rencontres a la fete hier soir." "La femme ne veut plus regarder ce film." "Les enfants jouent dans la cour." ...
        "Elle met du sel dans sa salade" "La femme ne veut plus regarder ce film" "Tu as bien dormi ?" "Ce debat ne nous menera nul part"  "L'enfant fait des ricochets sur l'eau" ...
        "Il a brise le bouchon de la bouteille de champagne" "Connaissez-vous l'adresse d'un excellent coiffeur ?" "L'electricien et le plombier sont rapidement arrives" ...
        "Je ne reconnais pas ces nouveaux etudiants." "Il a toujours aime prendre le bateau." "On est arrives largement en avance." "Il y a trop de voitures dans cette ville"];

        % Determine the trials after which a question will be asked
        list_question_ranges = floor(linspace(floor(nb_sentences_in_exp/nb_questions_in_exp), nb_sentences_in_exp, nb_questions_in_exp));

        % Generate the list of sentences to be presented as questions
        while 1
            list_questions = [];
            bool_questions_odds = [];
            for i_question_range=1:nb_questions_in_exp
                if rand()>0.5
                    if i_question_range == 1
                        index_repeated_sentence = randi([1, list_question_ranges(i_question_range)]);
                    else
                        index_repeated_sentence = randi([list_question_ranges(i_question_range-1)+1, list_question_ranges(i_question_range)]);
                    end
                    list_questions = [list_questions, {log_structure_exp(index_repeated_sentence).sentence}];
                    bool_questions_odds(i_question_range) = 0;
                else
                    odd_sentence_index = randi([1, length(list_possible_odd_sentences)]);
                    list_questions = [list_questions, list_possible_odd_sentences(odd_sentence_index)];
                    list_possible_odd_sentences(odd_sentence_index) = [];
                    bool_questions_odds(i_question_range) = 1;
                end
            end
            if (sum(bool_questions_odds) >= nb_questions_in_exp/3) && (sum(bool_questions_odds) <= nb_questions_in_exp/3*2)
                break;
            end
        end

        % Append the questions in the structure log
        range_lag = 1;
        for i_question=1:nb_questions_in_exp
            index_question = list_question_ranges(i_question) + range_lag;
            log_structure_exp = [log_structure_exp(1:index_question-1),log_structure_exp(index_question-1),log_structure_exp(index_question:end)];
            log_structure_exp(index_question).num_pair = [];
            log_structure_exp(index_question).num_sentence = [];
            log_structure_exp(index_question).predictable = [];
            log_structure_exp(index_question).diff_of_prob = [];
            log_structure_exp(index_question).video = [];
            log_structure_exp(index_question).num_pair = [];
            log_structure_exp(index_question).fixation_duration = [];
            log_structure_exp(index_question).sentence = list_questions(i_question);
            log_structure_exp(index_question).is_question = 1;
            if bool_questions_odds(i_question)
                log_structure_exp(index_question).is_odd = 1;
            else
                log_structure_exp(index_question).is_odd = 0;
            end
            range_lag = range_lag + 1;
        end
    end


    %% Determine the run number

    nb_questions = 0;
    for sentence_in_exp=1:size(log_structure_exp,2)
        if nb_questions >= nb_questions_in_exp/2
            log_structure_exp(sentence_in_exp).run = 2;
        else
            log_structure_exp(sentence_in_exp).run = 1;
        end
        if log_structure_exp(sentence_in_exp).is_question
            nb_questions = nb_questions + 1;
        end
    end
    save(file_custime_sentences_stim_subject, 'log_structure_exp');
end


%% Create the log structure for this run

% Load the subject's list of stimuli and count the nb of sentences to be displayed during this run
load(file_custime_sentences_stim_subject);
nb_trials_in_run = 0;
for i_sentence=1:size(log_structure_exp,2)
    if log_structure_exp(i_sentence).run == num_run
        nb_trials_in_run = nb_trials_in_run + 1;
    end
end

% Initialize the struct log
log_structure(nb_trials_in_run) = struct('subject', [], 'run', [], ...
    'num_pair', [], 'sentence', [], 'num_sentence', [], 'predictable', [], 'diff_of_prob', [], ...
    'video', [], 'sentence_onset_time', [], 'sentence_offset_time', [], 'sentence_marker', [], ...
    'is_question', [], 'is_odd', [], 'click_time', [], 'click_marker', [], 'response_time', [], 'response_validity', [], 'response_type', [], ...
    'fixation_onset_time', [], 'fixation_duration', [], 'fixation_marker', []);

% Append the sentences and their information to the structure log for the considered run
num_row_struct_exp = 1;
i_trial = 1;
while num_row_struct_exp <= nb_sentences_in_exp+nb_questions_in_exp && i_trial <= nb_trials_in_run
    if  isequal(log_structure_exp(num_row_struct_exp).run,num_run)
        log_structure(i_trial).subject = log_structure_exp(num_row_struct_exp).subject;
        log_structure(i_trial).run = log_structure_exp(num_row_struct_exp).run;
        log_structure(i_trial).num_pair = log_structure_exp(num_row_struct_exp).num_pair;
        log_structure(i_trial).sentence = log_structure_exp(num_row_struct_exp).sentence;
        log_structure(i_trial).num_sentence = log_structure_exp(num_row_struct_exp).num_sentence;
        log_structure(i_trial).predictable = log_structure_exp(num_row_struct_exp).predictable;
        log_structure(i_trial).diff_of_prob = log_structure_exp(num_row_struct_exp).diff_of_prob;
        log_structure(i_trial).video = log_structure_exp(num_row_struct_exp).video;
        log_structure(i_trial).is_question = log_structure_exp(num_row_struct_exp).is_question;
        log_structure(i_trial).is_odd = log_structure_exp(num_row_struct_exp).is_odd;
        log_structure(i_trial).fixation_duration = log_structure_exp(num_row_struct_exp).fixation_duration;
        i_trial = i_trial + 1;
    end
    num_row_struct_exp = num_row_struct_exp + 1;
end
clear('log_structure_exp');


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
    file_name_beh = fullfile(results_path_beh, sprintf('sub-S%d_ses-%014s_task-SentencesUsers_run-%01d_beh.csv', subject_id, date_of_run, num_run));
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

task_instructions_1_size = 45;
Screen('TextSize', win, task_instructions_1_size);
task_instructions_1_text = 'Restez attentif aux vid�os';
task_instructions_1_width = RectWidth(Screen('TextBounds',win,task_instructions_1_text));
task_instructions_1_height = RectHeight(Screen('TextBounds',win,task_instructions_1_text));
task_instructions_1_x = screen_center_x-ceil(task_instructions_1_width/2);
task_instructions_1_y = screen_center_y-ceil(task_instructions_1_height)-35;
Screen('DrawText', win, task_instructions_1_text, task_instructions_1_x, task_instructions_1_y);

task_instructions_2_size = 45;
Screen('TextSize', win, task_instructions_2_size);
task_instructions_2_text = 'Quand une phrase �crite apparait, indiquez si vous l''avez d�j� vu ou non';
task_instructions_2_width = RectWidth(Screen('TextBounds',win,task_instructions_2_text));
task_instructions_2_height = RectHeight(Screen('TextBounds',win,task_instructions_2_text));
task_instructions_2_x = screen_center_x-ceil(task_instructions_2_width/2);
task_instructions_2_y = screen_center_y-ceil(task_instructions_2_height)+30;
Screen('DrawText', win, task_instructions_2_text, task_instructions_2_x, task_instructions_2_y);

task_instructions_3_size = 35;
Screen('TextSize', win, task_instructions_3_size);
task_instructions_3_text = 'J''ai d�j� vu cette phrase : clic gauche / Je n''ai jamais vu cette phrase : clic droit';
task_instructions_3_width = RectWidth(Screen('TextBounds',win,task_instructions_3_text));
task_instructions_3_height = RectHeight(Screen('TextBounds',win,task_instructions_3_text));
task_instructions_3_x = screen_center_x-ceil(task_instructions_3_width/2);
task_instructions_3_y = screen_center_y-ceil(task_instructions_3_height)+90;
Screen('DrawText', win, task_instructions_3_text, task_instructions_3_x, task_instructions_3_y);

task_continue_size = 25;
Screen('TextSize', win, task_continue_size);
task_continue_text = 'Demarrage de l''enregistrement EEG...';
task_continue_width = RectWidth(Screen('TextBounds',win,task_continue_text));
task_continue_height = RectHeight(Screen('TextBounds',win,task_continue_text));
task_continue_x = screen_center_x-ceil(task_continue_width/2);
task_continue_y = screen_center_y-ceil(task_continue_height/2)+150;
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
    file_name_beh = fullfile(results_path_beh, sprintf('sub-S%d_ses-%014s_task-SentencesUsers_run-%01d_beh.csv', subject_id, date_of_run, num_run));
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
index_question_loop = 1;

fprintf('##### Entering trial loop... \n');

for i_trial=1:size(log_structure,2)
    

    if  log_structure(i_trial).is_question % Prepare and display the questions (only if it's a "question" trial)
        sentence_question_size = 50;
        Screen('TextSize', win, sentence_question_size);
        sentence_question_text = char(log_structure(i_trial).sentence);
        sentence_question_width = RectWidth(Screen('TextBounds',win,sentence_question_text));
        sentence_question_height = RectHeight(Screen('TextBounds',win,sentence_question_text));
        sentence_question_x = screen_center_x-ceil(sentence_question_width/2);
        sentence_question_y = screen_center_y-ceil(sentence_question_height);
        Screen('DrawText', win, sentence_question_text, sentence_question_x, sentence_question_y);

        question_instruction_size = 25;
        Screen('TextSize', win, question_instruction_size);
        question_instruction_text_1 = 'J''ai d�j� vu cette phrase : clic gauche';
        question_instruction_width_1 = RectWidth(Screen('TextBounds',win,question_instruction_text_1));
        question_instruction_height_1 = RectHeight(Screen('TextBounds',win,question_instruction_text_1));
        question_instruction_x_1 = screen_center_x-ceil(question_instruction_width_1/2);
        question_instruction_y_1 = screen_center_y-ceil(question_instruction_height_1/2)+75;
        Screen('DrawText', win, question_instruction_text_1, question_instruction_x_1, question_instruction_y_1);
        
        question_instruction_text_2 = 'Je n''ai jamais vu cette phrase : clic droit';
        question_instruction_width_2 = RectWidth(Screen('TextBounds',win,question_instruction_text_2));
        question_instruction_height_2 = RectHeight(Screen('TextBounds',win,question_instruction_text_2));
        question_instruction_x_2 = screen_center_x-ceil(question_instruction_width_2/2);
        question_instruction_y_2 = screen_center_y-ceil(question_instruction_height_2/2)+115;
        Screen('DrawText', win, question_instruction_text_2, question_instruction_x_2, question_instruction_y_2);
        
        last_time = Screen('Flip',win); % Display the sentence and the instructions for the question
        SendTrigger(112);
        Eyelink('Message', 'S112');
        log_structure(i_trial).sentence_onset_time = last_time - t0;
        fprintf('## Displaying question trial %d/%d\n',i_trial, size(log_structure,2));
        fprintf('   Question on sentence, waiting for mouse press... \n');  
        while 1
            [keyIsDown, secs, keyCode] = KbCheck; % Keyboard check
            [x,y,buttons] = GetMouse; % Mouse check
            if any(buttons)
                last_mouse_click = GetSecs - t0;
                if buttons(1) % Left mouse button
                    if ~log_structure(i_trial).is_odd
                        SendTrigger(113);
                        Eyelink('Message', 'S113');
                        log_structure(i_trial).click_time = last_mouse_click;
                        log_structure(i_trial).click_marker = 113;
                        log_structure(i_trial).response_time = log_structure(i_trial).click_time - last_time;
                        log_structure(i_trial).response_validity = 1;
                        log_structure(i_trial).response_type = 'HIT';
                        fprintf('   Hit on question trial %d/%d! Response time: %ds\n', i_trial, size(log_structure,2), log_structure(i_trial).response_time);
                    else
                        SendTrigger(114);
                        Eyelink('Message', 'S114');
                        log_structure(i_trial).click_time = last_mouse_click;
                        log_structure(i_trial).click_marker = 114;
                        log_structure(i_trial).response_time = log_structure(i_trial).click_time - last_time;
                        log_structure(i_trial).response_validity = 0;
                        log_structure(i_trial).response_type = 'FA';
                        fprintf('   False alarm on trial %d/%d... Response time: %ds\n', i_trial, size(log_structure,2), log_structure(i_trial).response_time);
                    end
                elseif buttons(3) % Right mouse button
                    if  log_structure(i_trial).is_odd
                        SendTrigger(113);
                        Eyelink('Message', 'S113');
                        log_structure(i_trial).click_time = last_mouse_click;
                        log_structure(i_trial).click_marker = 113;
                        log_structure(i_trial).response_time = log_structure(i_trial).click_time - last_time;
                        log_structure(i_trial).response_validity = 1;
                        log_structure(i_trial).response_type = 'CR';
                        fprintf('   Correct rejection on question trial %d/%d! Response time: %ds\n', i_trial, size(log_structure,2), log_structure(i_trial).response_time);
                    else
                        SendTrigger(114);
                        Eyelink('Message', 'S114');
                        log_structure(i_trial).click_time = last_mouse_click;
                        log_structure(i_trial).click_marker = 114;
                        log_structure(i_trial).response_time = log_structure(i_trial).click_time - last_time;
                        log_structure(i_trial).response_validity = 0;
                        log_structure(i_trial).response_type = 'MISS';
                        fprintf('   Miss on trial %d/%d... Response time: %ds\n', i_trial, size(log_structure,2), log_structure(i_trial).response_time);
                    end
                end
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
        index_question_loop = index_question_loop + 1;
        if exp_term
            break;
        end
        fprintf('   Question answered, returning to trials'' display... \n');
        
    else % Display the blank interval, the fixation cross and the video (only if it's a "display" trial)
        % First fixation cross preparation while blank interval display
        last_time = Screen('Flip',win); % Display the blank interval
        Screen('DrawLines', win, [fixation_line_x; fixation_line_y], fixation_width, fixation_color, [screen_center_x+fixation_position_x screen_center_y+fixation_position_y]);
        while GetSecs < last_time + isi
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

        % First fixation cross info saving while first fixation cross display
        last_time = Screen('Flip', win);
        SendTrigger(1);
        Eyelink('Message', 'S  1');
        log_structure(i_trial).fixation_onset_time = last_time - t0;
        log_structure(i_trial).fixation_marker = 1;

        % Video and trigger preparation while first fixation cross display.
        video_name = log_structure(i_trial).video;
        sentence_video = Screen('OpenMovie', win, video_name);
        Screen('PlayMovie', sentence_video, 1);
        first_image = 1;
        trigger_val =  128 + log_structure(i_trial).num_sentence;
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

        % Sentence info saving while sentence video display
        while 1
            tex = Screen('GetMovieImage', win, sentence_video); % Wait for next movie frame, retrieve texture handle to it
            if tex<=0 % Valid texture returned? A negative value means end of movie reached
                break;
            end
            [keyIsDown, secs, keyCode] = KbCheck;
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
                log_structure(i_trial).sentence_onset_time = last_time - t0;
                log_structure(i_trial).sentence_marker = trigger_val;
                first_image = 0;
                fprintf('# Displaying video trial %d/%d\n',i_trial, size(log_structure,2));
            end
            Screen('Close', tex); % Release texture
        end
        Screen('PlayMovie', sentence_video, 0); % Stop playback
        Screen('CloseMovie', sentence_video); % Close movie
        log_structure(i_trial).sentence_offset_time = last_time - t0;
        
        % Second fixation cross preparation and display
        Screen('DrawLines', win, [fixation_line_x; fixation_line_y], fixation_width, fixation_color, [screen_center_x+fixation_position_x screen_center_y+fixation_position_y]);
        last_time = Screen('Flip',win); % Display the blank interval
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
    end
    if spacing_of_rests > 0 % Implement the rest periods
        if ismember(i_trial, list_rests)
            fprintf('## Entering rest period... \n');
            WaitSecs(rest_duration);
            fprintf('## Exiting rest period. \n');
        end
    end
end

end_of_run_timestamp = GetSecs;
run_duration = end_of_run_timestamp - date_of_run;


%% Save struct log and eye-tracking files

fprintf('##### Saving struct log file... \n');
file_name_beh = fullfile(results_path_beh, sprintf('sub-S%d_ses-%014s_task-SentencesUsers_run-%01d_beh.csv', subject_id, date_of_run, num_run));
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

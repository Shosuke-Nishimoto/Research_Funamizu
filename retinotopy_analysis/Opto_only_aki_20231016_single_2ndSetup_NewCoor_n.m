function varargout = Opto_only_aki_20231016_single_2ndSetup_NewCoor_n(varargin)
% NI_LED_ONLY_AKI2 MATLAB code for NI_LED_only_aki2.fig
%      NI_LED_ONLY_AKI2, by itself, creates a new NI_LED_ONLY_AKI2 or raises the existing
%      singleton*.
%
%      H = NI_LED_ONLY_AKI2 returns the handle to a new NI_LED_ONLY_AKI2 or the handle to
%      the existing singleton*.
%
%      NI_LED_ONLY_AKI2('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in NI_LED_ONLY_AKI2.M with the given input arguments.
%
%      NI_LED_ONLY_AKI2('Property','Value',...) creates a new NI_LED_ONLY_AKI2 or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before NI_LED_only_aki2_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to NI_LED_only_aki2_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help NI_LED_only_aki2

% Last Modified by GUIDE v2.5 17-Oct-2023 12:48:21

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Opto_only_aki_20231016_single_2ndSetup_NewCoor_n_OpeningFcn, ...
                   'gui_OutputFcn',  @Opto_only_aki_20231016_single_2ndSetup_NewCoor_n_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ... 
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT

% --- Executes just before NI_LED_only_aki2 is made visible.
function Opto_only_aki_20231016_single_2ndSetup_NewCoor_n_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to NI_LED_only_aki2 (see VARARGIN)

% Choose default command line output for NI_LED_only_aki2
handles.output = hObject;
ver = datenum(version('-date')); %make sure matlab is 2016b or newer

% if ver < 736580 %release date for 2016b
%     error('Matlab version is older as version 2016b. This can cause problems with timing accuracy.')
% end    

%% initialize NI card    
handles = RecordMode_Callback(handles.RecordMode, [], handles); %check recording mode to create correct ni object

CheckPath(handles); %Check for data path, reset date and trialcount
if any(ismember(handles.driveSelect.String(:,1), 'g')) %start on G drive by default
    handles.driveSelect.Value = find(ismember(handles.driveSelect.String(:,1), 'g')); 
end
       
% UIWAIT makes NI_LED_only_aki2 wait for user response (see UIRESUME)
% uiwait(handles.NI_LED_only_aki2);

% --- Executes on selection change in RecordMode.
function handles = RecordMode_Callback(hObject, eventdata, handles)
% hObject    handle to RecordMode (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns RecordMode contents as cell array
%        contents{get(hObject,'Value')} returns selected item from RecordMode

handles.ExperimentType.Value = 1;
handles.AnimalID.Value = 1;

if hObject.Value == 1 %set standard settings for phase mapping
    handles.BaselineFrames.String = '2';
    handles.PostStimFrames.String = '23';
elseif  hObject.Value == 2 %set standard settings for bpod recording
    handles.BaselineFrames.String = '3.25';
    handles.PostStimFrames.String = '6';
elseif  hObject.Value == 3 %set paradigm to 'Spont' if doing spontaneous recording
    if ~isdir([handles.path.base 'Animals' filesep handles.AnimalID.String{handles.AnimalID.Value} filesep 'Spont'])
        mkdir([handles.path.base 'Animals' filesep handles.AnimalID.String{handles.AnimalID.Value} filesep 'Spont']);
    end
    handles = CheckPath(handles); %Check for data path, reset date and trialcount
    handles.ExperimentType.Value = find(ismember(handles.ExperimentType.String, 'Spont'));
    CheckPath(handles);
end
CheckPath(handles);

% check NI card
if isempty(daqlist)
    disp('No NI devices found - check connections and restart Matlab to try again')
    handles.dNIdevice = [];
else
    if isfield(handles,'dNIdevice')
        delete(handles.dNIdevice);
        delete(handles.aNIdevice);
        delete(handles.aNIdeviceOut);
    end
    
    handles.aNIdeviceOut = daq("ni"); %Clocked operations
%     ch_out = addoutput(handles.aNIdeviceOut,"Dev1",0:1,"Voltage");%nishimoto 230622
    ch_out = addoutput(handles.aNIdeviceOut,"Dev2",0:1,"Voltage");
%     addoutput(handles.aNIdeviceOut,"Dev1","Port0/Line0","Digital");%nishimoto 230622
    addoutput(handles.aNIdeviceOut,"Dev2","Port0/Line0","Digital");
    %ch_out2 = addoutput(handles.aNIdeviceOut,"Dev2",0,"Voltage");
    handles.aNIdeviceOut.Rate = 5000; %set sampling rate to 5kHz
    for i = 1:2
        ch_out(i).TerminalConfig = 'SingleEnded';
    end
    %ch_out2.TerminalConfig = 'SingleEnded';
        
    handles.dNIdevice = daq("ni"); %Single channel operations
%     addoutput(handles.dNIdevice,"Dev1","Port1/Line0:3","Digital");%nishimoto 230622
%     addinput(handles.dNIdevice,"Dev1","Port1/Line4","Digital");%nishimoto 230622
    addoutput(handles.dNIdevice,"Dev2","Port1/Line0:3","Digital");
    addinput(handles.dNIdevice,"Dev2","Port1/Line4","Digital");
    %addinput(handles.dNIdevice,"Dev1","Port0/Line0","Digital");
    
    handles.aNIdevice = daq("ni"); %Single channel operations
%     ch = addinput(handles.aNIdevice,"Dev1",0:8,"Voltage");%nishimoto 230622
    ch = addinput(handles.aNIdevice,"Dev2",0:8,"Voltage");
    %addinput(handles.aNIdevice,"Dev1",0:8,"Voltage");
    handles.aNIdevice.Rate = 1000; %set sampling rate to 1kHz
%     for i = 1:9
%         ch(i).TerminalConfig = 'Differential';
%     end
    ch(1).TerminalConfig = 'SingleEnded';
    ch(9).TerminalConfig = 'SingleEnded';
    %read()
    %start(d,"Continuous")
    
end
handles.use_zahyou = [];
handles.opt_schedule = [];
handles.save_NI_file = [];
guidata(hObject,handles);

% --- Executes on button press in BlueLight.
function BlueLight_Callback(hObject, eventdata, handles)
% hObject    handle to BlueLight (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% lightMode is 1 to 3
if isempty(handles.dNIdevice)
    disp('LED control not available - NI device is missing')
    set(hObject, 'Value',false)
else
    if hObject.Value
        out = false(1,4); out(handles.lightMode.Value) = true; %indicator for NI channel
        %out = false(1,3); out(handles.lightMode.Value) = true; %indicator for NI channel
        %outputSingleScan(handles.dNIdevice,out)
        write(handles.dNIdevice,out)
        if handles.lightMode.Value == 1
            hObject.BackgroundColor = [0 0 1];
            hObject.String = 'Task Stim';
        elseif handles.lightMode.Value == 2
            hObject.BackgroundColor = [.5 0 .5];
            hObject.String = 'Free Sine wave';
        elseif handles.lightMode.Value == 3
            hObject.BackgroundColor = [.25 0 .75];
            hObject.String = 'Free Max stim';
        elseif handles.lightMode.Value == 4
            hObject.BackgroundColor = [1 1 0];
            hObject.String = 'Continous Task';
        end
    else
        %outputSingleScan(handles.dNIdevice,false(1,3))
        %outputSingleScan(handles.dNIdevice,false(1,4))
        write(handles.dNIdevice,false(1,4))
        hObject.BackgroundColor = zeros(1,4);
            hObject.String = 'LED OFF';
    end
end
guidata(hObject,handles);

% --- Executes on key press with focus on BlueLight and none of its controls.
function BlueLight_KeyPressFcn(hObject, eventdata, handles)
% hObject    handle to BlueLight (see GCBO)
% eventdata  structure with the following fields (see MATLAB.UI.CONTROL.UICONTROL)
%	Key: name of the key that was pressed, in lower case
%	Character: character interpretation of the key(s) that was pressed
%	Modifier: name(s) of the modifier key(s) (i.e., control, shift) pressed
% handles    structure with handles and user data (see GUIDATA)

% --- Executes on button press in AcqusitionStatus.
% --- Executes on button press in AcqusitionStatus.

% --- Executes on button press in NI_start.
function NI_start_Callback(hObject, eventdata, handles)
% hObject    handle to NI_start (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of AcqusitionStatus
if hObject.Value
    hObject.BackgroundColor = [1 0 0];
    
    %Make the save file
    dir_name = pwd;
    temp_day = date;
    [hour,min,sec] = hms(datetime);
    sec = round(sec);
    if min < 10,
        min = sprintf('0%d',min);
    else
        min = sprintf('%d',min);
    end
    if sec < 10,
        sec = sprintf('0%d',sec);
    else
        sec = sprintf('%d',sec);
    end
    temp_file = sprintf('%s/NI_%s_%d%s_%s.bin', dir_name,temp_day,hour,min,sec)

    %Start recording
    %handles.fid1_NI = fopen('log.bin','w');
    handles.fid1_NI = fopen(temp_file,'w');
    %fid1 = fopen('log.bin','w');
    handles.aNIdevice.ScansAvailableFcn = @(src, evt) logData(src, evt, handles.fid1_NI);
    %handles.lh = addlistener(handles.aNIdevice,'DataAvailable',@(src, event)logData(src, event, handles.fid1_NI));
    %handles.aNIdevice.startBackground;
    start(handles.aNIdevice,"Continuous");
    pause(0.5); %To make sure that the save is working
    disp('start NI recording')
    %handles
    handles.save_NI_file = temp_file;
    guidata(hObject,handles);
else
    %handles
    hObject.BackgroundColor = [0 1 0];
    %handles.aNIdevice.stop;
    stop(handles.aNIdevice);
    %delete(handles.lh);
    fclose(handles.fid1_NI);
    disp('end NI recording')
    guidata(hObject,handles);
end

% --- Outputs from this function are returned to the command line.
function varargout = Opto_only_aki_20231016_single_2ndSetup_NewCoor_n_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Executes on button press in Opto_start.
function Opto_Task_start_Callback(hObject, eventdata, handles)
% hObject    handle to Opto_start (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%handles
zahyou = handles.use_zahyou;
[size_y,size_x] = size(zahyou);
if length(zahyou) == 0
    disp('cannot push this program')
    hoge
end
count = 0;

if hObject.Value
    hObject.BackgroundColor = [1 0 0];

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Initialize everything
    save_file = handles.save_NI_file;
    if isempty(save_file)
        save_file = 'temp_scan_both.mat';
        save_file2 = 'temp_scan_both2.mat';
    else
        save_file = [save_file,'_both.mat'];
        save_file2 = [save_file,'_both2.mat'];
    end
    %handles
    %Set initial save folder 
    %Make schedule for moving mirrors one rotation!!, With 5kHz
    %5000 is 1 sec
    %Make 40Hz wave forms: every 0.025*5 = 125
    flat_step = 110;
    move_step = 15;
    opt_schedule = nan(2000,7);
    for i = 1:size_y
        temp_opt_schedule = [i, i,zahyou(i,1), zahyou(i,2),zahyou(i,3), zahyou(i,4)];
        [opt_signal(i,1).matrix,opt_signal(i,2).matrix] = make_signal_wave_2ndSetup_single_20210930(temp_opt_schedule,flat_step,move_step);
    end
    for i = 1:2000
        temp_rand = ceil(rand * size_y);
        temp_side = round(rand) + 1;
        if temp_rand == 0
            temp_rand = 1;
        end
%        opt_schedule(i,:) = [i, temp_rand,zahyou(temp_rand,1), zahyou(temp_rand,2),zahyou(temp_rand,3), zahyou(temp_rand,4)];
        if rand > 0.2-1/size_y
            opt_schedule(i,:) = [i, temp_rand,zahyou(temp_rand,1), zahyou(temp_rand,2),zahyou(temp_rand,3), zahyou(temp_rand,4),temp_side];
        else
            opt_schedule(i,:) = [i, size_y,zahyou(size_y,1), zahyou(size_y,2),zahyou(size_y,3), zahyou(size_y,4),temp_side];
        end
        %opt_signal(i).matrix = make_signal_wave_1stSetup_20210825(opt_schedule(i,:),flat_step,move_step);
    end
    save(save_file2,'opt_schedule','zahyou');
    disp('start task optogenetics')
    save_flag = 1;
    task_flag = 1;
    change_flag = 1;
    for i = 1:2000
        count = i;

        if handles.Single_check.Value == 1
            temp_single = handles.SingleScanPlace;
            opt_schedule(i,:) = [i, temp_single,zahyou(temp_single,1), zahyou(temp_single,2),zahyou(temp_single,3), zahyou(temp_single,4),handles.check_right_stim.Value + 1];
            %opt_signal(i).matrix = make_signal_wave_1stSetup_20210825(opt_schedule(i,:),flat_step,move_step);
        end

        opt_schedule(i,:)
        temp = opt_schedule(i,2);
        temp2 = opt_schedule(i,7);
        %Start continuous movement
        start(handles.aNIdeviceOut,"RepeatOutput")
        write(handles.aNIdeviceOut,opt_signal(temp,temp2).matrix)
        while(1)
            % % read(handles.dNIdevice,"OutputFormat","Matrix")
            %Use the BPOD offset to change the mirror position!!
            temp_stop_matrix = read(handles.dNIdevice,"OutputFormat","Matrix");
            if temp_stop_matrix && change_flag == 1 %read the offset of laser stim
                stop(handles.aNIdeviceOut)
                change_flag = 0;
                break
            elseif ~temp_stop_matrix
                change_flag = 1;
            end
            if hObject.Value == 0 %end task
                disp('end task optogenetics')
                stop(handles.aNIdeviceOut)
                task_flag = 0;
                break
            end
            pause(eps)
        end
        if task_flag == 0
            break
        end
    end
else
    save_flag = 0;
end
if save_flag
    handles.save_NI_file
    %handles.opt_schedule = opt_schedule;
    save(save_file,'opt_schedule','zahyou','count');
    handles.save_NI_file = [];
end
stop(handles.aNIdeviceOut)
hObject.BackgroundColor = [1 1 0];
    
guidata(hObject,handles);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Executes on button press in Opto_order_start.
function Opto_order_start_Callback(hObject, eventdata, handles)
% hObject    handle to Opto_start (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%handles
zahyou = handles.use_zahyou;
[size_y,size_x] = size(zahyou);
if length(zahyou) == 0
    disp('cannot push this program')
    hoge
end

if hObject.Value
    hObject.BackgroundColor = [1 0 0];

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Initialize everything
    %Make schedule for moving mirrors one rotation!!, With 5kHz
    %5000 is 1 sec
    %Make 40Hz wave forms: every 0.025*5 = 125
    flat_step = 110;
    move_step = 15;
    %handles
    %Set initial save folder 
    opt_schedule = nan(200,6);
%     temp_schedule = [1, 1,zahyou(1,1), zahyou(1,2),zahyou(1,3), zahyou(1,4)];
%     temp_signal = make_signal_wave(temp_schedule,flat_step,move_step);
%     opt_signal = nan(length(temp_signal),200);
    for i = 1:200
        temp_rand = rem(i,size_y) + 1;
        opt_schedule(i,:) = [i, temp_rand,zahyou(temp_rand,1), zahyou(temp_rand,2),zahyou(temp_rand,3), zahyou(temp_rand,4)];
        opt_signal(i).matrix = make_signal_wave_2ndSetup_20210930(opt_schedule(i,:),flat_step,move_step);
    end
    disp('start task optogenetics')
    
    for i = 1:200
        %pause(2) %temporal for on/off
        %Start continuous movement
        start(handles.aNIdeviceOut,"RepeatOutput")
        write(handles.aNIdeviceOut,opt_signal(i).matrix)
        pause(2)
        stop(handles.aNIdeviceOut)
        write(handles.aNIdeviceOut,zeros(5000,3))
        pause(1)
        stop(handles.aNIdeviceOut)
        if hObject.Value == 0
            disp('end task optogenetics')
            break
        end
    end
else
%     stop(handles.aNIdeviceOut)
%     hObject.BackgroundColor = [0 1 1];
%     disp('end task optogenetics')
end
stop(handles.aNIdeviceOut)
hObject.BackgroundColor = [0 1 1];

guidata(hObject,handles);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
function signal1 = make_signal_wave_2ndSetup_20210930(use_opt_schedule,flat_step,move_step)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% %     flat_step = 110;
% %     move_step = 15;
% %     one step is 110 + 15 = 125
%     %There is delay about 12600us or 12000us in dev2
%     pre_step = 65;
%     all_step = flat_step + move_step;
    test1 = ones(1,flat_step) .* use_opt_schedule(3);
    test2 = ones(1,flat_step) .* use_opt_schedule(5);
    temp1 = [0:move_step-1] * (use_opt_schedule(5) - use_opt_schedule(3)) / (move_step-1)  + use_opt_schedule(3);
    temp2 = [0:move_step-1] * (use_opt_schedule(3) - use_opt_schedule(5)) / (move_step-1)  + use_opt_schedule(5);
    slope1 = [test1, temp1, test2, temp2];
    test1 = ones(1,flat_step) .* use_opt_schedule(4);
    test2 = ones(1,flat_step) .* use_opt_schedule(6);
    temp1 = [0:move_step-1] * (use_opt_schedule(6) - use_opt_schedule(4)) / (move_step-1)  + use_opt_schedule(4);
    temp2 = [0:move_step-1] * (use_opt_schedule(4) - use_opt_schedule(6)) / (move_step-1)  + use_opt_schedule(6);
    slope2 = [test1, temp1, test2, temp2];
    trig1 = zeros(1,length(slope1));
    trig1(1:10) = 1;
    trig1(flat_step+move_step+1:flat_step+move_step+10) = 1;
% %     trig1(1:10) = 5;
% %     trig1(flat_step+move_step+1:flat_step+move_step+10) = 5;
%     trig1(all_step-pre_step+1:all_step-pre_step+10) = 5;
%     trig1(2*all_step-pre_step+1:2*all_step-pre_step+10) = 5;
    
    rep_signal = ceil(5000/(2*(flat_step+move_step)));
    signal1 = [slope1',slope2',trig1'];
    signal1 = repmat(signal1,rep_signal,1);
    
    
    return
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
function [signal1,signal2] = make_signal_wave_2ndSetup_single_20210930(use_opt_schedule,flat_step,move_step)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%     flat_step = 110;
%     move_step = 15;
%     one step is 110 + 15 = 125
    %There is delay about 12600us or 12000us in dev2
    pre_step = 65;
    all_step = flat_step + move_step;
%     test1 = ones(1,flat_step) .* use_opt_schedule(3);
%     test2 = ones(1,flat_step) .* use_opt_schedule(5);
%     temp1 = [0:move_step-1] * (use_opt_schedule(5) - use_opt_schedule(3)) / (move_step-1)  + use_opt_schedule(3);
%     temp2 = [0:move_step-1] * (use_opt_schedule(3) - use_opt_schedule(5)) / (move_step-1)  + use_opt_schedule(5);
%     slope1 = [test1, temp1, test2, temp2];
%     test1 = ones(1,flat_step) .* use_opt_schedule(4);
%     test2 = ones(1,flat_step) .* use_opt_schedule(6);
%     temp1 = [0:move_step-1] * (use_opt_schedule(6) - use_opt_schedule(4)) / (move_step-1)  + use_opt_schedule(4);
%     temp2 = [0:move_step-1] * (use_opt_schedule(4) - use_opt_schedule(6)) / (move_step-1)  + use_opt_schedule(6);
%     slope2 = [test1, temp1, test2, temp2];
%     trig1 = zeros(1,length(slope1));
% %     trig1(1:10) = 1;
% %     trig1(flat_step+move_step+1:flat_step+move_step+10) = 1;
% %     trig1(1:10) = 5;
% %     trig1(flat_step+move_step+1:flat_step+move_step+10) = 5;
%     trig1(all_step-pre_step+1:all_step-pre_step+10) = 5;
%     trig1(2*all_step-pre_step+1:2*all_step-pre_step+10) = 5;
    
    test1_1 = ones(1,all_step) .* use_opt_schedule(5);
    test1_2 = ones(1,all_step) .* use_opt_schedule(6);
    test2_1 = ones(1,all_step) .* use_opt_schedule(3);
    test2_2 = ones(1,all_step) .* use_opt_schedule(4);

%     trig1 = zeros(1,all_step);
%     trig1(all_step-pre_step+1:all_step-pre_step+10) = 5;
    trig1 = zeros(1,all_step);
    trig1(1:10) = 1;
    
%     rep_signal = ceil(5000/(2*(flat_step+move_step)));
%     signal1 = [slope1',slope2',trig1'];
%     signal1 = repmat(signal1,rep_signal,1);
    rep_signal = ceil(5000/all_step);
    signal1 = [test1_1',test1_2',trig1'];
    signal1 = repmat(signal1,rep_signal,1);
    signal2 = [test2_1',test2_2',trig1'];
    signal2 = repmat(signal2,rep_signal,1);
    
    return
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Executes during object creation, after setting all properties.
function Bregma_Lambda_table_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Bregma_Lambda_table (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
set(hObject, 'Data', cell(2,4));
set(hObject, 'Data', [-1 -0.2 1.6 2.6 1.5 1.5 2.0; 0.7 0.6 0.4 0.1 -1 1.8 1.5]);
set(hObject, 'RowName', {'X', 'Y'}, 'ColumnName', {'Front', 'Bregma', 'Lambda', 'Headbar', 'Lambda_L', 'Lambda_R', 'Target'});
cell_value = get(hObject,'Data');
% cell_value = cell_value([1:3],[1:4]);
% cell_value = cell2mat(cell_value);
% handles.Ref_coordinate = cell_value;

% --- Executes when entered data in editable cell(s) in Bregma_Lambda_table.
function Bregma_Lambda_table_CellEditCallback(hObject, eventdata, handles)
% hObject    handle to Bregma_Lambda_table (see GCBO)
% eventdata  structure with the following fields (see MATLAB.UI.CONTROL.TABLE)
%	Indices: row and column indices of the cell(s) edited
%	PreviousData: previous data for the cell(s) edited
%	EditData: string(s) entered by the user
%	NewData: EditData or its converted form set on the Data property. Empty if Data was not changed
%	Error: error string when failed to convert EditData to appropriate value for Data
% handles    structure with handles and user data (see GUIDATA)
cell_value = get(hObject,'Data');
% cell_value = cell_value([1:3],[1:4]);
% cell_value = cell2mat(cell_value);
% handles.Ref_coordinate = cell_value;
% %disp('Update Ref. Coordinate')

% --- Executes on button press in togglebutton2.
function togglebutton2_Callback(hObject, eventdata, handles)
% hObject    handle to togglebutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of togglebutton2

cell_value = handles.Bregma_Lambda_table.Data;
cell_value = cell_value([1:2],[1:6]);
front  = [cell_value(2,1);cell_value(1,1)];
bregma = [cell_value(2,2);cell_value(1,2)];
lambda = [cell_value(2,3);cell_value(1,3)];
headbar = [cell_value(1,4);cell_value(2,4)]; %This is for X and Y

lambda_L = [cell_value(2,5);cell_value(1,5)];
lambda_R = [cell_value(2,6);cell_value(1,6)];

%%%%%
Target = [cell_value(2,7);cell_value(1,7)];
%%%%%

%Based on the 3 points estimate the slope
regress_x = [front(2);bregma(2);lambda(2)];
regress_y = [front(1);bregma(1);lambda(1)];
b = regress(regress_y,[ones(3,1),regress_x]);
%One step back from Lambda
%Two steps forward from Bregma

[new_lambda, seppen_lambda] = get_new_lambda(lambda, b);
[new_bregma, seppen_bregma] = get_new_lambda(bregma, b);
[new_front, seppen_front] = get_new_lambda(front, b);

gain = (new_lambda(2) - new_bregma(2)) * 0.03;
front_point  = new_front(2) + gain;
lambda_point = new_lambda(2) - gain;

%%%%%
disp('New Bregma point')
disp(new_bregma)
disp('New Lambda point')
disp(new_lambda)
%%%%%

%Front is the point of big blood vessel
%Make front part
center_x1 = [0:1:3] .* (new_bregma(2) - front_point) ./ 3 + front_point;
center_y1 = b(2) .* center_x1 + b(1); %x = ay + b (x and y are reversed)
%Make back part
center_x2 = [1:1:3] .* (lambda_point - new_bregma(2)) ./ 3 + new_bregma(2);
center_y2 = b(2) .* center_x2 + b(1); %x = ay + b (x and y are reversed)

center_x = [center_x1, center_x2];
center_y = [center_y1, center_y2];

%Max distance to lambda_L and lambda_R
[new_lambda_L, ~] = get_new_lambda(lambda_L, b);
[new_lambda_R, ~] = get_new_lambda(lambda_R, b);
each_x_L = (new_lambda_L(2) - lambda_L(2)) ./ 4;
each_y_L = (new_lambda_L(1) - lambda_L(1)) ./ 4;
each_x_R = (lambda_R(2) - new_lambda_R(2)) ./ 4;
each_y_R = (lambda_R(1) - new_lambda_R(1)) ./ 4;
%Make the diagonal lines
new_x_L = [-3.9,-2.9,-1.9,-0.9] .* each_x_L;
new_y_L = [-3.9,-2.9,-1.9,-0.9] .* each_y_L;
new_x_R = [0.9,1.9,2.9,3.9] .* each_x_R;
new_y_R = [0.9,1.9,2.9,3.9] .* each_y_R;
new_x = [new_x_L, new_x_R];
new_y = [new_y_L, new_y_R];

for i = 1:7
    zahyou_x(i,:) = new_x + center_x(i);
    zahyou_y(i,:) = new_y + center_y(i);
end
use_coordinate(1).matrix = [3:6];
use_coordinate(2).matrix = [2:7];
use_coordinate(3).matrix = [2:7];
use_coordinate(4).matrix = [1:8];
use_coordinate(5).matrix = [1:8];
use_coordinate(6).matrix = [1:8];
use_coordinate(7).matrix = [1:8];

use_zahyou = [];
count = 0;
figure
plot(regress_x,regress_y,'g.')
hold on
plot([lambda_L(2),lambda_R(2)],[lambda_L(1),lambda_R(1)],'g.')
hold on
plot([new_lambda_L(2),new_lambda_R(2)],[new_lambda_L(1),new_lambda_R(1)],'c.')
hold on
plot(center_x,center_y,'r.')
hold on
plot(new_lambda(2),new_lambda(1),'b.')
hold on
plot(new_bregma(2),new_bregma(1),'b.')
%for i = 1:8
for i = 1:7
    hold on
    plot(zahyou_x(i,:),zahyou_y(i,:),'m.')
    temp = use_coordinate(i).matrix;
    for j = 1:length(temp)/2
        temp1 = 5-j;
        temp2 = 4+j;
        hold on
        plot(zahyou_x(i,temp1),zahyou_y(i,temp1),'ko')
        hold on
        plot(zahyou_x(i,temp2),zahyou_y(i,temp2),'ko')
        
        count = count + 1;
        use_zahyou(count,:) = [zahyou_x(i,temp1),zahyou_y(i,temp1), zahyou_x(i,temp2),zahyou_y(i,temp2)];
    end
end
headbar = [headbar', headbar'];
% use_zahyou = [use_zahyou; headbar];
use_zahyou = [Target'; headbar];
%Check the most front point
if center_x(1) < front(2)
    disp('Front point is too far')
    hoge
end
handles.use_zahyou = use_zahyou;
%handles.use_zahyou
guidata(hObject,handles);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [new_lambda, seppen_new_lambda] = get_new_lambda(lambda, b)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

slope_x = b(2);
slope_y = -1/slope_x; %Diagonal from bregma-lambda line

seppen_new_lambda = lambda(1) - slope_y .* lambda(2);
new_lambda(2) = (seppen_new_lambda-b(1)) ./ (slope_x-slope_y);
%new_lambda(1) = (seppen_new_lambda*slope_x-b(1)*slope_y) ./ (slope_x-slope_y);
new_lambda(1) = slope_x .* new_lambda(2) + b(1);

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plotData(src,event)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     plot(event.TimeStamps,event.Data)
return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function readData
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid2 = fopen('log.bin','r');
[data,count] = fread(fid2,[3,inf],'double');
fclose(fid2);

t = data(1,:);
ch = data(2:3,:);
plot(t, ch);

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function logData(src, ~, fid)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Add the time stamp and the data values to data. To write data sequentially,
% % transpose the matrix.
% 
% %   Copyright 2011 The MathWorks, Inc.
% 
% data = [evt.TimeStamps, evt.Data]' ;
% fwrite(fid,data,'double');

%New way @ 20210816
[data, timestamps, ~] = read(src, src.ScansAvailableFcnCount, "OutputFormat", "Matrix");

data = [timestamps, data]' ;
fwrite(fid,data,'double');

return

% --- Executes on button press in ChangeDataPath.
function ChangeDataPath_Callback(hObject, eventdata, handles)
% hObject    handle to ChangeDataPath (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.DataPath.String = uigetdir; %overwrites the complete file path for data storage with user selection

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%Get the value for analog control
% --- Executes on slider movement.
function slider2_Callback(hObject, eventdata, handles)
% hObject    handle to slider2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
%value_slider2 = get(hObject,'Value');
handles.OptoScan1 = get(hObject,'Value');
%display(handles.slider2);
set(handles.edit14,'String',num2str(handles.OptoScan1));
%set the OptoScan value to the Analog outputs.
handles.OptoScan2 = get(handles.slider3,'Value');
[handles.OptoScan1, handles.OptoScan2]
%outputSingleScan(handles.aNIdeviceOut,[handles.OptoScan1, handles.OptoScan2])
write(handles.aNIdeviceOut,[handles.OptoScan1, handles.OptoScan2, 0])

% --- Executes on slider movement.
function slider3_Callback(hObject, eventdata, handles)
% hObject    handle to slider3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
%value_slider3 = get(hObject,'Value');
handles.OptoScan2 = get(hObject,'Value');
set(handles.edit16,'String',num2str(handles.OptoScan2));
%set the OptoScan value to the Analog outputs.
handles.OptoScan1 = get(handles.slider2,'Value');
[handles.OptoScan1, handles.OptoScan2]
%outputSingleScan(handles.aNIdeviceOut,[handles.OptoScan1, handles.OptoScan2, 0])
write(handles.aNIdeviceOut,[handles.OptoScan1, handles.OptoScan2, 0])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function DataPath_Callback(hObject, eventdata, handles)
% hObject    handle to DataPath (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of DataPath as text
%        str2double(get(hObject,'String')) returns contents of DataPath as a double


% --- Executes during object creation, after setting all properties.
function DataPath_CreateFcn(hObject, eventdata, handles)
% hObject    handle to DataPath (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on button press in CurrentStatus.
function CurrentStatus_Callback(hObject, eventdata, handles)
% hObject    handle to CurrentStatus (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of CurrentStatus

function RecordingID_Callback(hObject, eventdata, handles)
% hObject    handle to RecordingID (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of RecordingID as text
%        str2double(get(hObject,'String')) returns contents of RecordingID as a double


% --- Executes during object creation, after setting all properties.
function RecordingID_CreateFcn(hObject, eventdata, handles)
% hObject    handle to RecordingID (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in AnimalID.
function AnimalID_Callback(hObject, eventdata, handles)
% hObject    handle to AnimalID (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns AnimalID contents as cell array
%        contents{get(hObject,'Value')} returns selected item from AnimalID

CheckPath(handles);

    
% --- Executes during object creation, after setting all properties.
function AnimalID_CreateFcn(hObject, eventdata, handles)
% hObject    handle to AnimalID (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in ExperimentType.
function ExperimentType_Callback(hObject, eventdata, handles)
% hObject    handle to ExperimentType (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns ExperimentType contents as cell array
%        contents{get(hObject,'Value')} returns selected item from ExperimentType

CheckPath(handles);

% --- Executes during object creation, after setting all properties.
function ExperimentType_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ExperimentType (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function TrialNr_Callback(hObject, eventdata, handles)
% hObject    handle to TrialNr (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of TrialNr as text
%        str2double(get(hObject,'String')) returns contents of TrialNr as a double


% --- Executes during object creation, after setting all properties.
function TrialNr_CreateFcn(hObject, eventdata, handles)
% hObject    handle to TrialNr (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function handles = CheckPath(handles)

sizLim = 150; % check remaining disk space and suggest different drive if less than sizLim (in gb) is left
% Look for single-letter drives, starting at a: or c: as appropriate
ret = {};
for i = double('c') : double('z')
    if exist(['' i ':\'], 'dir') == 7
        ret{end+1} = [i ':']; %#ok<AGROW>
    end
end
handles.driveSelect.String = char(ret);

cPath = java.io.File(strtrim(handles.driveSelect.String(handles.driveSelect.Value,:)));
% if (cPath.getFreeSpace / 2^30) < sizLim && length(handles.driveSelect.String) > 1
%     answer = questdlg(['Only ' num2str((cPath.getFreeSpace / 2^30)) 'gb left on ' strtrim(handles.driveSelect.String(handles.driveSelect.Value,:)) filesep '. Change drive?'], ...
%         'Drive select', 'Yes', 'No', 'Yes');
%     if strcmp(answer, 'Yes')
%         
%         checker = true;
%         cDrive = handles.driveSelect.Value; %keep current drive index
%         while checker
%             handles.driveSelect.Value = rem(handles.driveSelect.Value,length(handles.driveSelect.String))+1; %increase drive selection value by 1
%             cPath = java.io.File(strtrim(handles.driveSelect.String(handles.driveSelect.Value,:)));
%             if (cPath.getFreeSpace / 2^30) > sizLim
%                 disp(['Changed path to drive ' strtrim(handles.driveSelect.String(handles.driveSelect.Value,:)) filesep '. ' num2str((cPath.getFreeSpace / 2^30)) 'gb remaining.'])
%                 checker = false;
%             elseif handles.driveSelect.Value == cDrive
%                 disp(['Could not find a drive with more then ' num2str(sizLim) 'gb of free space. Path unchanged.'])
%                 checker = false;
%             end
%         end
%     end
% end

% find basepath and look for present animals, experiment types and past recordings
if handles.RecordMode.Value == 1 || handles.RecordMode.Value == 3 
    handles.path.base = [handles.driveSelect.String(handles.driveSelect.Value,:) '\WidefieldImager\']; %get path of imaging code
elseif handles.RecordMode.Value == 2
    handles.path.base = [handles.driveSelect.String(handles.driveSelect.Value,:) '\BpodImager\']; %get path of imaging code
else
    error('Unknown recording mode');
end

if ~isdir([handles.path.base 'Animals']) %check for animal path to save data
    mkdir([handles.path.base 'Animals']) %create folder if required
end

handles.AnimalID.String = cellstr(handles.AnimalID.String);
folders = dir([handles.path.base 'Animals']); %find animal folders
folders = folders([folders.isdir] & ~strncmpi('.', {folders.name}, 1));
checker = true;
for iAnimals = 1:size(folders,1) %skip first two entries because they contain folders '.' and '..'
    AllAnimals{iAnimals} = folders(iAnimals).name; %get animal folders
    if checker
        if strcmp(handles.AnimalID.String{handles.AnimalID.Value},folders(iAnimals).name) %check if current selected animal coincides with discovered folder
            handles.AnimalID.Value = iAnimals; %keep animal selection constant
            checker = false;
        end
    end
end

if isempty(iAnimals) %Check if any animals are found
    AllAnimals{1} = 'Dummy Subject'; %create dummy animal if nothing else is found
    mkdir([handles.path.base 'Animals\Dummy Subject']) %create folder for default experiment
end

handles.AnimalID.String = AllAnimals; %update AnimalID selection
if handles.AnimalID.Value > length(AllAnimals)
    handles.AnimalID.Value = 1; %reset indicator
end
if ~isempty(handles.AnimalID.Value)
    handles.path.AnimalID = AllAnimals{handles.AnimalID.Value}; %update path for current animal
end

folders = dir([handles.path.base 'Animals\' AllAnimals{handles.AnimalID.Value}]); %find Experiment folders
folders = folders([folders.isdir] & ~strncmpi('.', {folders.name}, 1));
for iExperiments = 1:size(folders,1) %skip first two entries because they contain folders '.' and '..'
    AllExperiments{iExperiments} = folders(iExperiments).name; %get experiment folders
end
if isempty(iExperiments) %Check if any experiments are found
    AllExperiments{1} = 'Default'; %create default experiment if nothing else is found
    mkdir([handles.path.base 'Animals\' AllAnimals{1} '\Default']) %create folder for default experiment
end

handles.ExperimentType.String = AllExperiments; %update experiment type selection
if size(AllExperiments,2) < handles.ExperimentType.Value; handles.ExperimentType.Value = 1; end
handles.path.ExpType = AllExperiments{handles.ExperimentType.Value}; %update path for current experiment type
cPath = [handles.path.base 'Animals\' AllAnimals{handles.AnimalID.Value} '\' AllExperiments{handles.ExperimentType.Value}]; %assign current path

if size(ls([cPath '\' date]),1) > 2 %check if folder for current date exist already and contains data
    Cnt = 1;
    temp = num2cell(ls([cPath '\' date '*']),2); %find folders that contain current date and convert to cell
    while any(strcmp(temp,[date '_' num2str(Cnt)]))
        Cnt = Cnt +1; %update counter until it is ensured that current experiment name is not used already
    end
    handles.path.RecordingID = [date '_' num2str(Cnt)]; %set folder for recording day as the date + neccesarry counter
else
    handles.path.RecordingID = date; %set folder for current recording to recording date
end
handles.RecordingID.String = handles.path.RecordingID; %update GUI
handles.DataPath.String = [cPath '\' handles.path.RecordingID]; %set complete file path for data storage
set(handles.TrialNr,'String','0'); %reset TrialNr
handles.SnapshotTaken = false; %flag for snapshot - has to be taken in order to start data acquisition
handles.CurrentStatus.String = 'Not ready'; %reset status indicator
guidata(handles.WidefieldImager,handles);
function logAnalogData(src, evt, fid, flag)
% Add the time stamp and the data values to data. To write data sequentially,
% transpose the matrix.
% Modified to use an additional flag to stop ongoing data acquistion if
% false. Flag should be a handle to a control that contains a logical
% value.
%
% Example for addding function to a listener when running analog through StartBackground:           
% handles.aListen = handles.aNIdevice.addlistener('DataAvailable', @(src, event)logAnalogData(src,event,aID,handles.AcqusitionStatus)); %listener to stream analog data to disc


if src.IsRunning %only execute while acquisition is still active
    if evt.TimeStamps(1) == 0
        fwrite(fid,3,'double'); %indicate number of single values in the header
        fwrite(fid,evt.TriggerTime,'double'); %write time of acquisition onset on first run
        fwrite(fid,size(evt.Data,2)+1,'double'); %write number of recorded analog channels + timestamps      
        fwrite(fid,inf,'double'); %write number of values to read (set to inf since absolute recording duration is unknown at this point)     
    end
    
    data = [evt.TimeStamps*1000, evt.Data*1000]' ; %convert time to ms and voltage to mV
    fwrite(fid,uint16(data),'uint16');
%     plot(data(1,:),data(2:end,:))

    if ~logical(get(flag, 'value')) %check if acqusition is still active
        src.stop(); %stop recording
    end
end

% --- Executes on selection change in lightMode.
function lightMode_Callback(hObject, eventdata, handles)
% hObject    handle to lightMode (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns lightMode contents as cell array
%        contents{get(hObject,'Value')} returns selected item from lightMode

BlueLight_Callback(handles.BlueLight, [], handles) %switch LED
   
% --- Executes during object creation, after setting all properties.
function lightMode_CreateFcn(hObject, eventdata, handles)
% hObject    handle to lightMode (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on button press in NewAnimal.
function NewAnimal_Callback(hObject, eventdata, handles)
% hObject    handle to NewAnimal (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles = CheckPath(handles); %Check for data path, reset date and trialcount
dPrompt = {'Enter animal ID'};
pName = 'New animal';
newMouse = inputdlg(dPrompt,pName,1,{'New mouse'});
if ~isempty(newMouse)
    mkdir([handles.path.base 'Animals' filesep newMouse{1}])
    
    dPrompt = {'Enter experiment ID'};
    pName = 'New experiment';
    if strcmpi(strtrim(handles.RecordMode.String{handles.RecordMode.Value}),'Mapping')
        newExp = inputdlg(dPrompt,pName,1,{'PhaseMap'});
    elseif strcmpi(strtrim(handles.RecordMode.String{handles.RecordMode.Value}),'Bpod')
        newExp = inputdlg(dPrompt,pName,1,{'SpatialDisc'});
    else
        newExp = inputdlg(dPrompt,pName,1,{'New Experiment'});
    end
    
    mkdir([handles.path.base 'Animals' filesep newMouse{1} filesep newExp{1}])
    
    handles = CheckPath(handles); %Check for data path, reset date and trialcount
    handles.AnimalID.Value = find(ismember(handles.AnimalID.String,newMouse{1}));
    CheckPath(handles);
end

% --- Executes on button press in NewExperiment.
function NewExperiment_Callback(hObject, eventdata, handles)
% hObject    handle to NewExperiment (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

dPrompt = {'Enter experiment ID'};
pName = 'New experiment';
newExp = inputdlg(dPrompt,pName,1,{'New experiment'});
if ~isempty(newExp)
    mkdir([handles.path.base 'Animals' filesep handles.AnimalID.String{handles.AnimalID.Value} filesep newExp{1}])
    
    handles = CheckPath(handles); %Check for data path, reset date and trialcount
    handles.ExperimentType.Value = find(ismember(handles.ExperimentType.String,newExp{1}));
    CheckPath(handles);
end

function FrameRate_Callback(hObject, eventdata, handles)
% hObject    handle to FrameRate (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of FrameRate as text
%        str2double(get(hObject,'String')) returns contents of FrameRate as a double

src = getselectedsource(handles.vidObj);
if str2double(handles.FrameRate.String) > 10 && strcmp(handles.sBinning.String(handles.sBinning.Value),'1')
    answer = questdlg('Spatial binning is set to 1. This could produce too much data to handle. Proceed?');
    if strcmpi(answer,'Yes')
        src.E2ExposureTime = 1000/str2double(handles.FrameRate.String) * 1000; %set current framerate
    end
else
    src.E2ExposureTime = 1000/str2double(handles.FrameRate.String) * 1000; %set current framerate
end

% --- Executes during object creation, after setting all properties.
function FrameRate_CreateFcn(hObject, eventdata, handles)
% hObject    handle to FrameRate (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes when user attempts to close IntrinsicImagerGUI.
function WidefieldImager_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to WidefieldImagerGUI (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

%% move saved files to data save location
temp = ls(handles.path.base);

sFiles = ~cellfun('isempty',((strfind(cellstr(ls(handles.path.base)),'Snapshot_')))); %identify snapshot files
rFiles = ~cellfun('isempty',((strfind(cellstr(ls(handles.path.base)),'ROI.')))); %identify ROI file
aFiles = [temp(sFiles,:);temp(rFiles,:)]; %all files that should be moved

if ~isdir(get(handles.DataPath,'String')) && ~isempty(aFiles) %create data path if not existent and if there is data to be moved
    mkdir(get(handles.DataPath,'String'))
end
for iFiles = 1:size(aFiles,1)
    movefile([handles.path.base aFiles(iFiles,:)],[get(handles.DataPath,'String') '\' aFiles(iFiles,:)]); %move files
end

%% clear running objects
guidata(hObject,handles);
delete(hObject)

% --- Executes during object creation, after setting all properties.
function RecordMode_CreateFcn(hObject, eventdata, handles)
% hObject    handle to RecordMode (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on selection change in driveSelect.
function driveSelect_Callback(hObject, eventdata, handles)
% hObject    handle to driveSelect (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns driveSelect contents as cell array
%        contents{get(hObject,'Value')} returns selected item from driveSelect

a = strfind(handles.DataPath.String,filesep);
cPath = [strtrim(handles.driveSelect.String(handles.driveSelect.Value,:)) fileparts(handles.DataPath.String(a(1):end))];
if ~exist(cPath,'dir')
    mkdir(cPath);
end
handles = CheckPath(handles); %Check for data path, reset date and trialcount

% --- Executes during object creation, after setting all properties.
function driveSelect_CreateFcn(hObject, eventdata, handles)
% hObject    handle to driveSelect (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on button press in lockGUI.
function handles = lockGUI_Callback(hObject, eventdata, handles)
% hObject    handle to lockGUI (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if hObject.Value == 0
    handles.WaitForTrigger.Enable = 'on';
    hObject.String = 'Released';
elseif hObject.Value == 1
    handles.WaitForTrigger.Enable = 'off';
    hObject.String = 'Locked';
end

% --- Executes during object creation, after setting all properties.
function slider2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
set(hObject,'Min',-10)
set(hObject,'Max',10)
set(hObject,'Value',0)
%handles.OptoScan1 = 0;
% --- Executes during object creation, after setting all properties.
function slider3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
set(hObject,'Min',-10)
set(hObject,'Max',10)
set(hObject,'Value',0)
%handles.OptoScan2 = 0;

function edit14_Callback(hObject, eventdata, handles)
% hObject    handle to edit14 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit14 as text
%        str2double(get(hObject,'String')) returns contents of edit14 as a double
%edit14_value = str2double(get(hObject,'String'));
handles.OptoScan1 = str2double(get(hObject,'String'));
if handles.OptoScan1 > 10
    handles.OptoScan1 = 10;
    set(hObject,'String','10')
elseif handles.OptoScan1 < -10
    handles.OptoScan1 = -10;
    set(hObject,'String','-10')
end
set(handles.slider2,'Value',handles.OptoScan1);
%Set the output value to Analog output
handles.OptoScan2 = get(handles.slider3,'Value');
[handles.OptoScan1, handles.OptoScan2]
%outputSingleScan(handles.aNIdeviceOut,[handles.OptoScan1, handles.OptoScan2, 0])
write(handles.aNIdeviceOut,[handles.OptoScan1, handles.OptoScan2, 0])
% --- Executes during object creation, after setting all properties.
function edit14_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit14 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function edit16_Callback(hObject, eventdata, handles)
% hObject    handle to edit16 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit16 as text
%        str2double(get(hObject,'String')) returns contents of edit16 as a double
%edit16_value = str2double(get(hObject,'String'));
handles.OptoScan2 = str2double(get(hObject,'String'));
if handles.OptoScan2 > 10
    handles.OptoScan2 = 10;
    set(hObject,'String','10')
elseif handles.OptoScan2 < -10
    handles.OptoScan2 = -10;
    set(hObject,'String','-10')
end
set(handles.slider3,'Value',handles.OptoScan2);
%Set the output value to Analog output
handles.OptoScan1 = get(handles.slider2,'Value');
[handles.OptoScan1, handles.OptoScan2]
%outputSingleScan(handles.aNIdeviceOut,[handles.OptoScan1, handles.OptoScan2, 0])
write(handles.aNIdeviceOut,[handles.OptoScan1, handles.OptoScan2, 0])
% --- Executes during object creation, after setting all properties.
function edit16_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit16 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit32_Callback(hObject, eventdata, handles)
% hObject    handle to edit32 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.SingleScanPlace = str2double(get(hObject,'String'));
if handles.SingleScanPlace > 27
    handles.SingleScanPlace = 27;
    set(hObject,'String','27')
elseif handles.SingleScanPlace < 1
    handles.SingleScanPlace = 1;
    set(hObject,'String','1')
end
guidata(hObject,handles);

% --- Executes during object creation, after setting all properties.
function edit32_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit32 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
handles.SingleScanPlace = str2double(get(hObject,'String'));
guidata(hObject,handles);


% --- Executes on button press in Single_check.
function Single_check_Callback(hObject, eventdata, handles)
% hObject    handle to Single_check (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of Single_check


% --- Executes on button press in check_right_stim.
function check_right_stim_Callback(hObject, eventdata, handles)
% hObject    handle to check_right_stim (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of check_right_stim


% --- Executes during object deletion, before destroying properties.
function Bregma_Lambda_table_DeleteFcn(hObject, eventdata, handles)
% hObject    handle to Bregma_Lambda_table (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

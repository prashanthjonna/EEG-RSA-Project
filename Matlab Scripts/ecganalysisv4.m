%% ECG Analysis for Hunger Study: Ben Sawyer
    clc; clear all;
%%
    load('EEG_Test_9_22_Sigs_B.mat')
%%  Set Variables
    ECGStart_Time = 3590286;    % Array number in PACT_ECG correlating to PACT_Time*1022
    ECGStop_Time = 3848540;     % Array number in PACT_ECG correlating to PACT_Time*1022
    RIPStart_Time = 222000;     % Array number in PACT_Data_B correlating to PACT_Time_B*680
    RIPStop_Time = 230000;      % Array number in PACT_Data_B correlating to PACT_Time_B*680
    ECGfs = 1000;               % Sampling rate of ECG
    RIPfs = 100;                % Sampling rate of RIP Belt
    gr = 1;                     % gr = 1 print figures, gr = 0 no figures
    %%  Plot RIP Belt Signal  
    PACT_Data_B(1,:) = PACT_Data_B(3,:);
    PACT_Data_B(2,:) = PACT_Data_B(3,:);
    RIPbreathsig = -1*PACT_Data_B(:,1); %breathing
    RIPbreathsig = RIPbreathsig((RIPStart_Time):(RIPStop_Time));
    ts = ((0:numel(RIPbreathsig)-1)/RIPfs);
    figure(1);
    % find the peaks (inhale) and valley (exhale)
    [RIPpos_pks,RIPpos_loc] = findpeaks(RIPbreathsig,'MinPeakDistance',round(3.5*RIPfs));%,'MinPeakHeight',round(0.5*fs));
    [RIPneg_pks,RIPneg_loc] = findpeaks(-RIPbreathsig,'MinPeakDistance',round(3.5*RIPfs));%,'MinPeakHeight',round(0.5*fs));
    % plot the signal
    plot(RIPbreathsig, '-b')
    hold on
    plot(RIPpos_loc,  RIPpos_pks, '^r')
    plot(RIPneg_loc, -RIPneg_pks, 'vg')
    legend('Data', 'inhale', 'exhale','AutoUpdate','off')
    xlim([0 8000]);
    %Calculate the absolute times
    exhale = [ts(1) ts(RIPneg_loc)]; % mark exhale to exhale -> exhale to inhale pk to exhale
    inhale = [ts(1) ts(RIPpos_loc)]; % mark inhale peak
    %intervals between
    breathinterval = diff([0 exhale]);
    for i = 1:length(exhale)
        xline(exhale(:,i)*100,'--k');
%         xline(RIPpos_loc(i),'--r');
    end
    hold off
%%  Load ECG data
    load('EEG_Test_9_22_Sigs_E.mat')
%% Update ECG data Array based on Inhale and Exhale
    inhale_RR = 0;
    exhale_RR = 0;
    ecg = PACT_ECG(:,2);  % 
    ecg = ecg';
    % bandpass filter for Noise cancelation of other sampling frequencies(Filtering)
    f1=0.04;                              % cuttoff low frequency to get rid of baseline wander
    f2=15;                                % cuttoff frequency to discard high frequency noise
    Wn=[f1 f2]*2/ECGfs;                   % cutt off based on ECGfs
    N = 3;                                % order of 3 less processing
    [a,b] = butter(N,Wn);                 % bandpass filtering
    ecg_h = filtfilt(a,b,ecg);
    ecg_h = ecg_h/ max( abs(ecg_h));
    % derivative filter
    b = [1 2 0 -2 -1].*(1/8)*ECGfs;   
    ecg_d = filtfilt(b,1,ecg_h);
    ecg_d = ecg_d/max(ecg_d);
    for i = 2:length(RIPpos_pks)%(length(exhale)-20)
        % Update the array index based on inhale and exhale 
        inhaleECGStart = ECGStart_Time + (exhale(:,i)*1000);    % Array number in PACT_ECG correlating to PACT_Time
        inhaleECGStop = inhaleECGStart + (inhale(:,i+1)*1000);  % Array number in PACT_ECG correlating to PACT_Time
        exhaleECGStart = ECGStart_Time + (inhale(:,i)*1000);    % Array number in PACT_ECG correlating to PACT_Time
        exhaleECGStop = exhaleECGStart + (exhale(:,i+1)*1000);    % Array number in PACT_ECG correlating to PACT_Time
        % Set array index for filtered ecg signal
        inhaleecg = ecg_d(inhaleECGStart:inhaleECGStop); % PACT_ECG array location = PACT_Time / 0.001. 
        exhaleecg = ecg_d(exhaleECGStart:exhaleECGStop); % PACT_ECG array location = PACT_Time / 0.001.
        % Find the RR peaks per breath (Inhale and Exhale)
        [inhalepos_pks,ECGpos_loc] = findpeaks(inhaleecg,'MinPeakDistance',round(0.5*ECGfs));
%         [qrs_amp_raw,qrs_i_raw,delay,ecg,ecg_m,ecg_h,ecg_d,qrs_i,qrs_c] = pan_tompkin(inhaleecg,ECGfs,gr);
        inhale_RR = inhale_RR+length(inhalepos_pks);
        [exhalepos_pks,ECGpos_loc] = findpeaks(exhaleecg,'MinPeakDistance',round(0.5*ECGfs));
%         [qrs_amp_raw,qrs_i_raw,delay,ecg,ecg_m,ecg_h,ecg_d,qrs_i,qrs_c] = pan_tompkin(exhaleecg,ECGfs,gr);
        exhale_RR = exhale_RR+length(exhalepos_pks);
    end
    % Calculate the RSA based in the Inhale and Exhale RR peaks
    RSA_avg = inhale_RR/exhale_RR
%% Plot ECG RR peaks
    fullecg = ecg(ECGStart_Time:ECGStop_Time); % Array number in PACT_ECG correlating to PACT_Time
    [qrs_amp_raw,qrs_i_raw,delay,ecg,ecg_m,ecg_h,ecg_d,qrs_i,qrs_c] = pan_tompkin(fullecg,ECGfs);
%     [qrs_amp_raw,qrs_i_raw,delay,ecg,ecg_m,ecg_h,ecg_d,qrs_i,qrs_c] = pan_tompkin(exhaleecg,ECGfs,gr);
    figure(3);   
%     b = [1 2 0 -2 -1].*(1/8)*fs;   
%     ecg_d = filtfilt(b,1,ecg_h);
%     ecg_d = ecg_d/max(ecg_d);
    [ECGpos_pks,ECGpos_loc] = findpeaks(ecg_d,'MinPeakDistance',round(0.5*ECGfs));%,'MinPeakHeight',round(0.5*fs));
%     [ECGneg_pks,ECGneg_loc] = findpeaks(-ecg_d,'MinPeakDistance',round(1*ECGfs));%,'MinPeakHeight',round(0.5*fs));
    plot(ecg_d, '-b')
    hold on
    plot(ECGpos_loc,  ECGpos_pks, '^r')
%     plot(ECGneg_loc, -ECGneg_pks, 'vg')
    hold off
    title('RR Peaks Filtered Signal');
    % Calculate the Heartrate and ECG Power of signal
    Heartrate = length(ECGpos_pks)
    Ecgpower_m = obw(ecg_m,ECGfs);
    Ecgpower_h = obw(ecg_h,ECGfs);
    Ecgpower = Ecgpower_h/Ecgpower_m
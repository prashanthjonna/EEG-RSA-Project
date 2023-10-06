clc
close all
clear all
%%Load Data from file
files = {'H7_EC_pre.edf', 'H7_EC_post.edf','H8_EC_pre.edf', 'H8_EC_post.edf'};

for x=1:length(files)
    
    fileName = files{x};
    [hdr, record] = edfread(fileName); 
    
    %trim off the first 3 seconds and the end because this filestops recording at 299587, this should probably be automated
    record = record(:, 5*1000:299587); 
    
    dataToExportToExcel = zeros(66, 3);
    %%process each electrode
    for i=1:length(record(:, 1))
        frequency = record(i, :);
        frequencyAlpha = bandpass(frequency, [8 12], 1000); %Filter for alpha frequency
        frequencyTheta = bandpass(frequency, [4 8], 1000);
        frequencyDelta  = bandpass(frequency, [0.02 4], 1000);
        frequencyBeta = bandpass(frequency, [12.5 30], 1000);
        
        %-----------------------------------------------------------------------------
        t = 0:0.001 : (length(frequencyAlpha)-1)/1000;%create an x-axis given 1000 samples per second
        f = fft(frequencyAlpha);
        fs      = 1/((length(frequencyAlpha)-1)/1000);           % frequency stepping
        freq    = 0:fs:1000/2;   % Create frequency vector given input signal sampling
        fp = f(1:length(t)/2+1);            % positive half of frequencies
        fp2Alpha = (1/(1000*length(t))) * abs(fp);
        fp2Alpha(2:end-1) = 2*fp2Alpha(2:end-1);      % Double amplitude of frequencies which repeat
        
       %--------------------------------------------------------------------------------

       %-----------------------------------------------------------------------------
        t = 0:0.001 : (length(frequencyTheta)-1)/1000;%create an x-axis given 1000 samples per second
        f = fft(frequencyTheta);
        fs      = 1/((length(frequencyTheta)-1)/1000);           % frequency stepping
        freq    = 0:fs:1000/2;   % Create frequency vector given input signal sampling
        fp = f(1:length(t)/2+1);            % positive half of frequencies
        fp2Theta = (1/(1000*length(t))) * abs(fp);
        fp2Theta(2:end-1) = 2*fp2Theta(2:end-1);      % Double amplitude of frequencies which repeat
      %-----------------------------------------------------------------------------
        t = 0:0.001 : (length(frequencyDelta)-1)/1000;%create an x-axis given 1000 samples per second
        f = fft(frequencyDelta);
        fs      = 1/((length(frequencyDelta)-1)/1000);           % frequency stepping
        freq    = 0:fs:1000/2;   % Create frequency vector given input signal sampling
        fp = f(1:length(t)/2+1);            % positive half of frequencies
        fp2Delta = (1/(1000*length(t))) * abs(fp);
        fp2Delta(2:end-1) = 2*fp2Delta(2:end-1);      % Double amplitude of frequencies which repeat
      %-----------------------------------------------------------------------------
        t = 0:0.001 : (length(frequencyBeta)-1)/1000;%create an x-axis given 1000 samples per second
        f = fft(frequencyBeta);
        fs      = 1/((length(frequencyBeta)-1)/1000);           % frequency stepping
        freq    = 0:fs:1000/2;   % Create frequency vector given input signal sampling
        fp = f(1:length(t)/2+1);            % positive half of frequencies
        fp2Beta = (1/(1000*length(t))) * abs(fp);
        fp2Beta(2:end-1) = 2*fp2Beta(2:end-1);      % Double amplitude of frequencies which repeat
      %------------------------------------------------------------------------------------------
      
      
        [valueDelta, indexDelta] = max(abs(fp2Delta)); %find max value
        valueDelta = valueDelta*valueDelta; %sqaure the max value
        indexDelta = freq(indexDelta); %find the frequency that corrensponds to the max value
        
        [valueTheta, indexTheta] = max(abs(fp2Theta)); %find max value
        valueTheta = valueTheta*valueTheta; %sqaure the max value
        indexTheta = freq(indexTheta); %find the frequency that corrensponds to the max value
        
        [valueAlpha, indexAlpha] = max(abs(fp2Alpha)); %find max value
        valueAlpha = valueAlpha*valueAlpha; %sqaure the max value
        indexAlpha = freq(indexAlpha); %find the frequency that corrensponds to the max value
        
        [valueBeta, indexBeta] = max(abs(fp2Beta)); %find max value
        valueBeta = valueBeta*valueBeta; %sqaure the max value
        indexBeta = freq(indexBeta); %find the frequency that corrensponds to the max value
        
        disp(['File Name: ' fileName ' Max Square Amplitude: ' num2str(valueAlpha,'%0.12f') ' Frequency: ' num2str(indexAlpha, '%0.8f') ' Electrode: ' num2str(i, '%0.1f')]);
        dataToExportToExcel(i, 2) = valueDelta;
        dataToExportToExcel(i, 3) = indexDelta;
        
        dataToExportToExcel(i, 4) = valueTheta;
        dataToExportToExcel(i, 5) = indexTheta;
        
        dataToExportToExcel(i, 6) = valueAlpha;
        dataToExportToExcel(i, 7) = indexAlpha;
        
        dataToExportToExcel(i, 8) = valueBeta;
        dataToExportToExcel(i, 9) = indexBeta;
        
        dataToExportToExcel(i, 1) = i;
        toexcel(dataToExportToExcel)

    end

    %Change filePath (C:\Users\Name\Documents\) to where ever the output file needs stored
    outputFolderPath = '/Users/prashanthjonna/Projects/EEG-RSA UA Project 2022/';
    toexcel('Saveas', strcat(strcat(outputFolderPath, fileName(1:end-4)), 'FrequencysPowerAndPeak'));
    
    % ------------------------------------------------------------------------------------
    
    fileName = files{x};
    [hdr, record] = edfread(fileName); 
    
    %trim off the first 3 seconds and the end because this filestops recording at 299587, this should probably be automated
    record = record(:, 5*1000:5*1000+60000); 
    
    dataToExportToExcel = zeros(66, 3);
    %%process each electrode
    for i=1:length(record(:, 1))
        frequency = record(i, :);
        frequencyAlpha = bandpass(frequency, [8 12], 1000); %Filter for alpha frequency
        frequencyTheta = bandpass(frequency, [4 8], 1000);
        frequencyDelta  = bandpass(frequency, [0.02 4], 1000);
        frequencyBeta = bandpass(frequency, [12.5 30], 1000);
        
        %-----------------------------------------------------------------------------
        t = 0:0.001 : (length(frequencyAlpha)-1)/1000;%create an x-axis given 1000 samples per second
        f = fft(frequencyAlpha);
        fs      = 1/((length(frequencyAlpha)-1)/1000);           % frequency stepping
        freq    = 0:fs:1000/2;   % Create frequency vector given input signal sampling
        fp = f(1:length(t)/2+1);            % positive half of frequencies
        fp2Alpha = (1/(1000*length(t))) * abs(fp);
        fp2Alpha(2:end-1) = 2*fp2Alpha(2:end-1);      % Double amplitude of frequencies which repeat
        
       %--------------------------------------------------------------------------------

       %-----------------------------------------------------------------------------
        t = 0:0.001 : (length(frequencyTheta)-1)/1000;%create an x-axis given 1000 samples per second
        f = fft(frequencyTheta);
        fs      = 1/((length(frequencyTheta)-1)/1000);           % frequency stepping
        freq    = 0:fs:1000/2;   % Create frequency vector given input signal sampling
        fp = f(1:length(t)/2+1);            % positive half of frequencies
        fp2Theta = (1/(1000*length(t))) * abs(fp);
        fp2Theta(2:end-1) = 2*fp2Theta(2:end-1);      % Double amplitude of frequencies which repeat
      %-----------------------------------------------------------------------------
        t = 0:0.001 : (length(frequencyDelta)-1)/1000;%create an x-axis given 1000 samples per second
        f = fft(frequencyDelta);
        fs      = 1/((length(frequencyDelta)-1)/1000);           % frequency stepping
        freq    = 0:fs:1000/2;   % Create frequency vector given input signal sampling
        fp = f(1:length(t)/2+1);            % positive half of frequencies
        fp2Delta = (1/(1000*length(t))) * abs(fp);
        fp2Delta(2:end-1) = 2*fp2Delta(2:end-1);      % Double amplitude of frequencies which repeat
      %-----------------------------------------------------------------------------
        t = 0:0.001 : (length(frequencyBeta)-1)/1000;%create an x-axis given 1000 samples per second
        f = fft(frequencyBeta);
        fs      = 1/((length(frequencyBeta)-1)/1000);           % frequency stepping
        freq    = 0:fs:1000/2;   % Create frequency vector given input signal sampling
        fp = f(1:length(t)/2+1);            % positive half of frequencies
        fp2Beta = (1/(1000*length(t))) * abs(fp);
        fp2Beta(2:end-1) = 2*fp2Beta(2:end-1);      % Double amplitude of frequencies which repeat
      %------------------------------------------------------------------------------------------
      
      
        [valueDelta, indexDelta] = max(abs(fp2Delta)); %find max value
        valueDelta = valueDelta*valueDelta; %sqaure the max value
        indexDelta = freq(indexDelta); %find the frequency that corrensponds to the max value
        
        [valueTheta, indexTheta] = max(abs(fp2Theta)); %find max value
        valueTheta = valueTheta*valueTheta; %sqaure the max value
        indexTheta = freq(indexTheta); %find the frequency that corrensponds to the max value
        
        [valueAlpha, indexAlpha] = max(abs(fp2Alpha)); %find max value
        valueAlpha = valueAlpha*valueAlpha; %sqaure the max value
        indexAlpha = freq(indexAlpha); %find the frequency that corrensponds to the max value
        
        [valueBeta, indexBeta] = max(abs(fp2Beta)); %find max value
        valueBeta = valueBeta*valueBeta; %sqaure the max value
        indexBeta = freq(indexBeta); %find the frequency that corrensponds to the max value
        
        disp(['File Name: ' fileName ' Max Square Amplitude: ' num2str(valueAlpha,'%0.12f') ' Frequency: ' num2str(indexAlpha, '%0.8f') ' Electrode: ' num2str(i, '%0.1f')]);
        dataToExportToExcel(i, 2) = valueDelta;
        dataToExportToExcel(i, 3) = indexDelta;
        
        dataToExportToExcel(i, 4) = valueTheta;
        dataToExportToExcel(i, 5) = indexTheta;
        
        dataToExportToExcel(i, 6) = valueAlpha;
        dataToExportToExcel(i, 7) = indexAlpha;
        
        dataToExportToExcel(i, 8) = valueBeta;
        dataToExportToExcel(i, 9) = indexBeta;
        
        dataToExportToExcel(i, 1) = i;
        toexcel(dataToExportToExcel)

    end

    %Change filePath (C:\Users\Name\Documents\) to where ever the output file needs stored
    outputFolderPath = '/Users/prashanthjonna/Projects/EEG-RSA UA Project 2022/';
    toexcel('Saveas', strcat(strcat(outputFolderPath, fileName(1:end-4)), 'FrequencysPowerAndPeak_Clip1'));
    
    % ------------------------------------------------------------------------------------
    
    % ------------------------------------------------------------------------------------
    
    fileName = files{x};
    [hdr, record] = edfread(fileName); 
    
    %trim off the first 3 seconds and the end because this filestops recording at 299587, this should probably be automated
    record = record(:, 5*1000+60000:5*1000+120000);
    
    dataToExportToExcel = zeros(66, 3);
    %%process each electrode
    for i=1:length(record(:, 1))
        frequency = record(i, :);
        frequencyAlpha = bandpass(frequency, [8 12], 1000); %Filter for alpha frequency
        frequencyTheta = bandpass(frequency, [4 8], 1000);
        frequencyDelta  = bandpass(frequency, [0.02 4], 1000);
        frequencyBeta = bandpass(frequency, [12.5 30], 1000);
        
        %-----------------------------------------------------------------------------
        t = 0:0.001 : (length(frequencyAlpha)-1)/1000;%create an x-axis given 1000 samples per second
        f = fft(frequencyAlpha);
        fs      = 1/((length(frequencyAlpha)-1)/1000);           % frequency stepping
        freq    = 0:fs:1000/2;   % Create frequency vector given input signal sampling
        fp = f(1:length(t)/2+1);            % positive half of frequencies
        fp2Alpha = (1/(1000*length(t))) * abs(fp);
        fp2Alpha(2:end-1) = 2*fp2Alpha(2:end-1);      % Double amplitude of frequencies which repeat
        
       %--------------------------------------------------------------------------------

       %-----------------------------------------------------------------------------
        t = 0:0.001 : (length(frequencyTheta)-1)/1000;%create an x-axis given 1000 samples per second
        f = fft(frequencyTheta);
        fs      = 1/((length(frequencyTheta)-1)/1000);           % frequency stepping
        freq    = 0:fs:1000/2;   % Create frequency vector given input signal sampling
        fp = f(1:length(t)/2+1);            % positive half of frequencies
        fp2Theta = (1/(1000*length(t))) * abs(fp);
        fp2Theta(2:end-1) = 2*fp2Theta(2:end-1);      % Double amplitude of frequencies which repeat
      %-----------------------------------------------------------------------------
        t = 0:0.001 : (length(frequencyDelta)-1)/1000;%create an x-axis given 1000 samples per second
        f = fft(frequencyDelta);
        fs      = 1/((length(frequencyDelta)-1)/1000);           % frequency stepping
        freq    = 0:fs:1000/2;   % Create frequency vector given input signal sampling
        fp = f(1:length(t)/2+1);            % positive half of frequencies
        fp2Delta = (1/(1000*length(t))) * abs(fp);
        fp2Delta(2:end-1) = 2*fp2Delta(2:end-1);      % Double amplitude of frequencies which repeat
      %-----------------------------------------------------------------------------
        t = 0:0.001 : (length(frequencyBeta)-1)/1000;%create an x-axis given 1000 samples per second
        f = fft(frequencyBeta);
        fs      = 1/((length(frequencyBeta)-1)/1000);           % frequency stepping
        freq    = 0:fs:1000/2;   % Create frequency vector given input signal sampling
        fp = f(1:length(t)/2+1);            % positive half of frequencies
        fp2Beta = (1/(1000*length(t))) * abs(fp);
        fp2Beta(2:end-1) = 2*fp2Beta(2:end-1);      % Double amplitude of frequencies which repeat
      %------------------------------------------------------------------------------------------
      
      
        [valueDelta, indexDelta] = max(abs(fp2Delta)); %find max value
        valueDelta = valueDelta*valueDelta; %sqaure the max value
        indexDelta = freq(indexDelta); %find the frequency that corrensponds to the max value
        
        [valueTheta, indexTheta] = max(abs(fp2Theta)); %find max value
        valueTheta = valueTheta*valueTheta; %sqaure the max value
        indexTheta = freq(indexTheta); %find the frequency that corrensponds to the max value
        
        [valueAlpha, indexAlpha] = max(abs(fp2Alpha)); %find max value
        valueAlpha = valueAlpha*valueAlpha; %sqaure the max value
        indexAlpha = freq(indexAlpha); %find the frequency that corrensponds to the max value
        
        [valueBeta, indexBeta] = max(abs(fp2Beta)); %find max value
        valueBeta = valueBeta*valueBeta; %sqaure the max value
        indexBeta = freq(indexBeta); %find the frequency that corrensponds to the max value
        
        disp(['File Name: ' fileName ' Max Square Amplitude: ' num2str(valueAlpha,'%0.12f') ' Frequency: ' num2str(indexAlpha, '%0.8f') ' Electrode: ' num2str(i, '%0.1f')]);
        dataToExportToExcel(i, 2) = valueDelta;
        dataToExportToExcel(i, 3) = indexDelta;
        
        dataToExportToExcel(i, 4) = valueTheta;
        dataToExportToExcel(i, 5) = indexTheta;
        
        dataToExportToExcel(i, 6) = valueAlpha;
        dataToExportToExcel(i, 7) = indexAlpha;
        
        dataToExportToExcel(i, 8) = valueBeta;
        dataToExportToExcel(i, 9) = indexBeta;
        
        dataToExportToExcel(i, 1) = i;
        toexcel(dataToExportToExcel)

    end

    %Change filePath (C:\Users\Name\Documents\) to where ever the output file needs stored
    outputFolderPath = '/Users/prashanthjonna/Projects/EEG-RSA UA Project 2022/';
    toexcel('Saveas', strcat(strcat(outputFolderPath, fileName(1:end-4)), 'FrequencysPowerAndPeak_Clip2'));
    
    % ------------------------------------------------------------------------------------
    
    % ------------------------------------------------------------------------------------
    
    fileName = files{x};
    [hdr, record] = edfread(fileName); 
    
    %trim off the first 3 seconds and the end because this filestops recording at 299587, this should probably be automated
    record = record(:, 5*1000+120000:5*1000+180000);
    
    dataToExportToExcel = zeros(66, 3);
    %%process each electrode
    for i=1:length(record(:, 1))
        frequency = record(i, :);
        frequencyAlpha = bandpass(frequency, [8 12], 1000); %Filter for alpha frequency
        frequencyTheta = bandpass(frequency, [4 8], 1000);
        frequencyDelta  = bandpass(frequency, [0.02 4], 1000);
        frequencyBeta = bandpass(frequency, [12.5 30], 1000);
        
        %-----------------------------------------------------------------------------
        t = 0:0.001 : (length(frequencyAlpha)-1)/1000;%create an x-axis given 1000 samples per second
        f = fft(frequencyAlpha);
        fs      = 1/((length(frequencyAlpha)-1)/1000);           % frequency stepping
        freq    = 0:fs:1000/2;   % Create frequency vector given input signal sampling
        fp = f(1:length(t)/2+1);            % positive half of frequencies
        fp2Alpha = (1/(1000*length(t))) * abs(fp);
        fp2Alpha(2:end-1) = 2*fp2Alpha(2:end-1);      % Double amplitude of frequencies which repeat
        
       %--------------------------------------------------------------------------------

       %-----------------------------------------------------------------------------
        t = 0:0.001 : (length(frequencyTheta)-1)/1000;%create an x-axis given 1000 samples per second
        f = fft(frequencyTheta);
        fs      = 1/((length(frequencyTheta)-1)/1000);           % frequency stepping
        freq    = 0:fs:1000/2;   % Create frequency vector given input signal sampling
        fp = f(1:length(t)/2+1);            % positive half of frequencies
        fp2Theta = (1/(1000*length(t))) * abs(fp);
        fp2Theta(2:end-1) = 2*fp2Theta(2:end-1);      % Double amplitude of frequencies which repeat
      %-----------------------------------------------------------------------------
        t = 0:0.001 : (length(frequencyDelta)-1)/1000;%create an x-axis given 1000 samples per second
        f = fft(frequencyDelta);
        fs      = 1/((length(frequencyDelta)-1)/1000);           % frequency stepping
        freq    = 0:fs:1000/2;   % Create frequency vector given input signal sampling
        fp = f(1:length(t)/2+1);            % positive half of frequencies
        fp2Delta = (1/(1000*length(t))) * abs(fp);
        fp2Delta(2:end-1) = 2*fp2Delta(2:end-1);      % Double amplitude of frequencies which repeat
      %-----------------------------------------------------------------------------
        t = 0:0.001 : (length(frequencyBeta)-1)/1000;%create an x-axis given 1000 samples per second
        f = fft(frequencyBeta);
        fs      = 1/((length(frequencyBeta)-1)/1000);           % frequency stepping
        freq    = 0:fs:1000/2;   % Create frequency vector given input signal sampling
        fp = f(1:length(t)/2+1);            % positive half of frequencies
        fp2Beta = (1/(1000*length(t))) * abs(fp);
        fp2Beta(2:end-1) = 2*fp2Beta(2:end-1);      % Double amplitude of frequencies which repeat
      %------------------------------------------------------------------------------------------
      
      
        [valueDelta, indexDelta] = max(abs(fp2Delta)); %find max value
        valueDelta = valueDelta*valueDelta; %sqaure the max value
        indexDelta = freq(indexDelta); %find the frequency that corrensponds to the max value
        
        [valueTheta, indexTheta] = max(abs(fp2Theta)); %find max value
        valueTheta = valueTheta*valueTheta; %sqaure the max value
        indexTheta = freq(indexTheta); %find the frequency that corrensponds to the max value
        
        [valueAlpha, indexAlpha] = max(abs(fp2Alpha)); %find max value
        valueAlpha = valueAlpha*valueAlpha; %sqaure the max value
        indexAlpha = freq(indexAlpha); %find the frequency that corrensponds to the max value
        
        [valueBeta, indexBeta] = max(abs(fp2Beta)); %find max value
        valueBeta = valueBeta*valueBeta; %sqaure the max value
        indexBeta = freq(indexBeta); %find the frequency that corrensponds to the max value
        
        disp(['File Name: ' fileName ' Max Square Amplitude: ' num2str(valueAlpha,'%0.12f') ' Frequency: ' num2str(indexAlpha, '%0.8f') ' Electrode: ' num2str(i, '%0.1f')]);
        dataToExportToExcel(i, 2) = valueDelta;
        dataToExportToExcel(i, 3) = indexDelta;
        
        dataToExportToExcel(i, 4) = valueTheta;
        dataToExportToExcel(i, 5) = indexTheta;
        
        dataToExportToExcel(i, 6) = valueAlpha;
        dataToExportToExcel(i, 7) = indexAlpha;
        
        dataToExportToExcel(i, 8) = valueBeta;
        dataToExportToExcel(i, 9) = indexBeta;
        
        dataToExportToExcel(i, 1) = i;
        toexcel(dataToExportToExcel)

    end

    %Change filePath (C:\Users\Name\Documents\) to where ever the output file needs stored
    outputFolderPath = '/Users/prashanthjonna/Projects/EEG-RSA UA Project 2022/';
    toexcel('Saveas', strcat(strcat(outputFolderPath, fileName(1:end-4)), 'FrequencysPowerAndPeak_Clip3'));
    
    % ------------------------------------------------------------------------------------
    
    % ------------------------------------------------------------------------------------
    
    fileName = files{x};
    [hdr, record] = edfread(fileName); 
    
    %trim off the first 3 seconds and the end because this filestops recording at 299587, this should probably be automated
    record = record(:, 5*1000+180000:5*1000+240000);
    
    dataToExportToExcel = zeros(66, 3);
    %%process each electrode
    for i=1:length(record(:, 1))
        frequency = record(i, :);
        frequencyAlpha = bandpass(frequency, [8 12], 1000); %Filter for alpha frequency
        frequencyTheta = bandpass(frequency, [4 8], 1000);
        frequencyDelta  = bandpass(frequency, [0.02 4], 1000);
        frequencyBeta = bandpass(frequency, [12.5 30], 1000);
        
        %-----------------------------------------------------------------------------
        t = 0:0.001 : (length(frequencyAlpha)-1)/1000;%create an x-axis given 1000 samples per second
        f = fft(frequencyAlpha);
        fs      = 1/((length(frequencyAlpha)-1)/1000);           % frequency stepping
        freq    = 0:fs:1000/2;   % Create frequency vector given input signal sampling
        fp = f(1:length(t)/2+1);            % positive half of frequencies
        fp2Alpha = (1/(1000*length(t))) * abs(fp);
        fp2Alpha(2:end-1) = 2*fp2Alpha(2:end-1);      % Double amplitude of frequencies which repeat
        
       %--------------------------------------------------------------------------------

       %-----------------------------------------------------------------------------
        t = 0:0.001 : (length(frequencyTheta)-1)/1000;%create an x-axis given 1000 samples per second
        f = fft(frequencyTheta);
        fs      = 1/((length(frequencyTheta)-1)/1000);           % frequency stepping
        freq    = 0:fs:1000/2;   % Create frequency vector given input signal sampling
        fp = f(1:length(t)/2+1);            % positive half of frequencies
        fp2Theta = (1/(1000*length(t))) * abs(fp);
        fp2Theta(2:end-1) = 2*fp2Theta(2:end-1);      % Double amplitude of frequencies which repeat
      %-----------------------------------------------------------------------------
        t = 0:0.001 : (length(frequencyDelta)-1)/1000;%create an x-axis given 1000 samples per second
        f = fft(frequencyDelta);
        fs      = 1/((length(frequencyDelta)-1)/1000);           % frequency stepping
        freq    = 0:fs:1000/2;   % Create frequency vector given input signal sampling
        fp = f(1:length(t)/2+1);            % positive half of frequencies
        fp2Delta = (1/(1000*length(t))) * abs(fp);
        fp2Delta(2:end-1) = 2*fp2Delta(2:end-1);      % Double amplitude of frequencies which repeat
      %-----------------------------------------------------------------------------
        t = 0:0.001 : (length(frequencyBeta)-1)/1000;%create an x-axis given 1000 samples per second
        f = fft(frequencyBeta);
        fs      = 1/((length(frequencyBeta)-1)/1000);           % frequency stepping
        freq    = 0:fs:1000/2;   % Create frequency vector given input signal sampling
        fp = f(1:length(t)/2+1);            % positive half of frequencies
        fp2Beta = (1/(1000*length(t))) * abs(fp);
        fp2Beta(2:end-1) = 2*fp2Beta(2:end-1);      % Double amplitude of frequencies which repeat
      %------------------------------------------------------------------------------------------
      
      
        [valueDelta, indexDelta] = max(abs(fp2Delta)); %find max value
        valueDelta = valueDelta*valueDelta; %sqaure the max value
        indexDelta = freq(indexDelta); %find the frequency that corrensponds to the max value
        
        [valueTheta, indexTheta] = max(abs(fp2Theta)); %find max value
        valueTheta = valueTheta*valueTheta; %sqaure the max value
        indexTheta = freq(indexTheta); %find the frequency that corrensponds to the max value
        
        [valueAlpha, indexAlpha] = max(abs(fp2Alpha)); %find max value
        valueAlpha = valueAlpha*valueAlpha; %sqaure the max value
        indexAlpha = freq(indexAlpha); %find the frequency that corrensponds to the max value
        
        [valueBeta, indexBeta] = max(abs(fp2Beta)); %find max value
        valueBeta = valueBeta*valueBeta; %sqaure the max value
        indexBeta = freq(indexBeta); %find the frequency that corrensponds to the max value
        
        disp(['File Name: ' fileName ' Max Square Amplitude: ' num2str(valueAlpha,'%0.12f') ' Frequency: ' num2str(indexAlpha, '%0.8f') ' Electrode: ' num2str(i, '%0.1f')]);
        dataToExportToExcel(i, 2) = valueDelta;
        dataToExportToExcel(i, 3) = indexDelta;
        
        dataToExportToExcel(i, 4) = valueTheta;
        dataToExportToExcel(i, 5) = indexTheta;
        
        dataToExportToExcel(i, 6) = valueAlpha;
        dataToExportToExcel(i, 7) = indexAlpha;
        
        dataToExportToExcel(i, 8) = valueBeta;
        dataToExportToExcel(i, 9) = indexBeta;
        
        dataToExportToExcel(i, 1) = i;
        toexcel(dataToExportToExcel)

    end

    %Change filePath (C:\Users\Name\Documents\) to where ever the output file needs stored
    outputFolderPath = '/Users/prashanthjonna/Projects/EEG-RSA UA Project 2022/';
    toexcel('Saveas', strcat(strcat(outputFolderPath, fileName(1:end-4)), 'FrequencysPowerAndPeak_Clip4'));
    
    % ------------------------------------------------------------------------------------
    
    % ------------------------------------------------------------------------------------
    
    fileName = files{x};
    [hdr, record] = edfread(fileName); 
    
    %trim off the first 3 seconds and the end because this filestops recording at 299587, this should probably be automated
    record = record(:, 5*1000+240000:end);
    
    dataToExportToExcel = zeros(66, 3);
    %%process each electrode
    for i=1:length(record(:, 1))
        frequency = record(i, :);
        frequencyAlpha = bandpass(frequency, [8 12], 1000); %Filter for alpha frequency
        frequencyTheta = bandpass(frequency, [4 8], 1000);
        frequencyDelta  = bandpass(frequency, [0.02 4], 1000);
        frequencyBeta = bandpass(frequency, [12.5 30], 1000);
        
        %-----------------------------------------------------------------------------
        t = 0:0.001 : (length(frequencyAlpha)-1)/1000;%create an x-axis given 1000 samples per second
        f = fft(frequencyAlpha);
        fs      = 1/((length(frequencyAlpha)-1)/1000);           % frequency stepping
        freq    = 0:fs:1000/2;   % Create frequency vector given input signal sampling
        fp = f(1:length(t)/2+1);            % positive half of frequencies
        fp2Alpha = (1/(1000*length(t))) * abs(fp);
        fp2Alpha(2:end-1) = 2*fp2Alpha(2:end-1);      % Double amplitude of frequencies which repeat
        
       %--------------------------------------------------------------------------------

       %-----------------------------------------------------------------------------
        t = 0:0.001 : (length(frequencyTheta)-1)/1000;%create an x-axis given 1000 samples per second
        f = fft(frequencyTheta);
        fs      = 1/((length(frequencyTheta)-1)/1000);           % frequency stepping
        freq    = 0:fs:1000/2;   % Create frequency vector given input signal sampling
        fp = f(1:length(t)/2+1);            % positive half of frequencies
        fp2Theta = (1/(1000*length(t))) * abs(fp);
        fp2Theta(2:end-1) = 2*fp2Theta(2:end-1);      % Double amplitude of frequencies which repeat
      %-----------------------------------------------------------------------------
        t = 0:0.001 : (length(frequencyDelta)-1)/1000;%create an x-axis given 1000 samples per second
        f = fft(frequencyDelta);
        fs      = 1/((length(frequencyDelta)-1)/1000);           % frequency stepping
        freq    = 0:fs:1000/2;   % Create frequency vector given input signal sampling
        fp = f(1:length(t)/2+1);            % positive half of frequencies
        fp2Delta = (1/(1000*length(t))) * abs(fp);
        fp2Delta(2:end-1) = 2*fp2Delta(2:end-1);      % Double amplitude of frequencies which repeat
      %-----------------------------------------------------------------------------
        t = 0:0.001 : (length(frequencyBeta)-1)/1000;%create an x-axis given 1000 samples per second
        f = fft(frequencyBeta);
        fs      = 1/((length(frequencyBeta)-1)/1000);           % frequency stepping
        freq    = 0:fs:1000/2;   % Create frequency vector given input signal sampling
        fp = f(1:length(t)/2+1);            % positive half of frequencies
        fp2Beta = (1/(1000*length(t))) * abs(fp);
        fp2Beta(2:end-1) = 2*fp2Beta(2:end-1);      % Double amplitude of frequencies which repeat
      %------------------------------------------------------------------------------------------
      
      
        [valueDelta, indexDelta] = max(abs(fp2Delta)); %find max value
        valueDelta = valueDelta*valueDelta; %sqaure the max value
        indexDelta = freq(indexDelta); %find the frequency that corrensponds to the max value
        
        [valueTheta, indexTheta] = max(abs(fp2Theta)); %find max value
        valueTheta = valueTheta*valueTheta; %sqaure the max value
        indexTheta = freq(indexTheta); %find the frequency that corrensponds to the max value
        
        [valueAlpha, indexAlpha] = max(abs(fp2Alpha)); %find max value
        valueAlpha = valueAlpha*valueAlpha; %sqaure the max value
        indexAlpha = freq(indexAlpha); %find the frequency that corrensponds to the max value
        
        [valueBeta, indexBeta] = max(abs(fp2Beta)); %find max value
        valueBeta = valueBeta*valueBeta; %sqaure the max value
        indexBeta = freq(indexBeta); %find the frequency that corrensponds to the max value
        
        disp(['File Name: ' fileName ' Max Square Amplitude: ' num2str(valueAlpha,'%0.12f') ' Frequency: ' num2str(indexAlpha, '%0.8f') ' Electrode: ' num2str(i, '%0.1f')]);
        dataToExportToExcel(i, 2) = valueDelta;
        dataToExportToExcel(i, 3) = indexDelta;
        
        dataToExportToExcel(i, 4) = valueTheta;
        dataToExportToExcel(i, 5) = indexTheta;
        
        dataToExportToExcel(i, 6) = valueAlpha;
        dataToExportToExcel(i, 7) = indexAlpha;
        
        dataToExportToExcel(i, 8) = valueBeta;
        dataToExportToExcel(i, 9) = indexBeta;
        
        dataToExportToExcel(i, 1) = i;
        toexcel(dataToExportToExcel)

    end

    %Change filePath (C:\Users\Name\Documents\) to where ever the output file needs stored
    outputFolderPath = '/Users/prashanthjonna/Projects/EEG-RSA UA Project 2022/';
    toexcel('Saveas', strcat(strcat(outputFolderPath, fileName(1:end-4)), 'FrequencysPowerAndPeak_Clip5'));
    
    
end




clear all; clc;
%%
fs_ecg = 1000;

Folder = 'Extracted data';
folder = fullfile(pwd, Folder);%Save results
if(~isdir(folder))
    mkdir(folder);
end

FileName = ['C:\Users\esazo\Box\EEG-Hunger_pilot_data_collection\PACT data\2022-02-28 Tests\PACT - 1\000E.bin'];
FileName = ['C:\Users\esazo\Box\EEG-Hunger_pilot_data_collection\PACT data\2022-02-28 Tests\PACT - 2\001E.bin'];
FileName = 'C:\Users\esazo\Box\EEG-Hunger_pilot_data_collection\PACT data\2022-03-07 PACT Pilot Data\000E - Purple Right.bin';
FileName = 'C:\Users\esazo\Box\EEG-Hunger_pilot_data_collection\PACT data\2022-03-07 PACT Pilot Data\002E - Black Right.bin';
FileName = 'C:\Users\esazo\Box\EEG-Hunger_pilot_data_collection\PACT data\2022-03-09 PACT Pilot Data\000E-trial 1.bin';
FileName = 'C:\Users\esazo\Box\EEG-Hunger_pilot_data_collection\PACT data\2022-03-09 PACT Pilot Data\001E- trial 2.bin';
FileName = 'C:\Users\esazonov\Box\EEG-Hunger_pilot_data_collection\PACT data\2022-03-29 PACT Pilot Data\000E.bin';
FileName = '/Users/prashanthjonna/Projects/EEG-RSA UA Project 2022/Matlab Scripts/000B.bin';

fileID = fopen(FileName);
data = uint8(fread(fileID));
s = whos('data');
Nbytes = s.bytes;
SDCard_ByteSize = 8192;
NWrites_SDCard = Nbytes/SDCard_ByteSize;

NSamples_ecg = 1022;

fprintf('File contains %d bytes, %d writes to SD card\n\n', Nbytes, NWrites_SDCard);
fprintf('Total samples ecg: %d\n', NSamples_ecg*NWrites_SDCard);

t_data = NSamples_ecg*NWrites_SDCard*(1/fs_ecg);
hours = floor(t_data / 3600);
t_data = t_data - hours * 3600;
mins = floor(t_data / 60);
secs = t_data - mins * 60;
fprintf('Total time recorded ecg: %d:%d:%0.2f\n\n',hours, mins, secs);

%Header
PACT_GPS = zeros(1,19);
PACT_Date = zeros(NWrites_SDCard,4);
PACT_Time = zeros(NWrites_SDCard,4);

%ECG
PACT_ECG = zeros(NSamples_ecg*NWrites_SDCard,2);

for write=1:NWrites_SDCard
    fprintf('Memory Block %d -> %d to %d, ',write, SDCard_ByteSize*(write-1)+1, SDCard_ByteSize*write);

    x = data(SDCard_ByteSize*(write-1)+1:SDCard_ByteSize*write);
    s = whos('x');
    Nbytes = s.bytes;
    fprintf('%d bytes\n', Nbytes);

    %Subsec data
    s0 = 1;
    sf = s0 + 1-1;
%     [s0 sf]
    SubsecBytes = x(s0:sf)';
    ms = 1000-(double(SubsecBytes)*1000/256);
    
    %Blank data
    s0 = sf + 1;
    sf = s0 + 7-1;
%     [s0 sf]
    BlankBytes = x(s0:sf)';
    
    
    %Time data
    s0 = sf + 1;
    sf = s0 + 4-1;
%     [s0 sf]
    TimeBytes = x(s0:sf)';
    Time = typecast(uint8(TimeBytes), 'uint32');
    mask = bitor(uint32(3145728),uint32(983040));
    hour = double(bitsra(bitand(Time,mask),16));
    mask = bitor(uint32(28672),uint32(3840));
    min = double(bitsra(bitand(Time,mask),8));
    mask = bitor(uint32(112),uint32(15));
    sec = double(bitand(Time,mask));
    PACT_Time(write,:) = [hour, min, sec, ms];
    
    
    %Date data
    s0 = sf + 1;
    sf = s0 + 4-1;
%     [s0 sf]
    DateBytes = x(s0:sf)';
    Date = typecast(uint8(DateBytes), 'uint32');
    mask = bitor(uint32(15728640),uint32(983040));
    year = double(bitsra(bitand(Date,mask),16));
    mask = bitor(uint32(4096),uint32(3840));
    month = double(bitsra(bitand(Date,mask),8));
    mask = bitor(uint32(48),uint32(15));
    day = double(bitand(Date,mask));
    week_day = double(bitsra(bitand(Date,uint32(57344)),13));
    PACT_Date(write,:) = [year, month, day, week_day];
    
    
    %ECG Channel1 data
    s0 = sf+1;
    sf = s0 + NSamples_ecg*4-1;
%     [s0 sf]
    ECGCh1Bytes = x(s0:sf)';
    [ECGCh1Bytes,padded] = vec2mat(ECGCh1Bytes,4);
    for k=1:size(ECGCh1Bytes,1)
        PACT_ECG(k+(write-1)*NSamples_ecg,1) = double(typecast(uint8(ECGCh1Bytes(k,:)), 'int32'));
    end
  
    %ECG Channel2 data
    s0 = sf+1;
    sf = s0 + NSamples_ecg*4-1;
%     [s0 sf]
    ECGCh2Bytes = x(s0:sf)';
    [ECGCh2Bytes,padded] = vec2mat(ECGCh2Bytes,4);
    for k=1:size(ECGCh2Bytes,1)
        PACT_ECG(k+(write-1)*NSamples_ecg,2) = double(typecast(uint8(ECGCh2Bytes(k,:)), 'int32'));
    end
    
end
%Conversions
PACT_ECG(:,1) = PACT_ECG(:,1)*0.0000023848*4; %mV
PACT_ECG(:,2) = PACT_ECG(:,2)*0.000023848*0.5; %mV

%Convert hour, miute, sec to readable values
for i=1:3
    x = PACT_Time(:,i);
    value = uint8(x);
    tmp = bitsra(bitand(value,uint8(240)),uint8(4))*10;
    R = tmp + bitand(value,uint8(15));
    PACT_Time(:,i) = double(R);
end

%Convert year, month, and day to readable values
for i=1:3
    x = PACT_Date(:,i);
    value = uint8(x);
    tmp = bitsra(bitand(value,uint8(240)),uint8(4))*10;
    R = tmp + bitand(value,uint8(15));
    PACT_Date(:,i) = double(R);

end


fclose(fileID);
index = find(FileName=='.');
file=strrep(FileName,'bin','mat')
save(file,'PACT_Date','PACT_Time','PACT_ECG');

%%
% PACT_ECG = PACT_ECG(1:300*fs_ecg,:);

%% Plotting ECG
fs_ecg = 1000;
fig = 0;

x = PACT_ECG(:,2);
x = x - mean(x);
Y = fft(x);
L = numel(x);
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = fs_ecg*(0:(L/2))/L;

% fig = fig + 1;
% figure (fig);
% plot(f,P1);
% title('Single-Sided Amplitude Spectrum of X(t)')
% xlabel('f (Hz)');
% ylabel('|P1(f)|');


%Notch Filter for ECG
%251.bin good example of notch filter
fnyquist = 500; %Hz
fremove = 60; %Hz
fnorm = fremove/fnyquist;
fnotch = [fnorm-0.03, fnorm+0.03];
b = fir1(100,fnotch,'stop');
x2 = filter(b,1,x);

% fig = fig + 1;
% figure (fig);
% subplot(2,1,1),plot(x);
% subplot(2,1,2),plot(x2); title('After Filtering');
% xlabel('Time(sec)');

x2 = x2 - mean(x2);
Y = fft(x2);
L = numel(x2);
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = fs_ecg*(0:(L/2))/L;

% fig = fig + 1;
% figure (fig);
% plot(f,P1);
% title('Single-Sided Amplitude Spectrum of filtered X(t)')
% xlabel('f (Hz)');
% ylabel('|P1(f)|');

PACT_ECG(:,2) = x2;


PACT_ECG(1,:) = PACT_ECG(2,:);
x = PACT_ECG(:,1);

fig = fig + 1;
figure(fig);
ts = (0:numel(x)-1)/fs_ecg;
plot(ts,x);
ylabel('mV');
t = numel(x)/fs_ecg;
hours = floor(t / 3600);
t = t - hours * 3600;
mins = floor(t / 60);
secs = t - mins * 60;
% title(sprintf('%d hours, %d min, %0.2f sec (Total samples: %d)\nChannel 1', hours, mins, secs, numel(x)));
title('PACT Respiration Signal');
axis tight
xlabel('Time(sec)');


x = PACT_ECG(:,2);

ts = (0:numel(x)-1)/fs_ecg;
fig = fig + 1;
figure(fig);
plot(ts,x);
xlabel('Time(sec)');
ylabel('mV');
% title('Channel 2');
title('PACT ECG Signal');
axis tight

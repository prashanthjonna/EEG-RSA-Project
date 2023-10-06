clear all; clc;
%%
fs = 100;
fs_ecg = 1000;
%%
Folder = 'Extracted data';
folder = fullfile(pwd, Folder);%Save results
if(~isdir(folder))
    mkdir(folder);
end

FileName = 'C:\Users\esazo\Box\EEG-Hunger_pilot_data_collection\PACT data\2022-02-28 Tests\PACT - 1\000B.bin';
FileName = 'C:\Users\esazo\Box\EEG-Hunger_pilot_data_collection\PACT data\2022-02-28 Tests\PACT - 2\001B.bin';
FileName = 'C:\Users\esazo\Box\EEG-Hunger_pilot_data_collection\PACT data\2022-02-28 Tests\PACT - 4\001B.bin';
FileName = 'C:\Users\esazo\Box\EEG-Hunger_pilot_data_collection\PACT data\2022-03-07 PACT Pilot Data\000B - Purple Right.bin';
FileName = 'C:\Users\esazo\Box\EEG-Hunger_pilot_data_collection\PACT data\2022-03-07 PACT Pilot Data\002B - Black Right.bin';
FileName = 'C:\Users\esazo\Box\EEG-Hunger_pilot_data_collection\PACT data\2022-03-09 PACT Pilot Data\000B-trial 1.bin';
FileName = 'C:\Users\esazo\Box\EEG-Hunger_pilot_data_collection\PACT data\2022-03-09 PACT Pilot Data\001B- trial 2.bin';
FileName = '/Users/prashanthjonna/Projects/EEG-RSA UA Project 2022/Matlab Scripts/000B.bin';

fileID = fopen(FileName);
data = uint8(fread(fileID));
s = whos('data');
Nbytes = s.bytes;
SDCard_ByteSize = 8192;
NWrites_SDCard = Nbytes/SDCard_ByteSize;


NSamples_data = 680;

fprintf('File contains %d bytes, %d writes to SD card\n\n', Nbytes, NWrites_SDCard);
fprintf('Total samples data: %d\n', NSamples_data*NWrites_SDCard);

t_data = NSamples_data*NWrites_SDCard*(1/fs);
hours = floor(t_data / 3600);
t_data = t_data - hours * 3600;
mins = floor(t_data / 60);
secs = t_data - mins * 60;
fprintf('Total time recorded data: %d:%d:%0.2f\n',hours, mins, secs);

%Header
PACT_GPS_B = cell(NWrites_SDCard,9);
PACT_Date_B = zeros(NWrites_SDCard,4);
PACT_Time_B = zeros(NWrites_SDCard,4);

%Data
PACT_Data_B = zeros(NSamples_data*NWrites_SDCard,5);

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
        sf = s0 + 12-1;
        BlankBytes = x(s0:sf)';

        %GPS Fix data
        s0 = sf + 1;
        sf = s0 + 1-1;
        GPSFixBytes = x(s0:sf)';
%          PACT_GPS{write,1} = GPSFixBytes;
     
        %GPS Lat Degree data
        s0 = sf + 1;
        sf = s0 + 1-1;
        GPSLatDegreeBytes = x(s0:sf)';
         PACT_GPS_B{write,1} = GPSLatDegreeBytes;
        
        %GPS Lat Minute data
        s0 = sf + 1;
        sf = s0 + 1-1;
        GPSLatMinBytes = x(s0:sf)';
         PACT_GPS_B{write,2} = GPSLatMinBytes;
        
        %GPS Lat Co NS data
        s0 = sf + 1;
        sf = s0 + 1-1;
        GPSLatCoNSBytes = x(s0:sf)';
        
         PACT_GPS_B{write,4} = native2unicode(GPSLatCoNSBytes);

        %GPS Long Degree data
        s0 = sf + 1;
        sf = s0 + 1-1;
        GPSLongDegreeBytes = x(s0:sf)';
         PACT_GPS_B{write,5} = GPSLongDegreeBytes;
        
        %GPS Long Minute data
        s0 = sf + 1;
        sf = s0 + 1-1;
        GPSLongMinBytes = x(s0:sf)';
         PACT_GPS_B{write,6} = GPSLongMinBytes;
        
        %GPS Long Co EW data
        s0 = sf + 1;
        sf = s0 + 1-1;
        GPSLongCoEWBytes = x(s0:sf)';
        
         PACT_GPS_B{write,8} = native2unicode(GPSLongCoEWBytes);
        
        %GPS Lat Minute Frac data
        s0 = sf + 1;
        sf = s0 + 2-1;
        GPSLatMinFracBytes = x(s0:sf)';
        GPSLatMinFrac = typecast(uint8(GPSLatMinFracBytes), 'uint16');
         PACT_GPS_B{write,3} = GPSLatMinFrac;
        
        %GPS Long Minute Frac data
        s0 = sf + 1;
        sf = s0 + 2-1;
        GPSLongMinFracBytes = x(s0:sf)';
        GPSLongMinFrac = typecast(uint8(GPSLongMinFracBytes), 'uint16');
         PACT_GPS_B{write,7} = GPSLongMinFrac;
    
    
    %Hand data
    s0 = sf+1;
    sf = s0 + NSamples_data*2-1;
%     [s0 sf]
    HandBytes = x(s0:sf)';
    [HandBytes,padded] = vec2mat(HandBytes,2);
    for k=1:size(HandBytes,1)
        PACT_Data_B(k+(write-1)*NSamples_data,2) = double(typecast(uint8(HandBytes(k,:)), 'uint16'));
    end   
    
    %AccX data
    s0 = sf+1;
    sf = s0 + NSamples_data*2-1;
%     [s0 sf]
    AccXBytes = x(s0:sf)';
    [AccXBytes,padded] = vec2mat(AccXBytes,2);
    for k=1:size(AccXBytes,1)
        PACT_Data_B(k+(write-1)*NSamples_data,3) = double(typecast(uint8(AccXBytes(k,:)), 'int16'));
    end
    
    %AccY data
    s0 = sf+1;
    sf = s0 + NSamples_data*2-1;
%     [s0 sf]
    AccYBytes = x(s0:sf)';
    [AccYBytes,padded] = vec2mat(AccYBytes,2);
    for k=1:size(AccYBytes,1)
        PACT_Data_B(k+(write-1)*NSamples_data,4) = double(typecast(uint8(AccYBytes(k,:)), 'int16'));
    end
    
    %AccZ data
    s0 = sf+1;
    sf = s0 + NSamples_data*2-1;
%     [s0 sf]
    AccZBytes = x(s0:sf)';
    [AccZBytes,padded] = vec2mat(AccZBytes,2);
    for k=1:size(AccZBytes,1)
        PACT_Data_B(k+(write-1)*NSamples_data,5) = double(typecast(uint8(AccZBytes(k,:)), 'int16'));
    end

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
    PACT_Time_B(write,:) = [hour, min, sec, ms];
    
    
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
    PACT_Date_B(write,:) = [year, month, day, week_day];

    
    %Breathing data
    s0 = sf+1;
    sf = s0 + NSamples_data*4-1;
%     [s0 sf]
    BreathingBytes = x(s0:sf)';
    [BreathingBytes,padded] = vec2mat(BreathingBytes,4);
    for k=1:size(BreathingBytes,1)
        PACT_Data_B(k+(write-1)*NSamples_data,1) = double(typecast(uint8(BreathingBytes(k,:)), 'int32'));
    end
   
end
%Conversions
PACT_Data_B(:,2) = PACT_Data_B(:,2)*0.000683593; %mV

%Convert hour, miute, sec to readable values
for i=1:3
    x = PACT_Time_B(:,i);
    value = uint8(x);
    tmp = bitsra(bitand(value,uint8(240)),uint8(4))*10;
    R = tmp + bitand(value,uint8(15));
    PACT_Time_B(:,i) = double(R);
end

%Convert year, month, and day to readable values
for i=1:3
    x = PACT_Date_B(:,i);
    value = uint8(x);
    tmp = bitsra(bitand(value,uint8(240)),uint8(4))*10;
    R = tmp + bitand(value,uint8(15));
    PACT_Date_B(:,i) = double(R);

end

fclose(fileID);
index = find(FileName=='.');
file=strrep(FileName,'bin','mat')
save(file,'PACT_GPS_B','PACT_Date_B','PACT_Time_B','PACT_Data_B');

%%
% PACT_Data = PACT_Data(1:300*fs,:);
% PACT_ECG = PACT_ECG(1:300*fs_ecg,:);

%% Plotting Data
PACT_Data_B(1,:) = PACT_Data_B(3,:);
PACT_Data_B(2,:) = PACT_Data_B(3,:);
x = -1*PACT_Data_B(:,1); %breathing
ts = (0:numel(x)-1)/fs;

fig = 1;
figure(fig);
plot(ts,x);
xlabel('Time(sec)');
ylabel('Pulse Count (inverted)');
t = numel(x)/fs;
hours = floor(t / 3600);
t = t - hours * 3600;
mins = floor(t / 60);
secs = t - mins * 60;
% title(sprintf('%d hours, %d min, %0.2f sec (Total samples: %d)', hours, mins, secs, numel(x)));
title('PACT Breathing Signal');
axis tight

%% Plotting Hand
fig = fig + 1;
figure(fig);
 x = PACT_Data_B(:,2); %breathing
%     x = PACT_Data(2000:5000,2); %breathing
ts = (0:numel(x)-1)/fs;

plot(ts,x); 
title('PACT Proximity Signal');
xlabel('Time(sec)');
ylabel('V');
axis([ts(1),ts(end),0,3]);

%     fprintf('Avg. Hand: %0.4f+/-%0.4f V\n',mean(x),sqrt(var((x))));
%  //   fprintf('Avg. Acc Y: %0.4f+/-%0.4f g\n',mean(y),sqrt(var((y))));
%  //   fprintf('Avg. Acc Z: %0.4f+/-%0.4f g\n',mean(z),sqrt(var((z))));

%% Plotting ACC

x = PACT_Data_B(:,3); %breathing
y = PACT_Data_B(:,4); %breathing
z = PACT_Data_B(:,5); %breathing
ts = (0:numel(x)-1)/fs;

%Convert to gravitational (g) units from ADC units
gRange = 4;
adcBits = 12;
factor = gRange/(2^adcBits);
x = x*factor;
y = y*factor;
z = z*factor;

figure(10);
plot(ts,x,'k'); 
hold on;
plot(ts,y,'r'); 
plot(ts,z,'b'); 
xlabel('Time(sec)');
axis tight;
ylabel('Acceleration (g)');
legend('X','Y','Z');
title('PACT Accelerometer Signals');
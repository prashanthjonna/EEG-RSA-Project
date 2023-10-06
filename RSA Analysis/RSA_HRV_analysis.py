import neurokit2 as nk
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import csv, time, scipy
import numpy as np
from scipy.fft import fft, fftfreq
import pandas as pd
from scipy.signal import savgol_filter
import statistics
import json, math
from collections import defaultdict
import pprint

# Defining all known constants

rsp_sampling_rate = 100
ecg_sampling_rate = 1000                              # The ecg, rsp data were approximately sampled at 1000Hz and 100Hz respectively
rsp_sampling_period = 1/rsp_sampling_rate             # Sampling period / Sampling interval = 1 / Sampling frequency
ecg_sampling_period = 1/ecg_sampling_rate
scaling_factor = ecg_sampling_rate/rsp_sampling_rate

# This function finds the closest number in a list to a given number and returns it

def get_closest_greater_number(my_list,my_number):
    return min([ i for i in my_list if i >= my_number], key=lambda x:abs(x-my_number))

# This function performs downsampling by a factor of "factor" and returns the resultant downsampled data

def downsampling(samples_list,factor):

    result_list = []
    counter = 0

    for i in range(len(samples_list)):
        if(i==counter):
            result_list.append(samples_list[i])
            counter += factor
    
    return result_list

def upsampling(samples_list,factor):
    
    result = []

    for i in range(len(samples_list)-1):
        
        diff = (samples_list[i+1] - samples_list[i]) / factor

        for j in range(factor):

            result += [samples_list[i] + j*diff]

    result+=[samples_list[-1]]

    return result 

# STARTS HERE ------------------------------------------------------------------------------------------

def calculate_trial_RSA_HRV(trial_number):

    mat_E = scipy.io.loadmat('PACT_data/'+ trial_number +'E.mat')
    mat_B = scipy.io.loadmat('PACT_data/'+ trial_number +'B.mat')

    # Extracting the relevant from the data (ECG signal and RSP signal) and removing first and last 10 seconds data samples

    ecg_PACT = mat_E["PACT_ECG"]
    ecg_data = [i[1] for i in ecg_PACT]
    ecg_data = ecg_data[10000:len(ecg_data)-10000]

    rsp_PACT = mat_B["PACT_Data_B"]
    rsp_data = [i[0] for i in rsp_PACT]
    rsp_data = rsp_data[1000:len(rsp_data)-1000]
    rsp_data = upsampling(rsp_data,10)

    ecg_data = ecg_data[0:len(rsp_data)]
    samples_index = list(range(0,len(ecg_data)))

    # Plot ECG signal

    # plt.plot(samples_index,ecg_data)
    # plt.xlabel("Samples")
    # plt.ylabel("ECG Magnitude")
    # plt.title("ECG Signal Plot")
    # plt.show()


    # Plot RSP signal

    # samples_index = list(range(0,len(rsp_data)))
    # plt.plot(samples_index,rsp_data)
    # plt.xlabel("Samples")
    # plt.ylabel("RSP Magnitude")
    # plt.title("RSP Signal Plot")
    # plt.show()


    # R-peak detection in ECG signal

    signals, info = nk.ecg_process(ecg_data, sampling_rate=1000)
    rpeaks = info["ECG_R_Peaks"]
    ecg_data_filtered = signals["ECG_Clean"]

    # print("R-Peaks : ",rpeaks)

    # Plot Filtered ECG with R-Peak Detection

    # plt.plot(ecg_data_filtered)
    # plt.plot(rpeaks, ecg_data_filtered[rpeaks], "o")
    # plt.xlabel("Samples")
    # plt.ylabel("ECG Magnitude")
    # plt.title("Filtered ECG Signal R-Peak Detection")

    # for xc in rpeaks:
    #     plt.axvline(x=xc, ls=":")

    # plt.show()


    rsp_data_filtered = nk.rsp_clean(rsp_data)
    rsp_data_filtered = savgol_filter(rsp_data_filtered, 700, 3)

    # Plot Filtered RSP signal

    # plt.plot(samples_index,rsp_data_filtered)
    # plt.xlabel("Samples")
    # plt.ylabel("RSP Magnitude")
    # plt.title("Filtered RSP Signal Plot")
    # plt.show()


    rsp_peaks, _ = find_peaks(rsp_data_filtered, distance=2000, width=600)
    rsp_valleys, _ = find_peaks(-1*rsp_data_filtered, distance=2000, width=600)


    # Peak and Valley Detection in Filtered RSP signal

    # plt.xlabel("Samples")
    # plt.ylabel("RSP Magnitude")
    # plt.title("Filtered RSP Signal - Peak and Valley Detection")
    # plt.plot(rsp_data_filtered)
    # plt.plot(rsp_peaks, rsp_data_filtered[rsp_peaks], "o")
    # plt.plot(rsp_valleys, rsp_data_filtered[rsp_valleys], "o")

    # for xc in rsp_peaks:
    #     plt.axvline(x=xc, ls=":")

    # for xc in rsp_valleys:
    #     plt.axvline(x=xc, ls=":")

    # plt.show()

    # Plot both ECG and RSP with R-Peak and Peak-Valley Detection

    # plt.plot(ecg_data_filtered)
    # plt.plot(rpeaks, ecg_data_filtered[rpeaks], "o")
    # plt.plot(0.001*rsp_data_filtered)
    # plt.plot(rsp_peaks, 0.001*rsp_data_filtered[rsp_peaks], "o")
    # plt.plot(rsp_valleys, 0.001*rsp_data_filtered[rsp_valleys], "o")

    # for xc in rsp_peaks:
    #     plt.axvline(x=xc, ls="-")

    # for xc in rsp_valleys:
    #     plt.axvline(x=xc, ls="-")

    # for xc in rpeaks:
    #     plt.axvline(x=xc, ls=":", color='r')

    # plt.xlabel("Samples")
    # plt.ylabel("Signal Magnitude")
    # plt.title("ECG & RSP Signals")

    # plt.show()


    Inhalation_interval = []
    Exhalation_interval = []

    for i in range(len(rsp_valleys)-1): 
        
        next_peak = get_closest_greater_number(rsp_peaks,rsp_valleys[i])
        next_next_valley = get_closest_greater_number(rsp_valleys,next_peak)
        Inhalation_interval.append([rsp_valleys[i],next_peak])
        Exhalation_interval.append([next_peak,next_next_valley])

    # print('Inhalation intervals : ', Inhalation_interval)
    # print('Exhalation intervals : ', Exhalation_interval)

    Inhalation_index_range = []
    Exhalation_index_range = []

    for i in Inhalation_interval:
        Inhalation_index_range += list(range(i[0],i[1]))

    for i in Exhalation_interval:
        Exhalation_index_range += list(range(i[0],i[1]))


    # Plot the Inhalation and Exhalation Intervals

    # plt.xlabel("Samples")
    # plt.ylabel("RSP Magnitude")
    # plt.title("RSP Signal - Inhalation & Exhalation Intervals")
    # plt.plot(rsp_data_filtered)
    # plt.plot(Inhalation_index_range, rsp_data_filtered[Inhalation_index_range], "x", color = 'b')
    # plt.plot(Exhalation_index_range, rsp_data_filtered[Exhalation_index_range], "x", color = 'g')
    # plt.show()

    R_peak_inhale = []
    R_peak_exhale = []

    for i in Inhalation_interval:
        temp = []
        for j in rpeaks:
            if(j > i[0] and j < i[1]):
                temp += [j]
        R_peak_inhale.append(temp)

    for i in Exhalation_interval:
        temp = []
        for j in rpeaks:
            if(j > i[0] and j < i[1]):
                temp += [j]
        R_peak_exhale.append(temp)

    # print("R_peak_inhale : ", R_peak_inhale)
    # print("R_peak_exhale : ", R_peak_exhale)

    RR_diff_inhale = []
    RR_diff_exhale = []

    for i in R_peak_inhale:
        RR_diff_inhale.append([i[j + 1] - i[j] for j in range(len(i)-1)])

    for i in R_peak_exhale:
        RR_diff_exhale.append([i[j + 1] - i[j] for j in range(len(i)-1)])

    # print("RR_diff_inhale : ", RR_diff_inhale)
    # print("RR_diff_exhale : ", RR_diff_exhale)


    EI_div_list = []

    for i in range(len(RR_diff_inhale)):
        EI_div_list.append(max(RR_diff_exhale[i])/min(RR_diff_inhale[i]))

    EI_div_mean = np.average(EI_div_list)
    EI_div_std = np.std(EI_div_list)

    print("E/I Result : " + str(EI_div_mean) + " +/- " + str(EI_div_std))

    HR_inhale = []
    HR_exhale = []
    EI_diff_median = []
    EI_diff_mean = []

    for i in RR_diff_inhale:
        temp = []
        for j in i:
            temp.append(60000/j)
        HR_inhale.append(temp)

    for i in RR_diff_exhale:
        temp = []
        for j in i:
            temp.append(60000/j)
        HR_exhale.append(temp)

    #print(HR_inhale)
    #print(HR_exhale)

    for i in range(len(HR_inhale)):
        EI_diff_mean.append(np.average(HR_exhale[i])-np.average(HR_inhale[i]))

    EI_diff_mean_avg = np.average(EI_diff_mean)
    EI_diff_mean_std = np.std(EI_diff_mean)

    print("E-I Mean Result : " + str(abs(EI_diff_mean_avg)) + " +/- " + str(EI_diff_mean_std))

    for i in range(len(HR_inhale)):
        EI_diff_median.append(statistics.median(HR_exhale[i])-statistics.median(HR_inhale[i]))

    EI_diff_median_avg = np.average(EI_diff_median)
    EI_diff_median_std = np.std(EI_diff_median)

    print("E-I Median Result : " + str(abs(EI_diff_median_avg)) + " +/- " + str(EI_diff_median_std))

    EI_results = [EI_div_mean,EI_div_std,EI_diff_mean_avg,EI_diff_mean_std,EI_diff_median_avg,EI_diff_median_std]

    clean_signals, info = nk.bio_process(ecg_data_filtered, rsp_data_filtered, sampling_rate=1000)
    print("\nTraditional RSA measures : \n")

    RSA_values = nk.hrv_rsa(clean_signals, sampling_rate=1000)
    print(json.dumps(RSA_values,indent=4))

    print("\nHRV Measures :\n")
    hrv = nk.hrv_time(rpeaks, sampling_rate=ecg_sampling_rate, show=True)
    print(hrv.iloc[0],"\n")

    return EI_results, RSA_values, hrv


pre_list = []
post_list = []

with open('experiment_data.txt','r') as file:
   
    # reading each line    
    for line in file:
   
        if(line[4:7]=='pre'):
            pre_list.append(line[0:3])

        if(line[4:8]=='post'):
            post_list.append(line[0:3])

EI_list_pre = []
EI_list_post = []
RSA_list_pre = []
RSA_list_post = []
HRV_list_pre = []
HRV_list_post = []

print("Individual RSA Analysis of Pre-Meal Trials : ")

for i in pre_list:

    print("\n\n-------------------------------------\n")
    print("RSA Analysis of Trial Number : ",i)
    print("\n-------------------------------------\n\n")
    EI_list, RSA, HRV = calculate_trial_RSA_HRV(i)
    EI_list_pre.append(EI_list)
    RSA_list_pre.append(RSA)
    HRV_list_pre.append(HRV)


print("\n\nIndividual RSA Analysis of Post-Meal Trials : ")

for i in post_list:

    print("\n\n-------------------------------------\n")
    print("RSA Analysis of Trial Number : ",i)
    print("\n-------------------------------------\n\n")
    EI_list, RSA, HRV = calculate_trial_RSA_HRV(i)
    EI_list_post.append(EI_list)
    RSA_list_post.append(RSA)
    HRV_list_post.append(HRV)


print("\n\nCombined RSA Analysis of Pre-Meal Trials : \n")
print("\n-------------------------------------\n")

EI_div_mean = 0
EI_div_std = 0
EI_diff_mean_avg = 0
EI_diff_mean_std = 0
EI_diff_median_avg = 0
EI_diff_median_std = 0

for i in EI_list_pre:

    EI_div_mean += i[0]
    EI_diff_mean_avg += i[2]
    EI_diff_median_avg += i[4]

EI_div_mean /= len(EI_list_pre) 
EI_diff_mean_avg /= len(EI_list_pre) 
EI_diff_median_avg /= len(EI_list_pre) 

for i in EI_list_pre:

    EI_div_std += i[1]*i[1] + (EI_div_mean - i[0])*(EI_div_mean - i[0])
    EI_diff_mean_std += i[3]*i[3] + (EI_diff_mean_avg - i[2])*(EI_diff_mean_avg - i[2])
    EI_diff_median_std += i[5]*i[5] + (EI_diff_median_avg - i[4])*(EI_diff_median_avg - i[4])

EI_div_std = math.sqrt(EI_div_std/len(EI_list_pre)) 
EI_diff_mean_std = math.sqrt(EI_diff_mean_std/len(EI_list_pre))
EI_diff_median_std = math.sqrt(EI_diff_median_std/len(EI_list_pre))

print("E/I Result : " + str(EI_div_mean) + " +/- " + str(EI_div_std))
print("E-I Mean Result : " + str(abs(EI_diff_mean_avg)) + " +/- " + str(EI_diff_mean_std))
print("E-I Median Result : " + str(abs(EI_diff_median_avg)) + " +/- " + str(EI_diff_median_std))

print("\nTraditional RSA measures : \n")

dd = defaultdict(list)

for d in RSA_list_pre: # you can list as many input dicts as you want here
    for key, value in d.items():
        dd[key].append(value)

for key, value in dd.items():
        dd[key] = [np.average(dd[key]),np.std(dd[key])]

# print(json.dumps(dd,indent=4))
pprint.pprint(dict(dd))

print("\n\nHRV Results : \n")
HRV_result = pd.concat(HRV_list_pre)
print("\nMean of HRV Values\n")
print(HRV_result.mean())
print("\nStandard Deviation of HRV Values\n")
print(HRV_result.std())


print("\n\nCombined RSA Analysis of Post-Meal Trials : \n")
print("\n-------------------------------------\n")

EI_div_mean = 0
EI_div_std = 0
EI_diff_mean_avg = 0
EI_diff_mean_std = 0
EI_diff_median_avg = 0
EI_diff_median_std = 0

for i in EI_list_post:

    EI_div_mean += i[0]
    EI_diff_mean_avg += i[2]
    EI_diff_median_avg += i[4]

EI_div_mean /= len(EI_list_post) 
EI_diff_mean_avg /= len(EI_list_post) 
EI_diff_median_avg /= len(EI_list_post) 

for i in EI_list_post:

    EI_div_std += i[1]*i[1] + (EI_div_mean - i[0])*(EI_div_mean - i[0])
    EI_diff_mean_std += i[3]*i[3] + (EI_diff_mean_avg - i[2])*(EI_diff_mean_avg - i[2])
    EI_diff_median_std += i[5]*i[5] + (EI_diff_median_avg - i[4])*(EI_diff_median_avg - i[4])

EI_div_std = math.sqrt(EI_div_std/len(EI_list_post)) 
EI_diff_mean_std = math.sqrt(EI_diff_mean_std/len(EI_list_post))
EI_diff_median_std = math.sqrt(EI_diff_median_std/len(EI_list_post))

print("E/I Result : " + str(EI_div_mean) + " +/- " + str(EI_div_std))
print("E-I Mean Result : " + str(abs(EI_diff_mean_avg)) + " +/- " + str(EI_diff_mean_std))
print("E-I Median Result : " + str(abs(EI_diff_median_avg)) + " +/- " + str(EI_diff_median_std))

print("\nTraditional RSA measures : \n")

dd = defaultdict(list)

for d in RSA_list_post: # you can list as many input dicts as you want here
    for key, value in d.items():
        dd[key].append(value)

for key, value in dd.items():
        dd[key] = [np.average(dd[key]),np.std(dd[key])]

# print(json.dumps(dd,indent=4))
pprint.pprint(dict(dd))

print("\n\nHRV Results : \n")
HRV_result = pd.concat(HRV_list_post)
print("\nMean of HRV Values : \n")
print(HRV_result.mean())
print("\nStandard Deviation of HRV Values :\n")
print(HRV_result.std())

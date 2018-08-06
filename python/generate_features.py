import pandas as pd
import math
import numpy as np
import os
import peakutils
from numpy import linalg as LA
from scipy.fftpack import fft, ifft
import itertools as it
from heapq import nlargest

file = "/home/users/bhargavy/gait/data/demographics.csv"
csv = pd.read_csv(file, low_memory=False)

# Version numbers are meaningless
sixminutev4 = pd.read_csv('/home/users/bhargavy/gait/data/6minutewalkV4-V2.csv') # table 
sixminutev6 = pd.read_csv('/home/users/bhargavy/gait/data/6minutewalkV4-V6.csv')
sixminutev1 = pd.read_csv('/home/users/bhargavy/gait/data/6minutewalkV4-V1.csv') # table
sixminutev2 = pd.read_csv('/home/users/bhargavy/gait/data/6minutewalkV2.csv')
cvhealth = pd.read_csv('/home/users/bhargavy/gait/data/CVHealth.csv')

csv.drop(columns=['ROW_ID', 'ROW_VERSION', 'appVersion', 'phoneInfo', 'dataGroups', 'uploadDate', 'createdOn', 
                    'validationErrors', 'userSharingScope', 'NonIdentifiableDemographics.json.patientGoSleepTime', 
                    'NonIdentifiableDemographics.patientGoSleepTime', 'createdOnTimeZone',
                    'NonIdentifiableDemographics.json.patientWakeUpTime', 'NonIdentifiableDemographics.patientWakeUpTime'
                    , 'externalId'], 
            inplace=True)
csv.rename(columns={'NonIdentifiableDemographics.json.patientWeightPounds': 'weight', 'NonIdentifiableDemographics.patientWeightPounds': 'weight2', 
                        'NonIdentifiableDemographics.json.patientBiologicalSex': 'sex', 
                        'NonIdentifiableDemographics.patientBiologicalSex': 'sex2',
                    'NonIdentifiableDemographics.json.patientHeightInches': 'height',
                    'NonIdentifiableDemographics.patientHeightInches': 'height2',
                    'NonIdentifiableDemographics.json.patientCurrentAge': 'currentAge',
                    'NonIdentifiableDemographics.patientCurrentAge': 'currentAge2'}, inplace=True)

# drop the test version
csv = csv.iloc[7:]
csv.index = range(44804)
csv.dropna(how='all', subset=['currentAge','currentAge2'], inplace=True)
csv.dropna(how='all', subset=['weight', 'weight2'], inplace=True)
csv.dropna(how='all', subset=['sex', 'sex2'], inplace=True)
csv.dropna(how='all', subset=['height', 'height2'], inplace=True)

csv['currentAge'] = csv['currentAge'].fillna(csv['currentAge2'])
csv['weight'] = csv['weight'].fillna(csv['weight2'])
csv['height'] = csv['height'].fillna(csv['height2'])
csv['sex'] = csv['sex'].fillna(csv['sex2'])
csv = csv.drop('currentAge2',1)
csv = csv.drop('weight2',1)
csv = csv.drop('height2',1)
csv = csv.drop('sex2',1)

# Dropping the duplicate healthCode records... Don't know if this is the right thing to do but can easily be reversed
csv = csv.drop_duplicates(subset='healthCode', keep='last', inplace=False)

# Dropping the rows with 0 for any metric. This was not caught with the NaN cleaning/merging 
csv = csv[csv.currentAge != 0]
csv = csv[csv.weight != 0]
csv = csv[csv.height != 0]

sixminutev4.drop(columns=['phoneInfo', 'appVersion', 'dataGroups', 'externalId', 'phoneInfo', 'ROW_VERSION', 
                            'ROW_ID', 'deviceMotion_fitness.rest.items', 
                            'heartRate_fitness.rest.items', 'userSharingScope',
                            'pedometer_fitness.walk.items', 'accel_fitness_walk.json.items',
                            'deviceMotion_fitness.walk.items', 'heartRate_fitness.walk.items',
                            'accel_fitness_rest.json.items', 'createdOnTimeZone', 'recordId',
                            'uploadDate', 'validationErrors', 'createdOn', 'measurementSystem.measurementSystem',
                            'measurementSystem.deviceRegion'], inplace=True)
sixminutev6.drop(columns=['recordId', 'appVersion', 'externalId', 'dataGroups', 'createdOn', 'phoneInfo', 
                            'restingEnergyBurned_fitness.rest', 'restingEnergyBurned_fitness.walk', 'activeEnergyBurned_fitness.rest', 
                            'activeEnergyBurned_fitness.walk', 'measurementSystem.deviceRegion', 'measurementSystem.measurementSystem', 
                            'heartRate_fitness.rest', 'ROW_ID', 'ROW_VERSION', 'uploadDate',
                            'validationErrors', 'userSharingScope', 'accel_fitness_walk.json',
                            'deviceMotion_fitness.walk', 'userSharingScope','createdOnTimeZone',
                            'heartRate_fitness.walk', 'pedometer_fitness.walk', 'deviceMotion_fitness.rest',
                            'accel_fitness_rest.json'], inplace=True)

sixminutev1.drop(columns=['ROW_ID', 'ROW_VERSION', 'recordId', 'appVersion', 'uploadDate', 'phoneInfo',
                        'externalId', 'dataGroups', 'createdOn', 'createdOnTimeZone', 'userSharingScope',
                        'validationErrors', 'pedometer_fitness.walk.items', 'accel_fitness_walk.json.items',
                        'deviceMotion_fitness.walk.items', 'HKQuantityTypeIdentifierHeartRate_fitness.walk.items',
                        'accel_fitness_rest.json.items', 'deviceMotion_fitness.rest.items',
                        'HKQuantityTypeIdentifierHeartRate_fitness.rest.items', 'measurementSystem.measurementSystem',
                        'measurementSystem.deviceRegion'], inplace=True)

sixminutev2.drop(columns=['ROW_ID', 'ROW_VERSION', 'recordId', 'appVersion', 'uploadDate', 'phoneInfo', 'createdOn', 
                        'pedometer_fitness.walk.items', 'accel_fitness_walk.json.items', 'deviceMotion_fitness.walk.items', 
                        'HKQuantityTypeIdentifierHeartRate_fitness.walk.items', 'accel_fitness_rest.json.items',
                        'deviceMotion_fitness.rest.items', 'HKQuantityTypeIdentifierHeartRate_fitness.rest.items',
                        'externalId', 'dataGroups'], inplace=True)

sixminutewalktotal = pd.concat([sixminutev4, sixminutev6])
# Re-index to represent the actual length of the series
sixminutewalktotal.index = range(3373)
#csv
csv = csv[csv['healthCode'].isin(sixminutewalktotal['healthCode'])]
# Re-index 
csv.index = range(994)
cvhealth.drop(columns=['ROW_ID', 'ROW_VERSION', 'recordId', 'appVersion', 'phoneInfo', 'uploadDate', 
                            'externalId', 'dataGroups', 'createdOn', 
                            'createdOnTimeZone', 'userSharingScope', 'validationErrors', 
                            'family_history', 'medications_to_treat', 'vascular', 'ethnicity',
                            'race', 'education'], inplace=True)

csv = csv[csv['healthCode'].isin(cvhealth['healthCode'])]
# Re-index after dropping those without CVD data
csv.index = range(851)
healthCodeTotal = csv["healthCode"].values

v4 = csv[csv['healthCode'].isin(sixminutev4['healthCode'])]
v6 = csv[csv['healthCode'].isin(sixminutev6['healthCode'])]
v2 = csv[csv['healthCode'].isin(sixminutev2['healthCode'])]
v1 = csv[csv['healthCode'].isin(sixminutev1['healthCode'])]

v4 = np.asarray(v4['healthCode'].values)
v2 = np.asarray(v2['healthCode'].values)
v1 = np.asarray(v1['healthCode'].values)
v6 = np.asarray(v6['healthCode'].values)
sixminutetotalhc = np.concatenate((v4, v2, v1, v6))

# change this depending on data directory
directory_in_str = '/scratch/PI/euan/projects/mhc/data/6mwt/accel_walk_dir'
directory = os.fsencode(directory_in_str)

# number of gaps in a given 6MWT that surpasses the threshold 
def find_gaps(directory, table):
    '''
    Finds the number of gaps over 0.01 seconds and adds a gap if the difference between last and first timestamp is 
    less than 6 minutes
    Checking the length of the dict returned will show whether the amount of files matches the intersection of 
    6mwt table and demographics
    '''
    hc_gaps = dict()
    for subdir, dirs, files in os.walk(directory):
        # make in [list] a parameter of the function -- remember to change "in [list]" to the correct list
        if subdir.decode()[subdir.decode().rfind('/')+1:] in table:
            # Makes sure we only get one file per healthCode
            i = 0
            for file in files:
                while (i < 1):
                    a_df = pd.read_json(os.path.join(subdir.decode(), file.decode())).set_index('timestamp')
                    inst = list(a_df.index[1:]-a_df.index[:-1] > .02).count(True)
                    # This line checks for incomplete 6MWT
                    if (a_df.index[-1] - a_df.index[0]) < 358:
                        inst = inst + 1
                    hc_gaps.update({subdir.decode()[subdir.decode().rfind('/')+1:]: inst})
                    i += 1
                
    return hc_gaps


# Mean - 6 features
def get_time_mean(x_, y_, z_):
    mux_t = np.mean(x_, axis=1)
    muy_t = np.mean(y_, axis=1)
    muz_t = np.mean(z_, axis=1)
    return (mux_t, muy_t, muz_t)

def get_freq_mean(x_, y_, z_):
    mux_f = np.mean(np.absolute(fft(x_)), axis=1)
    muy_f = np.mean(np.absolute(fft(y_)), axis=1)
    muz_f = np.mean(np.absolute(fft(z_)), axis=1)
    return (mux_f, muy_f, muz_f)

# Median - 6 features
def get_time_median(x_, y_, z_):
    medx_t = np.median(x_, axis=1)
    medy_t = np.median(y_, axis=1)
    medz_t = np.median(z_, axis=1)
    return (medx_t, medy_t, medz_t)

def get_freq_median(x_, y_, z_):
    medx_f = np.median(np.absolute(fft(x_)), axis=1)
    medy_f = np.median(np.absolute(fft(y_)), axis=1)
    medz_f = np.median(np.absolute(fft(z_)), axis=1)
    return (medx_f, medy_f, medz_f)

# Magnitude - 6 features

# Don't really need magnitude since we already have 5 features from magnitude alone
#magx_t = (LA.norm(accelx_, axis=1) / 200)
#magy_t = (LA.norm(accely_, axis=1) / 200)
#magz_t = (LA.norm(accelz_, axis=1) / 200)

#magx_f = (LA.norm(fft(accelx_), axis=1) / 200)
#magy_f = (LA.norm(fft(accely_), axis=1) / 200)
#magz_f = (LA.norm(fft(accelz_), axis=1) / 200)

# Cross-correlation - 2 features

def get_cross_corr(x_, y_, z_):
    meanx = np.mean(x_, axis=1)
    meany = np.mean(y_, axis=1)
    meanz = np.mean(z_, axis=1)
    corr_xz = meanx/meanz
    corr_yz = meany/meanz
    return (corr_xz, corr_yz)

# Spectral Centroid - 3 features

def spectral_centroid(x_, y_, z_):
    accel_freqx = fft(x_)
    accel_freqy = fft(y_)
    accel_freqz = fft(z_)
    centroids_x = []
    centroids_y = []
    centroids_z = []
    for i in range(len(x_)):
        sumx = 0
        sumy = 0
        sumz = 0
        for j in range(len(x_[i])):
            sumx += (x_[i][j] * accel_freqx[i][j] / 200)
            sumy += (y_[i][j] * accel_freqy[i][j] / 200)
            sumz += (z_[i][j] * accel_freqz[i][j] / 200)
        centroids_x.append(sumx)
        centroids_y.append(sumy)
        centroids_z.append(sumz)
    return (np.absolute(centroids_x), np.absolute(centroids_y), np.absolute(centroids_z))

# Average Difference from the Mean - 3 features

def average_dist_mean(x_, y_, z_, mux_t, muy_t, muz_t):
    realx = []
    realy = []
    realz = []
    dummyx = []
    dummyy = []
    dummyz = []
    for i in range(len(x_)):
        for j in range(len(x_[i])):
            dummyx.append(abs(x_[i][j] - mux_t[i]))
            dummyy.append(abs(y_[i][j] - muy_t[i]))
            dummyz.append(abs(z_[i][j] - muz_t[i]))
        realx.append(np.mean(dummyx))
        realy.append(np.mean(dummyy))
        realz.append(np.mean(dummyz))
    return (realx, realy, realz)

def normalize_dataset(dataframe):
    return (dataframe - dataframe.mean())

def moving_window(accelx, length, step=1):
    streams = it.tee(accelx, length)
    return zip(*[it.islice(stream, i, None, step + 99) for stream, i in zip(streams, it.count(step=step))])

# prior to calling any of these functions please verify that dataframe is normalized & sliding windows are present
def fundamental_frequency(mag_):
    fundamental_freqs = []
    for mag_seg in mag_:
        ft_seg = fft(mag_seg)
        fundamental_freqs.append(LA.norm(np.mean(nlargest(3, ft_seg))))
    return np.asarray(fundamental_freqs)

def average_acceleration(mag_):
    return [np.mean(x) for x in mag_]

def peak_counter(mag_):
    '''
    This function counts the number of peaks in by window in each acceleration dimension using peakutils library
    Below we are averaging peaks over each dimension, combining all axes, and re-averaging for a final feature
    '''
    peaks = []
    for i in range(len(mag_)):
        peaks.append(len(peakutils.indexes(mag_[i])))
    return peaks


def find_max(mag_):
    maxes = [max(x) for x in mag_]
    return maxes

def find_min(mag_):
    mins = [min(x) for x in mag_]
    return mins

def generate_features(healthcode, fn, merged_demographics):
    ''' 
    Applies normalization and sliding windows to the provided data file and 
    returns a dataframe with calculated features
    '''
    a_df = normalize_dataset(pd.read_json(fn).set_index('timestamp'))
    a_df_norm = np.sqrt(np.square(a_df).sum(axis=1))
    mag = np.asarray(a_df_norm)
    mag_ = list(moving_window(mag, 200))
    x = np.asarray(a_df.x)
    y = np.asarray(a_df.y)
    z = np.asarray(a_df.z)
    x_ = list(moving_window(x, 200))
    y_ = list(moving_window(y, 200))
    z_ = list(moving_window(z, 200))
    index = [healthcode for x in range(len(mag_))]
    weight = [merged_demographics.loc[healthcode].weight for y in range(len(mag_))]
    height = [merged_demographics.loc[healthcode].height for z in range(len(mag_))]
    sex = [merged_demographics.loc[healthcode].sex for q in range(len(mag_))]
    currentAge = [merged_demographics.loc[healthcode].currentAge for m in range(len(mag_))]
    # Applied log transformation to the max feature
    return pd.DataFrame({'healthCode': index, 'weight': weight, 'height': height, 'sex': sex, 'currentAge': currentAge, 
                        'fundamental_freq': fundamental_frequency(mag_), 'average_accel': average_acceleration(mag_), 
                        'peakcount': peak_counter(mag_), 'max': find_max(mag_), 'min': np.log(find_min(mag_)),
                        'mut_x': get_time_mean(x_, y_, z_)[0], 'mut_y': get_time_mean(x_, y_, z_)[1], 
                        'mut_z': get_time_mean(x_, y_, z_)[2], 'muf_x': get_freq_mean(x_, y_, z_)[0],
                        'muf_y': get_freq_mean(x_, y_, z_)[1], 'muf_z': get_freq_mean(x_, y_, z_)[2],
                        'medt_x': get_time_median(x_, y_, z_)[0], 'medt_y': get_time_median(x_, y_, z_)[1],
                        'medt_z': get_time_median(x_, y_, z_)[2], 'medf_x': get_freq_median(x_, y_, z_)[0],
                        'medf_y': get_time_median(x_, y_, z_)[1], 'medf_z': get_freq_median(x_, y_, z_)[2],
                        'cross_xz': get_cross_corr(x_, y_, z_)[0], 'cross_yz': get_cross_corr(x_, y_, z_)[1],
                        'spect_cent_x': spectral_centroid(x_, y_, z_)[0], 
                        'spect_cent_y': spectral_centroid(x_, y_, z_)[1],
                        'spect_cent_z': spectral_centroid(x_, y_, z_)[2],
                        'average_dist_meanx': average_dist_mean(x_, y_, z_, np.mean(x_, axis=1), np.mean(y_, axis=1), np.mean(z_, axis=1))[0],
                        'average_dist_meany': average_dist_mean(x_, y_, z_, np.mean(x_, axis=1), np.mean(y_, axis=1), np.mean(z_, axis=1))[1],
                        'average_dist_meanz': average_dist_mean(x_, y_, z_, np.mean(x_, axis=1), np.mean(y_, axis=1), np.mean(z_, axis=1))[2]},
                        columns=['healthCode', 'weight', 'height', 'sex', 'currentAge', 'fundamental_freq', 
                        'average_accel', 'peakcount', 'max', 'min', 'mut_x', 'mut_y', 'mut_z',
                        'muf_x', 'muf_y', 'muf_z', 'medt_x', 'medt_y', 'medt_z', 'medf_x',
                        'medf_y', 'medf_z', 'cross_xz', 'cross_yz', 'spect_cent_x', 'spect_cent_y',
                        'spect_cent_z', 'average_dist_meanx', 'average_dist_meany', 'average_dist_meanz']).set_index('healthCode')


def filter_subjects(directory, table):
    # creates dict with healthcodes and number of gaps from find_gaps function
    new_dict = find_gaps(directory, table)
    # filters dict to remove those entries with more than 2 gaps
    filtered_new_dict = { k:v for k, v in new_dict.items() if v <= 10 }
    healthcodes = []
    for k, v in filtered_new_dict.items():
        healthcodes.append(k)
    return healthcodes

def create_frame(directory, table):
    '''
    Creates a final dataframe with all samples from valid healthcodes
    This: subdir.decode()[subdir.decode().rfind('/')+1:] just gets the healthcode part of a directory 
    Function should be slow with large amounts of files due to the concat Pandas function
    '''
    healthcodes = filter_subjects(directory, table)
    frames = []
    for subdir, dirs, files in os.walk(directory):
        if subdir.decode()[subdir.decode().rfind('/')+1:] in healthcodes:
            i = 0
            for file in files:
                while (i < 1):
                    # This line is generating our features dataframe with healthCode as first arg and filename as second arg
                    frames.append(generate_features(subdir.decode()[subdir.decode().rfind('/')+1:], 
                                                    os.path.join(subdir.decode(), file.decode()), 
                                                    filter_demographics(csv, table)))
                    i += 1
    return pd.concat(frames)

def filter_demographics(demographics, table):
    healthcodes = filter_subjects(directory, table)
    merged = pd.DataFrame()
    indexlist = list(csv.healthCode)
    for e in list(set(healthcodes)):
        if e in indexlist:
            merged = merged.append(csv.loc[csv['healthCode'] == e])
    merged = merged.drop(['recordId'], axis=1)
    merged = merged.set_index('healthCode')
    return merged

def find_lowest_num_samples(total_df):
    '''
    This function finds the minimum number of samples from all of the runs present so that we keep same dimensions
    for every run that we have
    '''
    total = [len(total_df.loc[x].average_accel) for x in list(set(total_df.index))]
    return min(total)

def min_df(lowest_num_of_samples, total_df):
    '''
    Returns the dataframe with features for every healthCode present so that there are only the minimum amount of 
    samples needed
    '''
    newdf = pd.DataFrame()
    unique_healthcodes = list(set(total_df.index))
    for elem in unique_healthcodes:
        newdf = newdf.append(total_df.loc[elem].iloc[:lowest_num_of_samples])
    return newdf

test_v4 = create_frame(directory, v4)

test_v4.to_hdf('tengaps_v4.h5', key='df', mode='w')

#This cell is dedicated to creating a balanced class training set
import matplotlib
matplotlib.use('Agg')
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

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

tengaps_total  = pd.read_hdf('tengaps_total-copy.h5', 'df')

# Min number of samples is 352... total subjects after filtering is 311
tengaps_total = min_df(find_lowest_num_samples(tengaps_total), tengaps_total)

tengaps_total = tengaps_total[tengaps_total.weight < 400]
tengaps_total = tengaps_total[tengaps_total.weight > 80]

# this dataframe doesn't have demographic data
tengaps_no_demog = tengaps_total.drop(columns=['sex', 'currentAge', 'weight', 'height'])

# Inserts sex into 
tengaps_no_demog.insert(25, 'sex', tengaps_total.sex.values)

numOfTotalHC = 311
numOfTrainHC = 217
numOfTestHC = 94

training = tengaps_no_demog.iloc[0:(numOfTrainHC * find_lowest_num_samples(tengaps_no_demog))]
testing = tengaps_no_demog.iloc[(numOfTrainHC * find_lowest_num_samples(tengaps_no_demog)):]

female_train = training.loc[training['sex'] == 'Female']
male_train = training.loc[training['sex'] == 'Male']
choppedmale_train = male_train.iloc[0:17600]

Xmale = np.array(choppedmale_train.iloc[:, 0:25])
Xfemale = np.array(female_train.iloc[:, 0:25])
X_scaledmale = preprocessing.scale(Xmale)
X_scaledfemale = preprocessing.scale(Xfemale)

X_scaledtotal = np.array(list(X_scaledmale)+list(X_scaledfemale))
y_total = np.array(list(choppedmale_train.iloc[:, 25]) + list(female_train.iloc[:, 25]))

X_train, X_test, y_train, y_test = train_test_split(X_scaledtotal, y_total, test_size=0.9, random_state=42)

X_mixedequal = np.concatenate((X_train, X_test), axis=0)
y_mixedequal = np.concatenate((y_train, y_test), axis=0)

# Binary encoding
male_zero = [1 if x=='Female' else 0 for x in y_mixedequal]

# creating odd list of K for KNN
myList = list(range(1, 20))

# subsetting just the odd ones
neighbors = filter(lambda x: x % 2 != 0, myList)

# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_mixedequal, male_zero, cv=5, scoring='roc_auc')
    cv_scores.append(scores.mean())

neighbors = list(filter(lambda x: x % 2 != 0, myList))

# changing to misclassification error
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)

# plot misclassification error vs k
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.savefig("/home/users/bhargavy/gait/python/optimalk.png")

cvscores = open('/home/users/bhargavy/gait/python/cvscores.txt','w')

cvscores.write(str(optimal_k))


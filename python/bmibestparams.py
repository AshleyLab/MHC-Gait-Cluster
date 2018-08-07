import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
tengaps_total  = pd.read_hdf('/home/users/bhargavy/gait/tengaps_total.h5', 'df')

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

# Min number of samples is 352... total subjects after filtering is 311
tengaps_total = min_df(find_lowest_num_samples(tengaps_total), tengaps_total)


tengaps_bmi = tengaps_total[tengaps_total.weight < 400]
tengaps_bmi = tengaps_total[tengaps_total.weight > 80]

# this dataframe doesn't have demographic data
tengaps_bmi = tengaps_total.drop(columns=['sex', 'currentAge', 'weight', 'height'])

def create_bins(BMI):
    '''
    Returns a list of bins (labels) depending on the BMI category
    0 - severely underweight
    1 - underweight
    2 - normal
    3 - overweight
    4 - severely overweight
    '''
    bins = []
    for x in range(0, len(BMI)):
        if BMI[x] < 18.5:
            bins.append(0)
        elif BMI[x] >= 18.5 and BMI[x] < 25:
            bins.append(1)
        elif BMI[x] >= 25 and BMI[x] < 30:
            bins.append(2)
        else:
            bins.append(3)
    return bins

BMI = []
for q in list(set(tengaps_total.index.values)):
    temp = []
    temp.append(((tengaps_total.loc[q].weight.values) / (((tengaps_total.loc[q].height.values)**2))) * 703)
    BMI.append(temp)
    
BMI = np.array(BMI)
BMI = BMI.flatten()

final_bins = create_bins(BMI)
tengaps_bmi.insert(25, 'BMIbins', final_bins)


numOfTotalHC = 311
numOfTrainHC = 217
numOfTestHC = 94

# Shouldn't have to one-hot encode the sex because using random forest
training = tengaps_bmi.iloc[0:(numOfTrainHC * find_lowest_num_samples(tengaps_bmi))]

X = np.array(training.iloc[:, 0:25])
X_scaled = preprocessing.scale(X)
y = np.array(training.iloc[:, 25])

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)

X_mixed = np.concatenate((X_train, X_test), axis=0)
y_mixed = np.concatenate((y_train, y_test), axis=0)
# Splits the training and testing into 80/20 with no sample scrambling
# Must scramble after so that training set has no test samples in it
#training = everything.iloc[0:(numOfTrainHC * find_lowest_num_samples(newtotal_df))]
testing = tengaps_bmi.iloc[(numOfTrainHC * find_lowest_num_samples(tengaps_bmi)):]


n_estimators = [int(x) for x in np.linspace(start = 10, stop = 200, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rfbmi = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_bmi = RandomizedSearchCV(estimator = rfbmi, param_distributions = random_grid, n_iter = 20, scoring='accuracy', cv = 5, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_bmi.fit(X_mixed, y_mixed)

best_params = open('/home/users/bhargavy/gait/python/best_params.txt', 'w')

for k, v in rf_bmi.best_params_.items():
    best_params.write(str(k) + ' >>> '+ str(v) + '\n\n')

best_params.close()

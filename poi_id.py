#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', "bonus", "expenses"] 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
# remove row of totals
data_dict.pop("TOTAL", 0)

### Task 3: Create new feature(s)
import math
# sum all emails and emails to/from poi to calculate share of messages with poi
for person in data_dict:
    to = float(data_dict[person]["to_messages"])
    fr = float(data_dict[person]["from_messages"])
    fr_p = float(data_dict[person]["from_this_person_to_poi"])
    to_p = float(data_dict[person]["from_poi_to_this_person"])
    data_dict[person]["poi_msg_ratio"] = 0
    if not math.isnan(to) and not math.isnan(fr):
        all_msgs = to + fr
        if math.isnan(fr_p):
            fr_p = 0.
        if math.isnan(to_p):
            to_p = 0.
        poi_msgs = to_p + fr_p
        if all_msgs > 0:
            data_dict[person]["poi_msg_ratio"] = poi_msgs / all_msgs
        
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

import sklearn.preprocessing
import sklearn.ensemble
import sklearn.pipeline
# scaler not strictly necessary
clf = sklearn.pipeline.make_pipeline(sklearn.preprocessing.MinMaxScaler(), 
                                     sklearn.ensemble.AdaBoostClassifier())
# for reproducability
clf.set_params(adaboostclassifier__random_state=42)

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

import sklearn.grid_search
# grid search for best params
parameters = {"adaboostclassifier__n_estimators" : (10, 15, 25, 40, 70),
              "adaboostclassifier__learning_rate" : (.1, .2, .4, .7, 1)}
# score recall because that is the lower value for this dataset/ classifier
grid_search = sklearn.grid_search.GridSearchCV(clf, parameters,
                                               scoring="recall", cv=6)
grid_search.fit(features, labels)
# apply found params
clf.set_params(adaboostclassifier__n_estimators=
               grid_search.best_params_["adaboostclassifier__n_estimators"],
               adaboostclassifier__learning_rate=
               grid_search.best_params_["adaboostclassifier__learning_rate"])

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

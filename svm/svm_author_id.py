#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###
from sklearn import svm
from sklearn.metrics import accuracy_score

Cs = [1.0, 10.0, 100.0, 1000.0, 10000.0]



def training(C, features_train=features_train, features_test=features_test, labels_train=labels_train, labels_test=labels_test):
    #clf = svm.SVC(C=1.0, kernel='linear')
    clf = svm.SVC(C=C, kernel='rbf')

    # with smaller dataset, 1% of original
    #features_train = features_train[:len(features_train)/100]
    #labels_train = labels_train[:len(labels_train)/100] 

    t0 = time()
    clf.fit(features_train, labels_train)
    print "trainning time: ", round(time()-t0, 3), "s"

    t0 = time()
    pred = clf.predict(features_test)
    print "prediction time: ", round(time()-t0, 3), "s"

    accuracy = accuracy_score(labels_test, pred)

    print "C = ", C, "; accuracy: ", accuracy

    return pred

        
# try different parameters
#for c in Cs:
#    training(c)
# best accuracy with C = 10000.0

pred = training(10000.0)

print sum(pred), " positives of ", len(pred)

#########################################################



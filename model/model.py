#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 2017

Author: Yanfei Wu
Script for a simple classification model on trading decisions (buy, hold, sell)
"""

import numpy as np
import pandas as pd
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import sys

def process_data(ticker):
    """function to calculate price change for a given stock for generating label"""
    days = 5
    data = pd.read_csv('./data/all_close_data.csv')
    data.set_index('Date', inplace=True)
    data.fillna(method='bfill', inplace=True)
    tickers = data.columns.values.tolist()

    # keep an original copy of the data for feature generation
    data_original = data.copy().drop([ticker], axis=1)
    
    for i in range(1, days+1):
        data['{}_{}d'.format(ticker, i)] = (data[ticker].shift(-i) - data[ticker])/data[ticker]
        
    data.fillna(0, inplace=True)
    return tickers, data_original, data

def buy_hold_sell(*args):
    """
    function to generate label based on price change of the stock
    Note: threshold_1 and threshold_2 are thresholds for generating labels of the data
          these thresholds need to be modified and re-tested for different target stock
    """
    cols = [c for c in args]
    threshold_1 = 0.0075
    threshold_2 = 0.04
    # all(col >= threshold_1 for col in cols) 
    if np.mean(cols) > threshold_1 and any(col > threshold_2 for col in cols):
        return 1 # buy
    elif np.mean(cols) < threshold_1 and any(col < -threshold_2 for col in cols):
        return -1 # sell
    return 0 # hold


def extract_features(ticker):
    """function to calculate daily return as features"""
    tickers, data_original, data = process_data(ticker)

    # generate label 
    labels = list(map(buy_hold_sell, 
                      data['{}_1d'.format(ticker)], 
                      data['{}_2d'.format(ticker)], 
                      data['{}_3d'.format(ticker)], 
                      data['{}_4d'.format(ticker)],
                      data['{}_5d'.format(ticker)]))
    data['{}_target'.format(ticker)] = labels
    str_labels = [str(l) for l in labels]
    print('Label Spread:', Counter(str_labels))

    # generate daily return as features
    features = data_original.pct_change()
    features.fillna(0, inplace=True)
    
    X = features.values
    y = data['{}_target'.format(ticker)].values
    print('Feature shape: {}, target shape: {}'.format(X.shape, len(y)))
    
    return X, y

def model(ticker):
    """function to build classification models"""
    X, y = extract_features(ticker)
    
    # model evaluation
    print('******MODEL EVALUATION****** ')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=55)
    print('Training set size: {} by {}'.format(X_train.shape[0], X_train.shape[1]))
    print('Test set size: {} by {}'.format(X_test.shape[0], X_test.shape[1]))
    
    clf = VotingClassifier([('svc', SVC(kernel='linear', class_weight='balanced', random_state=10,
                                       probability=True)),
                            ('lr', LogisticRegression(class_weight='balanced')),
                            ('rf', RandomForestClassifier(n_estimators=200, 
                                                          max_depth=5,
                                                         random_state=10,
                                                         class_weight='balanced'))],
                          voting='soft')
    
    clf.fit(X_train, y_train)
    print('Training accuracy (with training set):', accuracy_score(y_train, clf.predict(X_train)))
    
    predictions = clf.predict(X_test)
    print('predicted class counts:', Counter(predictions))
    print('Prediction accuracy:', accuracy_score(y_test, predictions))
    
    # make predictions with the model 
    print('******MODEL IN ACTION****** ')
    test_data = pd.read_csv('./data/test_data.csv').drop(ticker, axis=1)
    test_data.set_index('Date', inplace=True)
    test_data = test_data.pct_change().dropna()
    print('Test set size: {} by {}'.format(test_data.shape[0], test_data.shape[1]))
    
    clf = VotingClassifier([('svc', SVC(kernel='linear', class_weight='balanced', 
                                        random_state=30, probability=True)),
                            ('lr', LogisticRegression(class_weight='balanced')),
                            ('rf', RandomForestClassifier(n_estimators=200, 
                                                          max_depth=5,
                                                         random_state=30))],
                          voting='soft')
    clf.fit(X, y)
    print('Training accuracy (with all data):', accuracy_score(y, clf.predict(X)))
    
    predictions = clf.predict(test_data)
    pred_probability = clf.predict_proba(test_data)

    for date, pred, prob in zip(test_data.index, predictions, pred_probability):
        print(date, pred, prob)

    #print(test_data.index)
    #print(predictions)
    #print(clf.predict_proba(test_data))
      
if __name__ == '__main__':
    ticker = sys.argv[1]
    model(ticker)


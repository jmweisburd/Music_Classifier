from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import cPickle as pickle
import os

from training_data import *

class classifiers:
    def __init__(self, training_data):
        self.fft_matrix = training_data.fft_matrix
        self.mfcc_matrix = training_data.mfcc_matrix
        self.t_song_genres = training_data.t_song_genres

        self.train_label = []
        self.test_label = []
        self.train_fft = np.zeros((810, 1000))
        self.train_mfcc = np.zeros((810,13))
        self.test_fft = np.zeros((90, 1000))
        self.test_mfcc = np.zeros((90, 13))

        self.rfc = None
        self.lr = None

    def split_training_data(self, i):
        train_index = 0
        test_index = 0
        for j in range(900):
            if j % 10 == i:
                self.test_label.append(self.t_song_genres[j])
                self.test_fft[test_index] = self.fft_matrix[j]
                self.test_mfcc[test_index] = self.mfcc_matrix[j]
                test_index += 1
            else:
                self.train_label.append(self.t_song_genres[j])
                self.train_fft[train_index] = self.fft_matrix[j]
                self.train_mfcc[train_index] = self.mfcc_matrix[j]
                train_index += 1

        print(self.test_label)

    def train_classifiers(self, feature):
        print("Training classifiers..." + "\n")
        self.rfc = RandomForestClassifier(n_estimators = 100, max_depth = 10)
        self.lr = LogisticRegression()
        if feature == "mfcc":
            self.rfc.fit(self.train_mfcc, self.train_label)
            #self.lr.fit(self.train_mfcc, self.train_label)
        elif feature == "fft":
            self.rfc.fit(self.train_fft, self.train_label)
            #self.lr.fit(self.train_fft, self.train_label)

    def predict(self, feature, data_set):
        if data_set == "test":
            if feature == "mfcc":
                return self.rfc.score(self.test_mfcc, self.test_label)
                #return self.lr.score(self.test_mfcc, self.test_label)
            elif feature == "fft":
                return self.rfc.score(self.test_fft, self.test_label)
                #return self.lr.score(self.train_fft, self.train_label)
                #return self.rfc.predict(self.test_fft)

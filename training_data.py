#from read_wav_files import *
import numpy as np
from constants import *

class training_data:

    def __init__(self):
        self.training_songs = []
        self.label_matrix = np.zeros((1, 900))
        self.ftt_matrix = np.zeros((900, 1000))
        self.mfcc_matrix = np.zeros((900, 13))

    def read_all_songs(self):
        for genre in GENRES:
            for i in range(SONGS_PER_GENRE):
                path = "/genres/" + genre + "/" + genre + "." + get_zeros(i) + str(i) + ".wav"
                print(path)
                #self.training_songs.append(extract_features(path, genre))

    def fill_label_matrix(self):
        for i in range(len(self.training_songs)):
            self.label_matrix[i] = self.training_songs[i].genre

    def fill_ftt_matrix(self):
        for i in range(len(self.training_songs)):
            self.ftt_matrix[i] = self.training_songs[i].fft_features

    def fill_mfcc_matrix(self):
        for i in range(len(self.training_songs)):
            self.fcc_matrix[i] = self.training_songs[i].mfcc_features

def get_zeros(i):
    if i < 10:
        return "0000"
    else:
        return "000"

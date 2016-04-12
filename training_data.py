import scipy
import numpy as np
import scipy.io.wavfile
from scikits.talkbox.features import mfcc

from song import *
import numpy as np
from constants import *

def extract_features(path, genre):
        sample_rate, X = scipy.io.wavfile.read(path)
        fft_features = abs(scipy.fft(X) [:1000])

        ceps, mspec, spec = mfcc(X)
        num_ceps = ceps.shape[0]
        mfcc_features = np.mean(ceps[int(num_ceps*1/10):int(num_ceps*9/10)], axis = 0)

        return song(fft_features, mfcc_features, genre)

class training_data:

    def __init__(self):
        self.training_songs = []
        self.label_list = []
        self.ftt_matrix = np.zeros((900, 1000))
        self.mfcc_matrix = np.zeros((900, 13))

    def read_all_songs(self):
        for genre in GENRES:
            print("I am reading in " + genre)
            for i in range(SONGS_PER_GENRE):
                path = "genres/" + genre + "/" + genre + "." + self.get_zeros(i) + str(i) + ".wav"
                #print(path)
                self.training_songs.append(extract_features(path, genre))

    def fill_label_matrix(self):
        for i in range(len(self.training_songs)):
            self.label_list.append(self.training_songs[i].genre)

    def fill_ftt_matrix(self):
        for i in range(len(self.training_songs)):
            self.ftt_matrix[i] = self.training_songs[i].fft_features

    def fill_mfcc_matrix(self):
        for i in range(len(self.training_songs)):
            self.mfcc_matrix[i] = self.training_songs[i].mfcc_features

    def get_zeros(self, i):
        if i < 10:
            return "0000"
        else:
            return "000"

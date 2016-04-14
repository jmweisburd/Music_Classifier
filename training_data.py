import scipy
import numpy as np
import scipy.io.wavfile
from scikits.talkbox.features import mfcc

import numpy as np
from constants import *
import utility

def extract_features(path):
        sample_rate, X = scipy.io.wavfile.read(path)
        fft_features = abs(scipy.fft(X) [:1000])

        n = fft_features.size
        timestep = (sample_rate/2.)/1000
        max_time = timestep*n
        freq = np.arange(0, max_time, timestep)
        centroid = centroid = np.sum(fft_features*freq)/np.sum(fft_features)

        ceps, mspec, spec = mfcc(X)
        num_ceps = ceps.shape[0]
        mfcc_features = np.mean(ceps[int(num_ceps*1/10):int(num_ceps*9/10)], axis = 0)

        return fft_features, mfcc_features, centroid

class training_data:

    def __init__(self):
        self.t_song_genres = []
        self.fft_matrix = np.zeros((900, 1000))
        self.mfcc_matrix = np.zeros((900, 13))
        self.centroid = np.zeros((900,1))

    def read_all_songs(self):
        song_num = 0
        for genre in GENRES:
            for i in range(SONGS_PER_GENRE):
                path = "genres/" + genre + "/" + genre + "." + utility.get_zeros(i) + str(i) + ".wav"
                print(path)
                self.t_song_genres.append(GENRES.index(genre))
                self.fft_matrix[song_num], self.mfcc_matrix[song_num], self.centroid[song_num] = extract_features(path)
                song_num += 1

    def find_NaN(self):
        #self.mfcc_matrix = np.nan_to_num(self.mfcc_matrix)
        for i in range(900):
            for j in range(13):
                if np.isinf(self.mfcc_matrix[i,j]) or np.isnan(self.mfcc_matrix[i,j]):
                    self.mfcc_matrix[i,j] = 0

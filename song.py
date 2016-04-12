from constants import *

class song:

    def __init__(self, fft_features, mfcc_features, genre):
        self.fft_features = fft_features
        self.mfcc_features = mfcc_features
        self.genre = genre

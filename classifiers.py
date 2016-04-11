import cPickle as pickle
import os

from training_data import *

data_path = "song_data.p"
td = None

def load_song_data():
    if (os.path.isfile(data_path)):
        with open(data_path, 'rb') as handle:
            td = pickle.load(handle)
    else:
        td = training_data()
        td.read_all_songs()
        td.fill_label_matrix()
        td.fill_ftt_matrix()
        td.fill_mfcc_matrix()
        with open(data_path, 'wb') as handle:
            pickle.dump(td, handle)


print("Reading in song data...")
load_song_data()

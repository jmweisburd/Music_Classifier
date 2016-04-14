import cPickle as pickle
import os
from training_data import *

def get_zeros(i):
    if i < 10:
        return "0000"
    else:
        return "000"

def load_training_data():
    data_path = "pickle/training_data.pickle"
    if (os.path.isfile(data_path)):
        with open(data_path, 'rb') as handle:
            td = pickle.load(handle)
            return td
    else:
        td = training_data()
        td.read_all_songs()
        with open(data_path, 'wb') as handle:
            pickle.dump(td, handle)
        return td

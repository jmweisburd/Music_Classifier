from utility import *
from constants import *
from training_data import *
from classifiers import *

print("Reading in song data...")
td = utility.load_training_data()
td.find_NaN()
cl = classifiers(td)

cl.split_training_data(0)
cl.train_classifiers("mfcc")
print(cl.predict("mfcc", "test"))

#import pickle5 as pickle
import pickle as pickle
import pandas as pd

def read_pickle(filename: str):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def write_pickle(filename: str, a):
    with open(filename, 'wb') as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    read_pickle(r'D:\Disturbances')
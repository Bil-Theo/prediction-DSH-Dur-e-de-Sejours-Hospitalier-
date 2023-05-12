import numpy as np
import csv


def extraire():
    file =  open("./datasets/LengthOfStay.csv", "r")

    resultcsv = csv.reader(file, delimiter = ",")
    i = 0
    data = dict()

    for row in resultcsv:
        data[i] = row
        i = i + 1
    
    file.close()
    return data

def numpyOfdict(input, length):

    input = dict(list(input.items())[:length])
    rec = list(input.items())
    data = np.array(rec)
    X = np.array(data[0][1])
    for i in range(data.shape[0]):
        if( i >0): 
            X = np.vstack((X, data[i][1]))

    return X


#[0,1,2,3,26,27]
def purge(data, indesirable):
    data = data[1:, :]
    y = data[:, 27]
    x = np.delete(data, indesirable, axis=1)
    return (x, y)

#eviter l'OverflowError
def normalization(X):
    return  (X.reshape(-1, X.shape[-1]) - X.min())/(X.max() - X.min())
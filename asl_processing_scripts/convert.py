import cv2
import pandas as pd
import numpy as np
import os
import string

print('running convert')

# requirements
# create folder name _datasets (repository for the encoded images)
# specify the resize dimension 
size = 50

for folder in string.ascii_lowercase:
    folder = folder.upper()
    print('processing folder ' + folder)
    baseDir = './' + folder
    filenames = os.listdir(baseDir)

    data = []
    columns = map(lambda x: 'pixel' + str(x),range(1, size * size + 1))

    for filename in filenames:
        img = cv2.imread(baseDir + '/' + filename, cv2.IMREAD_GRAYSCALE)
        resizedImg = cv2.resize(img, (size, size), interpolation = cv2.INTER_AREA)
        encodedImg = resizedImg.flatten()
        data.append(encodedImg)

    df = pd.DataFrame(data, columns = columns)
    df.to_csv('_datasets/'+ folder + '.csv', index = False)
    print('dataset ' + folder + ' created')

print('convert finished')

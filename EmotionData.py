import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

class EmotionData:
    def __init__(self, datapath, labelpath, batch_size=32):
        self.classes = ["neutral", "happiness", "surprise","sadness","anger","disgust","fear","contempt",
            "unknown","NF"]
        self.data = pd.read_csv(datapath)
        self.labels = pd.read_csv(labelpath)
        self.dataGen = self.getGen(batch_size)
            
    def getLabel(self, index):
        F = []
        for col in self.classes:
            F.append( self.labels[col][index]/10 )
        return np.array(F)
    
    def getData(self, index, laplacian = True):
        original = np.mat( self.data.pixels[index] ).reshape(48, 48).astype(np.uint8)
        if laplacian:
            return cv2.Laplacian(original, cv2.CV_64F).reshape((48, 48, 1)).astype(np.uint8)
        return original.reshape((48, 48, 1)).astype(np.uint8)
    
    def getGen(self, batch_size, laplacian = True):
        n = len(self.data)
        while(True):
            for start in range(0, n, batch_size):
                end = min(n, start + batch_size)            
                data =   [ self.getData(i, laplacian) for i in range(start, end)]
                labels = [ self.getLabel(i)  for i in range(start, end) ]
                yield np.array(data), np.array(labels)

    def getWhole(self, laplacian):
        n = len(self.data)
        data =   [ self.getData(i, laplacian) for i in range(n)]
        labels = [ self.getLabel(i)  for i in range(n) ]
        return np.array(data), np.array(labels)
        
    def showData(self):
        index = random.randint(1,30000)
        original = self.getData(index, laplacian=False)
        laplacian = self.getData(index)
        y = self.getLabel(index)
        emotion = self.classes[ np.argmax(y) ]
        prob = y.max()
        emotion = f"{emotion} : {prob}"
        # plot the original image on right
        fig, axs = plt.subplots(1,2)
        axs[1].imshow(255 - original, cmap='Greys')
        axs[1].set_title(emotion)

        # and laplacian on the left
        axs[0].imshow(laplacian, cmap='Greys')
        axs[0].set_title(emotion)
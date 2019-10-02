import numpy as np
import matplotlib.pyplot as plt
import math
from  tqdm import tqdm

class logisticRegressor(object):
    def __init__(self, n_clas, n_feat, reg):
        self.classes = n_clas
        self.features = n_feat
        self.W = np.ones((n_feat,n_clas))/n_feat
        self.b = np.ones((1,n_clas))/n_feat
        self.lam = reg
        print("Weights shape:", self.W.shape)
        print("Bias shape: ", self.b.shape)

    def __len__(self):
        return self.classes

    def oneHot(self, y):
        m = y.shape[0]
        Y = np.zeros((m, self.classes), dtype=np.int)
        Y[np.arange(m), y] = 1
        return Y

    def hyp(self, X):
        Z = np.matmul(X, self.W) + self.b
        return np.exp(Z)/(1+np.exp(Z))

    def gradient(self, X, Y):
        m = X.shape[0]
        D = self.hyp(X) - Y
        return (np.matmul(X.T, D) + self.lam * self.W) / m

    def cost(self, X, Y):
        # linear regression loss
        m = X.shape[0]
        H = self.hyp(X)
        loss = Y * np.log(H) + (1-Y) * np.log(1-H)
        loss = - np.sum(loss, axis=0)
        reg = np.sum(self.W**2, axis=0)/2
        return (loss + self.lam*reg)/m
    
    def fit(self, data, epochs=100, learning_rate=0.1, val=False):
        history = {}
        if val:
            (X, y), (x_val, y_val) = data
            Y_val = self.oneHot(y_val)
            history['val_loss'] = np.empty( (epochs, self.classes) )
        else:
            (X, y) = data
        
        # convert to one hot encoding
        Y = self.oneHot(y)
        history['train_loss'] = np.empty( (epochs, self.classes) )
        for i in tqdm(range(epochs)):
            self.W = self.W - learning_rate * self.gradient(X, Y)
            history['train_loss'][i] = np.round(self.cost(X,Y), 4)
            if val:
                history['val_loss'][i]=np.round(self.cost(x_val, Y_val), 4)
        return history

    def plot_history(self, history, width=3):
        nrows = math.ceil(self.classes/width)
        ncols  = width
        index = 0
        fig, axs = plt.subplots(nrows, ncols, figsize=(10*ncols, 4.8*nrows))
        for row in range(nrows):
            for col in range(ncols):
                if index < self.classes:
                    axs[row, col].plot(
                        history['train_loss'][:,index],
                        label = 'train_loss'    
                    )
                    if 'val_loss' in history:
                        axs[row, col].plot(
                            history['val_loss'][:,index],
                            label = 'val_loss'    
                        )
                    axs[row, col].legend()
                    index+=1
        plt.show()
    
    def accuracy(self, X, y):
        # take max along cols, the 2nd dim
        predictions = np.argmax(self.hyp(X), axis=1)
        correct = np.sum((predictions==y).astype(np.int))
        return round(correct/y.shape[0],4)

    def predict(self, X, prob=False):
        m = X.shape[0]
        if X.ndim == 1:
            X = X.reshape((1,m))
            m=1
        h = self.hyp(X)
        indices = np.argmax(h, axis=1)
        if prob:
            return indices, np.round(h[np.arange(m), indices],2)
        else:
            return indices
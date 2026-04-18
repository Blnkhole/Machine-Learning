import pandas as pd
import numpy as np

class SoftmaxRegression:

    def __init__(self, X, Y, n_classes = 10, lr = 0.1, batch_size = 600):
        self.X = np.array(X, dtype = float)
        self.Y = np.array(Y)
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.W = np.zeros((self.X.shape[1], n_classes)) 
        self.b = np.zeros((1, n_classes))
        self.cross = np.eye(n_classes)[self.Y]     
        self.lr = lr     

    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis = 1, keepdims = True))
        return exp_Z / np.sum(exp_Z, axis = 1, keepdims = True)

    def loss(self, X, Y_true):
        Z = np.dot(X, self.W) + self.b
        Y_pred = self.softmax(Z)
        eps = 1e-12
        Y_pred = np.clip(Y_pred, eps, 1 - eps)
        Y_true_onehot = np.eye(self.n_classes)[Y_true]
        loss = -np.mean(np.sum(Y_true_onehot * np.log(Y_pred), axis = 1))
        return loss

    def train(self, epoches = 20, X_test = None, Y_test = None):

        n_samples = self.X.shape[0] 

        for loop in range(epoches):

            indices = np.arange(n_samples) 
            np.random.shuffle(indices)
            X_shuffled = self.X[indices]
            Y_shuffled = self.cross[indices]

            for start in range(0, n_samples, self.batch_size):

                end = start + self.batch_size

                X_batch = X_shuffled[start : end]
                Y_batch = Y_shuffled[start : end]  

                Z = np.dot(X_batch, self.W) + self.b        
                Y_pred = self.softmax(Z)      

                grad_W = np.dot(X_batch.T, (Y_batch - Y_pred)) / X_batch.shape[0]
                grad_b = np.mean(Y_batch - Y_pred, axis = 0, keepdims = True)

                self.W += self.lr * grad_W
                self.b += self.lr * grad_b

            if loop % 10 == 0:
                train_loss = self.loss(self.X, self.Y)
                msg = f"Epoch {loop:3d} | Train loss: {train_loss:.4f}"
                if X_test is not None and Y_test is not None:
                    test_loss = self.loss(X_test / 255.0, Y_test)
                    msg += f" | Test loss: {test_loss:.4f}"
                print(msg)

    def predict(self, X):

        X = np.array(X, dtype = float)
        Z = np.dot(X, self.W) + self.b
        Y_pred = self.softmax(Z)

        return np.argmax(Y_pred, axis = 1)

    def precision_recall(self, Y_true, Y_pred):

        TP = np.sum((Y_pred == 1) & (Y_true == 1))
        FP = np.sum((Y_pred == 1) & (Y_true == 0))
        FN = np.sum((Y_pred == 0) & (Y_true == 1))

        precision = TP / (TP + FP + 1e-12)
        recall    = TP / (TP + FN + 1e-12)
        
        return precision, recall
            

df_train = pd.read_csv("creditcard_train.csv")
X_train = df_train.drop(columns = ['Class', 'Time']).values
Y_train = df_train["Class"].values

df_test = pd.read_csv("creditcard_test.csv")
X_test = df_test.drop(columns = ['Class', 'Time']).values
Y_test = df_test["Class"].values

model = SoftmaxRegression(X_train, Y_train, lr = 0.1, batch_size = 600)
model.train(epoches = 100, X_test = X_test, Y_test = Y_test) 

y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == Y_test)
print(f"\nTest accuracy: {accuracy * 100 : .2f}%")

precision, recall = model.precision_recall(Y_test, y_pred)
print(f"Precision     : {precision * 100: .2f}%")
print(f"Recall        : {recall * 100: .2f}%")

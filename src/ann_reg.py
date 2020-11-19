import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data_handler import get_data
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

class Net(nn.Module):

    def __init__(self, n_features, n_hidden, n_output):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden) 
        self.fc3 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        
        x = self.fc1(x)

        x = F.relu(x)

        x = self.fc3(x)

        return x

class MyDataset(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)



class ANN_trainer:

    def __init__(self):

            self.LEARNING_RATE = 0.001
            self.L2_REG = 0.9
            self.EPOCHS = 50
            self.BATCH_SIZE = 32

            self.n_hidden = []
            self.error = []

            self.test_predict = []
            self.test_true = []

            self.train_loss = []


    def train(self, train_loader, net, criterion, optimizer):

        train_loss_list = []

        for e in range(self.EPOCHS):

            epoch_loss = 0

            for X_train_batch, y_train_batch in train_loader:
                
                pred = net(X_train_batch)

                train_loss = criterion(pred, y_train_batch.unsqueeze(1)) 

                train_loss.backward()

                optimizer.step()

                epoch_loss += train_loss.item()
            
            train_loss_list.append(epoch_loss/len(train_loader))
        
        return train_loss_list
        

    def test(self, test_loader, net, criterion):
        pred_list = []
        y_test_list = []

        with torch.no_grad():
            for X_test_batch, y_test in test_loader:
                pred = net(X_test_batch).numpy()[0][0]
                pred_list.append(pred)
                y_test_list.append(y_test.numpy()[0])


        return np.power(np.array(pred_list)-np.array(y_test_list),2).sum().astype(float)/np.array(y_test_list).shape[0], pred_list, y_test_list


    def innerrun(self, M, n_hiddens, data_train_outer, target_train_outer, data_test_outer, target_test_outer):

        best_hidden = 0
        error = None

        predict_list = []
        y_test_list = []

        train_loss = []
        test_loss = []
        best_hiddens = []


        for index in range(len(data_train_outer)):



                X_train = data_train_outer[index]
                y_train = target_train_outer[index]
                X_test = data_test_outer[index]
                y_test = target_test_outer[index]


                train_dataset = MyDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
                test_dataset = MyDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())

                train_loader = DataLoader(dataset=train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
                test_loader = DataLoader(dataset=test_dataset, batch_size=1)



                for n_hidden in n_hiddens:

                    best_hidden_1 = 0

                    error_1 = None

                    predict_best = []

                    y_test_best = []

                    train_loss_best = []
                    test_loss_best = []

                    for i in range(5):

                        net = Net(M, n_hidden, 1)

                        optimizer = optim.SGD(net.parameters(), lr=self.LEARNING_RATE, weight_decay=self.L2_REG)

                        criterion = nn.MSELoss()

                        train_loss = self.train(train_loader, net, criterion, optimizer)

                        n_error, n_predict_list, n_y_test_list = self.test(test_loader, net, optimizer) 

                        if error_1 == None or (n_error< error_1):
                            error_1 = n_error
                            best_hidden_1 = n_hidden
                            predict_list=predict_best
                            y_test_list=y_test_best
                            train_loss_best = train_loss
                            test_loss_best = n_error                           



                    if error == None or (error_1< error):
                          error = error_1
                          best_hidden = best_hidden_1

                    predict_list.append(predict_best)
                    y_test_list.append(y_test_best)
                    train_loss.append(train_loss_best)
                    test_loss.append(train_loss_best)
                    best_hiddens.append(best_hidden_1)
        
        return best_hidden, error, predict_list, y_test_list, train_loss, test_loss, best_hiddens


    def run(self, N, M, attributeNames, n_hiddens, data_train, target_train, data_test, target_test, data_train_outer, target_train_outer, data_test_outer, target_test_outer):

        for index in range(len(data_train)):

            X_train_outer = data_train[index]
            y_train_outer = target_train[index]
            X_test_outer = data_test[index]
            y_test_outer = target_test[index]


            train_dataset_outer = MyDataset(torch.from_numpy(X_train_outer).float(), torch.from_numpy(y_train_outer).float())
            test_dataset_outer = MyDataset(torch.from_numpy(X_test_outer).float(), torch.from_numpy(y_test_outer).float())

            train_loader_outer = DataLoader(dataset=train_dataset_outer, batch_size=self.BATCH_SIZE, shuffle=True)
            test_loader_outer = DataLoader(dataset=test_dataset_outer, batch_size=1)

            best_n_hidden, _, _, _, _, _, _ = self.innerrun(M, n_hiddens, data_train_outer[index], target_train_outer[index], data_test_outer[index], target_test_outer[index])


            error = None
            pred = None
            true = None
            train_loss = None

            for i in range(5):
                
                net = Net(M, best_n_hidden, 1)

                optimizer = optim.SGD(net.parameters(), lr=self.LEARNING_RATE, weight_decay=self.L2_REG)

                criterion = nn.MSELoss()

                n_train_loss = self.train(train_loader_outer, net, criterion, optimizer)

                n_error, n_pred, n_true = self.test(test_loader_outer, net, criterion)


                
                if error==None or (n_error < error):
                    error = n_error
                    pred = n_pred
                    true = n_true
                    train_loss = n_train_loss
            

            self.test_true.append(true)
            self.test_predict.append(pred)
            self.n_hidden.append(best_n_hidden)
            self.error.append(error)
            self.train_loss.append(train_loss)

            print("-----------------ANN_run" +str(index)+ "-----------------")


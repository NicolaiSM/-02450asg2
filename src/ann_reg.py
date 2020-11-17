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
            self.EPOCHS = 150
            self.BATCH_SIZE = 64



            

    def train(self, train_loader, net, criterion, optimizer):

        train_loss = []

        for e in range(self.EPOCHS):

            epoch_loss = 0

            for X_train_batch, y_train_batch in train_loader:
                
                pred = net(X_train_batch)

                train_loss = criterion(pred, y_train_batch.unsqueeze(1)) 

                train_loss.backward()

                optimizer.step()

                epoch_loss += train_loss.item()
            
            train_loss.append(epoch_loss/len(train_loader))
        
        return train_loss
        

        

    def test(self, test_loader, net):
        pred_list = []
        y_test_list = []

        with torch.no_grad():
            for X_test_batch, y_test in test_loader:
                pred = net(X_test_batch).numpy()
                pred_list.append(pred)
                y_test_list.append(y_test.numpy())

        return np.power(pred_list-y_test_list,2).sum().astype(float)/y_test_list.shape[0]




    def run(self, X, y, N, M, attributeNames, CV, n_hiddens):

        for train_index, test_index in CV.split(X, y):



            X_train_outer = X[train_index]
            y_train_outer = y[train_index]
            X_test_outer = X[test_index]
            y_test_outer =y[test_index]


            train_dataset_outer = MyDataset(torch.from_numpy(X_train_outer).float(), torch.from_numpy(y_train_outer).float())
            test_dataset_outer = MyDataset(torch.from_numpy(X_test_outer).float(), torch.from_numpy(y_test_outer).float())

            train_loader_outer = DataLoader(dataset=train_dataset_outer, batch_size=self.BATCH_SIZE, shuffle=True)
            test_loader_outer = DataLoader(dataset=test_dataset_outer, batch_size=1)


            inner_bedst = 


            for train_index, test_index in CV.split(X_train_outer,y_train_outer):


                X_train = X[train_index]
                y_train = y[train_index]
                X_test = X[test_index]
                y_test =y[test_index]

                train_dataset = MyDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
                test_dataset = MyDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())

                train_loader = DataLoader(dataset=train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
                test_loader = DataLoader(dataset=test_dataset, batch_size=1)

                best_hidden = None

                for n_hidden in n_hiddens:


                    net = Net(M, n_hidden, 1)

                    optimizer = optim.SGD(net.parameters(), lr=self.LEARNING_RATE, weight_decay=self.L2_REG)

                    criterion = nn.MSELoss()

                    self.train(train_loader, net, criterion, optimizer)

                    self.test(test_loader, net)





            self.train(train_loader_outer, net, criterion, optimizer)

            self.test(test_loader_outer, net)


                
"""

Need to implement cross validation

"""




























        









"""

def train(train_loader):

    data, target, n_observation, n_features, attributeNames = get_data()

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=69)

    train_dataset = RegressionDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    test_dataset = RegressionDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())

    EPOCHS = 150
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_FEATURES = n_features


    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1)



    net = Net(n_features, n_features, 1)

    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, weight_decay=0.9)

    criterion = nn.MSELoss()

    test_loss = []

    for e in range(EPOCHS):

        epoch_loss = 0

        for X_train_batch, y_train_batch in train_loader:
            
            pred = net(X_train_batch)

            train_loss = criterion(pred, y_train_batch.unsqueeze(1)) 

            train_loss.backward()

            optimizer.step()

            epoch_loss += train_loss.item()
        
        test_loss.append(epoch_loss/len(train_loader))

    plt.plot(list(range(EPOCHS)), test_loss)
    plt.show()

def test(test_loader):
    pred_list = []
    y_test_list = []
    x_test = []
    with torch.no_grad():
        for X_test_batch, y_test in test_loader:
            pred = net(X_test_batch).numpy()
            pred_list.append(pred)
            y_test_list.append(y_test.numpy())
            x_test.append(X_test_batch.numpy())



    y_pred_list = [a.squeeze().tolist() for a in pred_list]
    x_test = [a.squeeze().tolist() for a in x_test]
    #y_test_list = [a.squeeze().tolist() for a in y_test_list]

    mse = mean_squared_error(y_test_list, y_pred_list)
    r_square = r2_score(y_test_list, y_pred_list)

    print(mse)
    print(np.sqrt(mse))
    print(r_square)




train(train_loader)
test(test_loader)



"""

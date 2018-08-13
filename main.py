import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
from Event import Event
import pandas as pd
import numpy as np
import pickle
import sys


# UTIL FUNCTIONS
def grouplen(sequence, chunk_size):
    return list(zip(*[iter(sequence)] * chunk_size))

#LOADING SETS
def load_data_sets():
    filename = "train.p"
    train_list= pickle.load(open(filename , "rb" ))
    train_data=[]
    train_position=[]
    for event in train_list:
        for moment in event.moments:
            for player in moment.players:
                train_data.append(player.id)
                train_data.append(player.x)
                train_data.append(player.y)
                train_position.append(player.x)
                train_position.append(player.y)

    #has team,id,x,y for every player for every moment       
    train_data = grouplen(train_data,3)
    torch.set_printoptions(precision=8)
    train_data = torch.tensor(np.array(train_data))
    train_data = Variable(train_data)
    #train_position has every player position every event
    train_position = grouplen(train_position,2)
    train_position = torch.tensor(np.array(train_position))
    train_position = Variable(train_position)

    data = pickle.load(open("test_data.p" , "rb" ))
    for j in range(151):
        test_data=[]
        test_position=[]        
        for i in range(6):
            for obj in vars(data[j][i])["players"]:
                test_data.append(obj.team.id)
                test_data.append(obj.id)
                test_data.append(obj.x)
                test_data.append(obj.y)
                test_position.append(obj.x)
                test_position.append(obj.y)

    #test_Data has team,id,x,y for every player for every moment with (-1,-1)       
    test_data = grouplen(test_data,4)
    test_data = grouplen(test_data,10)
    test_data = torch.tensor(np.array(test_data), dtype = torch.double)
    test_data = Variable(test_data)

    #test_position has x,y for every player for every moment with (-1,-1)
    test_position = grouplen(test_position,2)
    test_position = grouplen(test_position,10)
    test_position = torch.tensor(np.array(test_position), dtype = torch.double)
    test_position = Variable(test_position)

    # print test_position
    # print test_data

    # num = test_data.size()
    # num2 = test_position.size()
    # num3 = torch.numel(test_data)
    # num4 = torch.numel(test_position)
    # print 'Tensor sizes: ', num,' ', num2
    return train_data,train_position


#RNN_LSTM N
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size,2)

    def forward(self, x):
        # Set initial hidden and cell states 
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        h0 = Variable(torch.zeros(self.num_layers, 1 ,self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers, 1 ,self.hidden_size))
        # Forward propagate LSTM
        out, hidden = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        # out = self.out(out[:, -1, :])
        out = out.view(-1, hidden_size) 
        return out


def train_LSTM(path_to_store_weight_file=None, number_of_iteration=1):
    #HYPER-PARAMETERS
    sequence_length = 28
    input_size = 180
    hidden_size = 1
    num_layers = 1
    batch_size = 64 #number of sequences I want to process in parallel
    num_epochs = 1 #train the data 1 time
    learning_rate = 0.1 #learning rate
   
   

    train_data,train_position,test_data,test_position = load_data_sets()
    print('Train data size', train_data.size())
    print('Train output size',train_position.size())
    print('Test data size', test_data.size())
    print('Test output size',test_position.size())
    
    model = RNN(input_size, hidden_size, num_layers)
    print model

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss = nn.CrossEntropyLoss()

    #begin to train 
    for epoch in range(number_of_iterarion):
        print 'STEP:', epoch
        # Pytorch accumulates gradients. We need to clear them out before each instance
        optimizer.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
       
        state = Variable(torch.zeros(num_layers,batch_size,hidden_size))
        cell = Variable(torch.zeros(num_layers,batch_size,hidden_size))
      
        out, state = RNN(train_data, state, cell)
        #TINC EL PROBLEMA AQUI AMB EL TRAINING DATA
    
        err = loss(out, train_position)
        err.backward()
        optimizer.step()

    print('-------done')
    torch.save(model, path_to_store_weight_file)

    # # Esto lo que hace es mirar en el test_data los jugadores que tengo que predecir la position 
    # # # i = 1
    # # gt_test = "test_data.p"
    # # gt_list = pickle.load(open(gt_test , "rb"))
    # # for lista in gt_list:
    # #     for moment in lista:
    # #         for player in moment.players:
    # #             if(player.x == -1 and player.y == -1):
    # #                 print 'Momento ', i
    # #                 print 'Jugador encontrado que tengo que predecir ', player.get_info()
    # #         i = i+ 1
    
#TWO LAYER NN
class TwoLayerNet(nn.Module):
    def __init__(self,D_in,H,D_out):
       super(TwoLayerNet,self).__init__()
       self.linear1 = torch.nn.Linear(D_in,H)
       self.linear2 = torch.nn.Linear(H,D_out)
    def forward(self,x):
        h_relu =self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

def train(path_to_store_weight_file=None, number_of_iteration=1):
    N, D_in, H, D_out = 8, 30, 100, 15

    x , y = load_data_sets()
    #pensar en això
    # x1 = torch.randn(N, D_in)
    # y1 = torch.randn(N, D_out)

    x = x.view(N,D_in)
    y = y.view(N,D_out)
    model = TwoLayerNet(D_in, H, D_out)


    loss_fn = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    for t in range(number_of_iteration):
        y_pred = model(x.float())
        loss = loss_fn(y_pred, y.float())
        print(t, loss.item())
        # zero gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print('-------done')
    torch.save(model, path_to_store_weight_file)


            
if sys.argv[1] == "train":
    train(sys.argv[2], int(sys.argv[3]))
elif sys.argv[1] == "test":
    test(sys.argv[2], sys.argv[3], sys.argv[4])   
elif sys.argv[1] == "train_LSTM":
    train(sys.argv[2], int(sys.argv[3]))
elif sys.argv[1] == "data":
    load_data_sets()
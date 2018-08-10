import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
from Event import Event
import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot





# UTIL FUNCTIONS
def sigmoid(x, derivative = False):
    if(derivative == True):
        return x*(1-x)
    return 1/(1+np.exp(-x))


def ReLU(x, derivative = False):
    if(derivative == True):
        return 1. * (x > 0)
    return x * (x > 0)

def convert2tensor(x):
    x = torch.FloatTensor(x)
    return x

def Save():
    torch.save(model, PATH)

def grouplen(sequence, chunk_size):
    return list(zip(*[iter(sequence)] * chunk_size))


#RECURRENT NEURAL NETWORK
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size)
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, hidden = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


# if __name__ == '__main__':

#     #HYPER-PARAMETERS
#     sequence_length = 28
#     input_size = 1 
#     hidden_size = 128
#     num_layers = 1
#     batchsize = 1 #number of sequences I want to process in parallel
#     num_epochs = 1 #train the data 1 time
#     learning_rate = 0.01 #learning rate

#     model = RNN(input_size, hidden_size, num_layers)
#     #Loss, optimizer
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#     for epoch in range(num_epochs):
#         for i , data in enumerate(train_loader,0):

#                 inputs, labels = data
#                 inputs = Variable(inputs.view(-1, batch_size, input_size))
#                 labels = Variable(labels.view(-1, batch_size, input_size))

#                 print('Input/Label Size :::: ', inputs.size(), labels.size())

#                 state = Variable(torch.zeros(num_layer,batch_size,hidden_size))
#                 cell = Variable(torch.zeros(num_layer,batch_size,hidden_size))

#                 out, state = RNN(inputs, state, cell)

#                 optimizer.zero_grad()

#                 out = out.view(-1, hidden_size)
#                 labels = labels.view(-1).long()

#                 print('Output/Label Size :::: ', out.size(), labels.size())

#                 loss = nn.CrossEntropyLoss()
#                 err = loss(out, labels)
#                 err.backward()
#                 optimizer.step()

#                 print('[input]', inputs.view(1,-1))
#                 print('[target]', labels.view(1,-1))
#                 print('[prediction] ', out.data.max(1)[1])


#     print('-------done')


# def load_train_test():


filename = "train.p"
train_list= pickle.load(open(filename , "rb" ))
train_data=[]
train_position=[]
for event in train_list:
    for moment in event.moments:
        for player in moment.players:
            train_data.append(player.get_info())
            train_position.append(player.x)
            train_position.append(player.y)

#has teamid,x,y for every player for every moment       
train_data = grouplen(train_data,4)
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
            test_data.append(obj.get_info())
            test_position.append(player.x)
            test_position.append(player.y)

#test_Data has teamid,x,y for every player for every moment with (-1,-1)       
test_data = grouplen(test_data,4)
test_data = torch.tensor(np.array(test_data))
test_data = Variable(test_data)

#test_position has x,y for every player for every moment with (-1,-1)
test_position = grouplen(test_position,2)
test_position = torch.tensor(np.array(test_position))
test_position = Variable(test_position)

# return train_data,train_position,test_data,test_position



  
#     # #TODO: fer lo mateix per train i mirar que polles hi ha 
#     # 



# gt_data = pickle.load(open("gt_data.p", "rb"))
# data = pickle.load(open("test_data.p" , "rb" ))
# for j in range(151):
#     list_1=[]
#     for i in range(6):
#         list_1.append('Players(team,id,x,y): ')
#         for obj in vars(data[j][i])["players"]:
#             list_1.append(obj.get_info())
#         list_1.append('Quarter: ')
#         list_1.append(vars(data[j][i])["quarter"])
#         list_1.append('Game Clock: ')
#         list_1.append(vars(data[j][i])["game_clock"])
#         list_1.append('Ball(x,y,radius): ')
#         list_1.append(vars(data[j][i])["ball"].get_info())
#         list_1.append('Shot Clock: ')
#         list_1.append(vars(data[j][i])["shot_clock"])


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
 


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


def load_train_test():

    gt_data = pickle.load(open("gt_data.p", "rb")) #gt_data contains the correct position output
    positions = torch.from_numpy(np.array(gt_data))

    data = pickle.load(open("test_data.p" , "rb" ))
    for j in range(151):
        test_y=[]
        test_x =[]
        for i in range(6):
            for obj in vars(data[j][i])["players"]:
                test_y.append(obj.get_info())
            test_x.append(vars(data[j][i])["quarter"])
            test_x.append(vars(data[j][i])["game_clock"])
            test_x.append(vars(data[j][i])["ball"].get_info())
            test_x.append(vars(data[j][i])["shot_clock"])
            
    test_y = grouplen(test_y,14)
    torch.set_printoptions(precision=8)
    test_y = torch.tensor(np.array(test_y))
    test_y = Variable(test_y)

    test_x = grouplen(test_x,14)
    test_x = torch.tensor(np.array(test_x))
    test_x = Variable(test_x)
    
    return Variable(test_y),Variable(test_x),Variable(positions)


# data = torch.from_numpy(np.array(list_1))
# print data

# for x in range(len(gt_data)):
#    xs =  gt_data[x][0]
#    ys = gt_data[x][1]

# for x in range(len(list_1)):
#     print list_1[x]

# plt.scatter(xs, ys)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()
# trainloader = torch.utils.data.DataLoader(data,batch_size=batchsize,shuffle=True,num_workers=2)
# output = torch.utils.data.DataLoader(gt_data,batch_size=batchsize,shuffle=True,num_workers=2)


# df = pd.DataFrame(columns=['Players','Quarter','Game Clock','Ball','Shot Clock'])
# df.head()

# plt.figure(figsize = (18,9))
# plt.plot(range(df.shape[0]),(df['Low']+df['High'])/2.0)
# plt.xticks(range(0,df.shape[0],500),df['Player'].loc[::500],rotation=45)
# plt.xlabel('X',fontsize=18)
# plt.ylabel('Y',fontsize=18)
# plt.show()


#     x = Variable(torch.Tensor(list_1).type(dtype), requires_grad=False)
#     y = Variable(torch.Tensor(gt_data).type(dtype), requires_grad=False)


  
#     # #TODO: fer lo mateix per train i mirar que polles hi ha 
#     # filename = "train.p"
#     # test_list= pickle.load(open(filename , "rb" ))
#     # test_data=[]
#     # for event in test_list:
#     #     for moment in event.moments:
#     #         for player in moment.players:
#     #             test_data.append(player.get_info())
#     #         test_data.append(moment.game_clock)
#     #         test_data.append(moment.quarter)
#     #         test_data.append(moment.ball)
#     #         test_data.append(moment.shot_clock)

#     # print 'Size of test_data', len(test_data)
#     # for x in range(len(test_data)):
#     #     print test_data[x]




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
 


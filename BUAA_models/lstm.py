import torch
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd
import numpy as np
import random
from Event import Event
import pickle
import sys

def generate_sample_test():
    filename = "train.p"
    #Change train.p to /Data/train.p
    test_list= pickle.load(open(filename , "rb" ))

    given_length = 5
    predict_period = 2
    total_len = given_length + predict_period

    output = [] # data used for train/test
    gt_data = [] # ground truth data
    
    for event in test_list:
        if len(event.moments) > 200:
            duration = int(event.moments[0].game_clock - event.moments[-1].game_clock)
            game_clock_list = []
            game_clock_s_list = []
            for moment in event.moments:
                if int(moment.game_clock) not in game_clock_s_list:
                    game_clock_s_list.append(int(moment.game_clock))            
                    game_clock_list.append(moment.game_clock)
            game_clock_list = list(sorted(set(game_clock_list)))
            data_list = game_clock_list[2:-2] # remove first and last 2 seconds
            if len(data_list) > 10:
                random_index = random.randrange(0,len(data_list[:-7]))
                index_list = range(random_index, random_index+total_len, 1)
                choosen_time = np.asarray(game_clock_list)[index_list]
                choosen_time = np.delete(choosen_time, 5) # remove 6th second data
                given_seq = choosen_time[:5]
                predict_seq = choosen_time[-1]
            
                test_data = []
                pred_data = []
                for moment in event.moments:
                    # print moment.game_clock, predict_seq,moment.game_clock == predict_seq
                    if moment.game_clock in list(given_seq):
                        test_data.append(moment)
                    elif moment.game_clock == predict_seq:
                        pred_data = moment
                rand_player = random.choice(pred_data.players) # select a random player from list
                ground_truth = (rand_player.x, rand_player.y)
                rand_player.x = -1
                rand_player.y = -1
                test_data.append(pred_data)
                # ground_truth
                gt_data.append(ground_truth)
                output.append(test_data)

    return output, gt_data
    # pickle.dump(output, open("test_data.p" , "wb" ))
    # pickle.dump(gt_data, open("gt_data.p" , "wb" ))

#RNN_LSTM 
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,batch_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size,hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size,2)
        self.batch_size = batch_size
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Set initial hidden and cell states 
        return( Variable(torch.zeros(self.num_layers, self.batch_size ,self.hidden_size)), Variable(torch.zeros(self.num_layers, self.batch_size ,self.hidden_size)))

    def forward(self, x):
        # Forward propagate LSTM
        lstm_out, self.hidden = self.lstm(Variable(x), self.hidden) 
        out = self.out(lstm_out.view(-1,self.hidden_size))
        # out = self.out(lstm_out[:, -1, :])
        # Decode the hidden state of the last time step
        # out = self.out(out[:, -1, :])
        #out = out.view(-1, hidden_size) 
        return out

def train_LSTM(path_to_store_weight_file=None, number_of_iteration=1):
    #HYPER-PARAMETERS
    input_size = 2436
    output_size = 2
    hidden_size = 100
    num_layers = 1
    batch_size = 151 #number of sequences I want to process in parallel
    num_epochs = 1 #train the data 1 time
    learning_rate = 0.1 #learning rate
   
   

    def flatten(list_):
        for el in list_:
            if hasattr(el, "__iter__") and not isinstance(el, basestring):  
                for sub in flatten(el):  
                    yield sub  
            else:  
                yield el
    output, test_position= generate_sample_test() 
    # generate_sample_test() 
    # output= pickle.load(open("test_data.p" , "rb" ))
    # test_position = pickle.load(open("gt_data.p","rb"))
    list = []
    for j in range(151):
        test_data=[] 
        for i in range(6):
            for obj in vars(output[j][i])["players"]:
                for obj in vars(output[j][i])["players"]:
                    test_data.append(obj.get_info())
            test_data.append(vars(output[j][i])["quarter"])
            test_data.append(vars(output[j][i])["game_clock"])
            test_data.append(vars(output[j][i])["ball"].get_info())
            test_data.append(vars(output[j][i])["shot_clock"])
        list_1_scaled = [x for x in flatten(test_data)]
        list.append(list_1_scaled)         
    
    data = pd.DataFrame(list)
    data_1 = data.copy()
    data_1 = data_1.values
    from sklearn.preprocessing import Imputer
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    imputer = Imputer(strategy="mean")
    imputer.fit(data_1)
    data_1 = imputer.transform(data_1)
    data_1_scaled = scaler.fit_transform(data_1)
    test_data = torch.DoubleTensor(np.array(data_1_scaled))
    test_data = test_data.contiguous()
    test_position = torch.FloatTensor(np.array(test_position))
    test_position = test_position.contiguous()
    y = Variable(test_position) 
    x = test_data.view(batch_size,input_size)
    y = y.view(batch_size,output_size)

    model = RNN(input_size, hidden_size, num_layers,len(output))
    print model
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss = torch.nn.MSELoss(size_average=False)

    #begin to train 
    for epoch in range(number_of_iteration):
        # Pytorch accumulates gradients. We need to clear them out before each instance
        optimizer.zero_grad()
        model.hidden = model.init_hidden()
        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        out = model(x.unsqueeze(1).float())

        #TINC EL PROBLEMA AQUI AMB EL TRAINING DATA
      
        err = loss(out, y)
        err.backward()
        optimizer.step()
   
    print('-------done LSTM')
    torch.save(model, path_to_store_weight_file)



def test_LSTM(path_to_load_weight_file=None, path_to_test_data=None, path_to_output=None):

    model = torch.load(path_to_load_weight_file)

    def flatten(list_):
        for el in list_:
            if hasattr(el, "__iter__") and not isinstance(el, basestring):  
                for sub in flatten(el):  
                    yield sub  
            else:  
                yield el


    list = []
    output = pickle.load(open(path_to_test_data, "rb" ))
    for j in range(len(output)):
        test_data=[] 
        for i in range(len(output[j])):
            for obj in vars(output[j][i])["players"]:
                for obj in vars(output[j][i])["players"]:
                    test_data.append(obj.get_info())
            test_data.append(vars(output[j][i])["quarter"])
            test_data.append(vars(output[j][i])["game_clock"])
            test_data.append(vars(output[j][i])["ball"].get_info())
            test_data.append(vars(output[j][i])["shot_clock"])
        list_1_scaled = [x for x in flatten(test_data)]
        list.append(list_1_scaled)         
    
    data = pd.DataFrame(list)
    data_1 = data.copy()
    data_1 = data_1.values
    from sklearn.preprocessing import Imputer
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    imputer = Imputer(strategy="mean")
    imputer.fit(data_1)
    data_1 = imputer.transform(data_1)
    data_1_scaled = scaler.fit_transform(data_1)
    test_data = torch.DoubleTensor(np.array(data_1_scaled))
    x = test_data.contiguous()
    y_pred= model(x.unsqueeze(1).float())
    y_pred = y_pred.numpy()
    # y_pred = y_pred.detach()
    pickle.dump(y_pred, open(path_to_output , "wb" ))


def check_distance(path_to_output):
    result_list= pickle.load(open(path_to_output , "rb"))
    gt_list = pickle.load(open("gt_data.p" , "rb"))
    result_list = np.asarray(result_list) 
    gt_list = np.asarray(gt_list)
    
    error_distance = np.sum((result_list-gt_list))
    print "your error distance is ",error_distance


if sys.argv[1] == "train":
    train_LSTM(sys.argv[2], int(sys.argv[3]))
elif sys.argv[1] == "test":
    test_LSTM(sys.argv[2], sys.argv[3], sys.argv[4])  
elif sys.argv[1] == "check_distance":
    check_distance(sys.argv[2])
elif sys.argv[1] == "prova":
    generate_sample_test()
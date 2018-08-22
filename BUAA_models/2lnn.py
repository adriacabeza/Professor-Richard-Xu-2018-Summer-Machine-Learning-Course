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
    N, D_in, H, D_out = 151, 2436, 100,2
    def flatten(list_):
            for el in list_:
                if hasattr(el, "__iter__") and not isinstance(el, basestring):  
                    for sub in flatten(el):  
                        yield sub  
                else:  
                    yield el
    output, test_position= generate_sample_test() 
    # test_position= pickle.load(open("gt_data.p" , "rb" ))
    # output = pickle.load(open("test_data.p" , "rb" ))
    list = []
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
    test_data = test_data.contiguous()
    test_position = torch.FloatTensor(np.array(test_position))
    test_position = test_position.contiguous()
    y = test_position 
    x = test_data.view(N,D_in)
    x = Variable(x)
    y = y.view(N,D_out)
    y = Variable(y)
    model = TwoLayerNet(D_in, H, D_out)


    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss(size_average=False)

    for t in range(number_of_iteration):
        y_pred = model(x.float())
        loss = loss_fn(y_pred, y.float())
        # zero gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    
    print('-------done')
    torch.save(model, path_to_store_weight_file)

   
def test(path_to_load_weight_file=None, path_to_test_data=None, path_to_output=None):

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
    x = Variable(x)
    y_pred= model(x.float())
    y_pred = y_pred.data.numpy()
    pickle.dump(y_pred, open(path_to_output , "wb" ))



def check_distance(path_to_output):
    result_list= pickle.load(open(path_to_output , "rb"))
    gt_list = pickle.load(open("gt_data.p" , "rb"))
    result_list = np.asarray(result_list) 
    gt_list = np.asarray(gt_list)
    
    error_distance = np.sum((result_list-gt_list))
    print "your error distance is ",error_distance
            
if sys.argv[1] == "train":
    train(sys.argv[2], int(sys.argv[3]))
elif sys.argv[1] == "test":
    test(sys.argv[2], sys.argv[3], sys.argv[4])  
  
elif sys.argv[1] == "check_distance":
    check_distance(sys.argv[2])
elif sys.argv[1] == "prova":
    generate_sample_test()
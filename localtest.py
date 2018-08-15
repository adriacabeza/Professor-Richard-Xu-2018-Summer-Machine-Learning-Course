import subprocess
import os
import numpy as np
import pickle
# server specify
weight_path = "weight.w"
number_of_iteration = 350
path_to_test_data = "test_data.p"
path_to_output = "output.p"
gt_test = "gt_data.p"
#user specify
path_to_load_weight_file = "weight.w" 

cmd1 = "python main.py train_LSTM "+ weight_path + " " + str(number_of_iteration)
cmd2 = "python main.py test_LSTM "+path_to_load_weight_file+" "+path_to_test_data+" "+path_to_output


subprocess.call(cmd1, shell=True)
if os.path.isfile(weight_path) is not True:
    print "can not find weight file"


subprocess.call(cmd2, shell=True)
if os.path.isfile(path_to_output) is not True:
    print "can not find output file"
else:
    result_list= pickle.load(open(path_to_output , "rb"))
    gt_list = pickle.load(open(gt_test , "rb"))
    result_list = np.asarray(result_list) #falta el detach
    gt_list = np.asarray(gt_list)
    
    error_distance = np.sum((result_list-gt_list))
    print "your error distance is ",error_distance


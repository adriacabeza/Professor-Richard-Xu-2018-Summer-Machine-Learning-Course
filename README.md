# Professor Richard Xu's 2018 Summer Machine Learning Course Competition
This competition is used in conjunction with Professor Richard Xu's summer course practice. Hoping that students can apply what they have learned in class to real-world problems. Deepen your understanding of the course content.

All the data in this competition comes from real NBA games(https://github.com/linouk23/NBA-Player-Movements). The position of players on the field can vary with different tactics and different time.

##### The competition requirement is to predict the position of a particular player after 2 seconds based on the historical position information of all players in the course.

![](spurs.gif)

It has been made using:

- Pytorch 0.4
- TwoLayerNeural Net(waiting to be upgraded to a RNN/LSTM)
- This data :arrow_right:https://github.com/linouk23/NBA-Player-Movements


````python
weight_path = "weight.w"
number_of_iteration = 1
path_to_test_data = "test_data.p"
path_to_output = "output.p"
gt_test = "gt_data.p"
#user specify
path_to_load_weight_file = "weight.w" 

cmd1 = "python main.py train "+ weight_path + " " + str(number_of_iteration)
cmd2 = "python main.py test "+path_to_load_weight_file+" "+path_to_test_data+" "+path_to_output


subprocess.call(cmd1, shell=True)
if os.path.isfile(weight_path) is not True:
    print "can not find weight file"

````

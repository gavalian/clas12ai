#./ml-cli.py train -t $PROJECT/data/02-04-2020/005038_0100/training_data_node_6.txt -e $PROJECT/data/02-04-2020/005038_0110/training_data_node_6_test.txt -f 6 --model-type et -m dc_tracks_neg.nnet
./ml-cli.py train -t $PROJECT/data/10-07-2019/training_data_node_36.txt -e $PROJECT/data/10-07-2019/training_data_node_36_test.txt -f 36 --model-type cnn -m dc_tracks_neg.nnet
#./ml-cli.py train -t $PROJECT/data/10-07-2019/training_data_node_6.txt -e $PROJECT/data/10-07-2019/training_data_node_6_test.txt -f 6 --model-type mlp -m dc_tracks_neg.nnet

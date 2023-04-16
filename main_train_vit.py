import numpy as np
import pandas as pd
from vit_model import train as vit_train
from vit_model import predict as vit_predict
from utils import read_data
import os

val = False
TRAIN = True
PREDICT = True
PRE_train = True
n_class = 2
ep =3
device = 'cuda:0'
if __name__ == '__main__':
    
    data_path = './data/data.txt'
        
    if os.path.exists('./res_dir/label/data' ) is False:
        os.makedirs('./res_dir/label/data' )
    if os.path.exists('./res_dir/train_res/data' ) is False:
        os.makedirs('./res_dir/train_res/data')
    if os.path.exists('./res_dir/weights/data' ) is False:
        os.makedirs('./res_dir/weights/data' )
    if os.path.exists('./res_dir/best_res') is False:
        os.makedirs('./res_dir/best_res')


    train, train_label, test, test_label = read_data(data_path, 0.2, n_class=n_class, seed=2)
    if TRAIN:

            vit_train(train, train_label, classes=n_class, device=device, val=False,  epochs=ep, init=PRE_train)

            print('train over')
    if PREDICT:

            score4 = vit_predict(test, test_label, num_class=n_class)

            pd.DataFrame(np.array(test_label)).to_csv('./res_dir/label.csv', header=None, index=None)
            print('predict over')

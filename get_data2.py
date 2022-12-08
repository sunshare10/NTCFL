import pandas as pd
import numpy as np
import glob, os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from sklearn import model_selection 
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import accuracy_score
# import matplotlib.pyplot as plt
#本地
import utils2 as utils
import dataanalyzer as da

# visulaize the important characteristics of the dataset
def get_data2(num_headers = 16,pck_len=72,t=0):
    seed = 1
     #临时更改
    if(t==0):
        dirs=["e:/1/client1/07050021/split/h5/16/","e:/1/client1/07100002/split/h5/16/",\
            "e:/1/client2/07101940/split/h5/16/","e:/1/client3/07110005/split/h5/16/"]
    elif pck_len==1500:
        dirs=['e:/1/LAST/LAST_16_1500/','e:/1/last/version819/split/h5_16_1500/']
        # dirs=['e:/1/LAST/LAST_16_1500ipniming/','e:/1/last/version819/split/h5_16_1500ipniming/']
    elif pck_len==72:
        dirs=['D:/Code/data/datasets/LinuxChrome/16/',
              'D:/Code/data/datasets/WindowsAndreas/16/',
              'D:/Code/data/datasets/WindowsChrome/16/',
              'D:/Code/data/datasets/WindowsFirefox/16/',
              'D:/Code/data/datasets/WindowsSalik/16/']
        # dirs=['e:/1/LAST/LAST_16_60/','e:/1/last/version819/split/h5_16_60/']
    elif num_headers==16 and pck_len==60:
        # dirs=['D:/iscxvpn/NonVPN/h5_16_60/']
        dirs=['e:/1/LAST/LAST_16_60/','e:/1/last/version819/split/h5_16_60/']
    elif num_headers==60 and pck_len==60:
        # dirs=['D:/iscxvpn/NonVPN/h5_16_602bu/']
        dirs=['e:/1/LAST/LAST_16_602bu/','e:/1/last/version819/split/h5_16_602bu/']
    # step 1: get the data
    dataframes = []
    # num_examples = 0
    for dir in dirs:
        for fullname in glob.iglob(dir + '*.h5'):
            filename = os.path.basename(fullname)
            print(dir)
            df = utils.load_h5(dir, filename)
            dataframes.append(df)
            # num_examples = len(df.values)
    # create one large dataframe
    data = pd.concat(dataframes)
    data.sample(frac=1, random_state=seed).reset_index(drop=True)
    # num_rows = data.shape[0]

    # step 3: get features (x) and scale the features
    # get x and convert it to numpy array
    # x = da.getbytes(data, 1460)
    standard_scaler = StandardScaler()
    if pck_len==72:
        x = da.getbytes(data, num_headers*pck_len)  #临时更改
        # a=len(x)
        # ze=np.zeros((a,4032))  #alll
       
        # x = np.hstack((x,ze))
        
    elif pck_len==1500:
        x = da.getbytes(data, 24000)  #16X10*150
        # x2=np.zeros((len(x),25600))
        # for i, v in enumerate(x):
        #     v=v.reshape(-1,150)
            
        #     zjjz=v[1:,0:10]
        #     ze=np.zeros((1,10))
        #     zjjz=np.vstack((zjjz,ze))
        #     v=np.hstack((v,zjjz))
        #     x2[i]=v.reshape(1,-1)
        # x=x2
        
    elif num_headers==60 and pck_len==60:
        x=da.getbytes(data, 3600)
    elif num_headers==16 and pck_len==60:    
        x=da.getbytes(data, 960)
        a=len(x)
        ze=np.zeros((a,pck_len*(pck_len-num_headers)))  #alll
        x = np.hstack((x,ze))
    x_std = standard_scaler.fit_transform(x)
    # step 4: get class labels y and then encode it into number
    # get class label data
    y = data['label'].values
    # encode the class label
    # class_labels = np.unique(y)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    # classes_path='get_label.txt'
    # names,indexs=get_info(classes_path)
    # for ii, ylabel in y:
    #     for i,name in names:
    #          if ylabel==name:
    #              y[ii]=indexs[i]

    # step 5: split the data into training set and test set
    test_percentage = 0.2
    # x_tests = []
    # y_tests = []
    x_train, x_test, y_train, y_test = train_test_split(x_std, y, test_size=test_percentage, random_state=seed)
    return x_train, y_train,x_test, y_test

def get_info(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    names = []
    indexs = []
    for data in class_names:
        name,index = data.split(' ')
        names.append(name)
        indexs.append(int(index))
        
    return names,indexs
if __name__ == '__main__':
    NUM_HEADERS=16
    PCK_LEN=1500
    # NUM_HEADERS=60
    # PCK_LEN=60
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t=1
    
    x_train, y_train,x_test, y_test=get_data2(num_headers = NUM_HEADERS,pck_len=PCK_LEN,t=t)
# def remove():
#     dirs=["D:/1/client1/07050021/split/h5/16/","D:/1/client1/07100002/split/h5/16/",\
#             "D:/1/client2/07101940/split/h5/16/","D:/1/client3/07110005/split/h5/16/"]
#     for dir in dirs:
#         for fullname in glob.iglob(dir + '*.h5'):
#             os.remove(fullname)
#             print(fullname)
# remove()
# get_data2(t=1)
#导入ISCX数据集
# from pathlib import Path
# from petastorm import make_reader
# from petastorm.pytorch import DataLoader
# def train_dataloader(data_path):
#         reader = make_reader(Path(data_path).absolute().as_uri(), reader_pool_type='process', workers_count=12,
#                              pyarrow_serialize=True, shuffle_row_groups=True, shuffle_row_drop_partitions=2,
#                              num_epochs=15)
#         dataloader = DataLoader(reader, batch_size=16, shuffling_queue_capacity=4096)
#         return dataloader
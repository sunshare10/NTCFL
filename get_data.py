import pandas as pd
import numpy as np
import glob, os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import math
# from sklearn import model_selection 
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import accuracy_score
# import matplotlib.pyplot as plt
#本地
import utils2 as utils
import dataanalyzer as da

# visulaize the important characteristics of the dataset
def get_data2(num_headers = 16,pck_len=1500,pngchang=160,t=1,niming_flag=1,test=False):
    seed = 1
     #临时更改
    if(t==0):
        dirs=["e:/1/client1/07050021/split/h5/16/","e:/1/client1/07100002/split/h5/16/",\
            "e:/1/client2/07101940/split/h5/16/","e:/1/client3/07110005/split/h5/16/"]
    elif niming_flag==0:
        # dirs=['e:/1/LAST/LAST_'+str(num_headers)+'_'+str(pck_len)+'/','e:/1/last/version819/split/h5_'+str(num_headers)+'_'+str(pck_len)+'/']
        dirs=['/hy-tmp/last_'+str(num_headers)+'_'+str(pck_len)+'/','/hy-tmp/h5_'+str(num_headers)+'_'+str(pck_len)+'/']
    elif niming_flag==1:
        dirs=['e:/1/LAST/LAST_'+str(num_headers)+'_'+str(pck_len)+'ipniming3/','e:/1/last/version819/split/h5_'+str(num_headers)+'_'+str(pck_len)+'ipniming/']
        # dirs=['e:/1/last/version819/split/h5_'+str(num_headers)+'_'+str(pck_len)+'ipniming/']
        # dirs=['e:/1/client1/07050021/split/h5_'+str(num_headers)+'_'+str(pck_len)+'_8ipniming/',
        #       'e:/1/client1/07100002/split/h5_'+str(num_headers)+'_'+str(pck_len)+'_8ipniming/',
        #       'e:/1/client2/07101940/split/h5_'+str(num_headers)+'_'+str(pck_len)+'_8ipniming/',
        #       'e:/1/client3/07110005/split/h5_'+str(num_headers)+'_'+str(pck_len)+'_8ipniming/',
        #       'e:/1/last/version819/split/h5_'+str(num_headers)+'_'+str(pck_len)+'_8ipniming/']
        # dirs=['e:/1/last/16niming/h5_16_1500ipniming/','e:/1/last/16niming/last_16_1500ipniming/']
        # dirs=['/hy-tmp/last_'+str(num_headers)+'_'+str(pck_len)+'ipniming/','/hy-tmp/h5_'+str(num_headers)+'_'+str(pck_len)+'ipniming/']
    elif niming_flag==2:
        # dirs=['e:/1/LAST/LAST_'+str(num_headers)+'_'+str(pck_len)+'ipniming_daifangxiang/','e:/1/last/version819/split/h5_'+str(num_headers)+'_'+str(pck_len)+'ipniming_daifangxiang/']
        dirs=['/hy-tmp/last_'+str(num_headers)+'_'+str(pck_len)+'ipniming_daifangxiang/','/hy-tmp/h5_'+str(num_headers)+'_'+str(pck_len)+'ipniming_daifangxiang/']
    elif niming_flag==3:
        # dirs=['e:/1/LAST/LAST_'+str(num_headers)+'_'+str(pck_len)+'ipniming_255/','e:/1/last/version819/split/h5_'+str(num_headers)+'_'+str(pck_len)+'ipniming_255/']
        dirs=['/hy-tmp/last_'+str(num_headers)+'_'+str(pck_len)+'ipniming_255/','/hy-tmp/h5_'+str(num_headers)+'_'+str(pck_len)+'ipniming_255/']
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
    # step 3: get features (x) and scale the features
    # get x and convert it to numpy array
    standard_scaler = StandardScaler()
    if num_headers==16 and pck_len==72:
        x = da.getbytes(data, num_headers*pck_len)  #临时更改
        a=len(x)
        ze=np.zeros((a,60*60-num_headers*pck_len))  #alll
        x = np.hstack((x,ze))
    elif num_headers==16 and pck_len==54:    
        x=da.getbytes(data, num_headers*pck_len)
        a=len(x)
        ze=np.zeros((a,(60*60-num_headers*pck_len)))
        x = np.hstack((x,ze))
    elif t==4:
        x = da.getbytes(data, num_headers*pck_len)  #16X10*150
        x2=np.zeros((len(x),pngchang*pngchang))
        chang1=math.ceil(1500/(pngchang/num_headers)) #一行中不重复的数据长度
        for i, v in enumerate(x):
            v=v.reshape(-1,pck_len)
            for z in range(int(pngchang/num_headers-1)):
                x2[i][(num_headers*pngchang*z):(num_headers*pngchang*z+num_headers*pngchang)]=v[:,(chang1*z):(chang1*z+pngchang)].reshape(1,-1)
            x2[i][(num_headers*pngchang*(z+1)):num_headers*((pngchang*(z+1))+pck_len-(z+1)*chang1)]=(v[0:num_headers,((z+1)*chang1):pck_len]).reshape(1,-1)
        x=x2
    elif num_headers==16 and pck_len==1500:
        if t==1:
            x = da.getbytes(data, 24000)  #16X10*150
            x2=np.zeros((len(x),25600))
            for i, v in enumerate(x):
                v=v.reshape(-1,1500)
                for z in range(9):
                    x2[i][(2560*z):(2560*z+2560)]=v[:,(150*z):(150*z+160)].reshape(1,-1)
                x2[i][23040:25600]=np.hstack((v[:,1350:1500],v[:,0:10])).reshape(1,-1)
            x=x2
        elif t==10:
            x = da.getbytes(data, 24000)
            a=len(x)
            ze=np.zeros((a,1600))  #alll
            x = np.hstack((x,ze))
    elif (num_headers==4 or num_headers==8 or num_headers==12) and pck_len==1500:
        if t==1:
            x = da.getbytes(data, num_headers*pck_len)  #16X10*150
            x2=np.zeros((len(x),25600))
            for i, v in enumerate(x):
                v=v.reshape(-1,pck_len)
                for z in range(9):
                    x2[i][(num_headers*160*z):(num_headers*160*(z+1))]=v[:num_headers,(150*z):(150*z+160)].reshape(1,-1)
                x2[i][(num_headers*160*(z+1)):(num_headers*160*(z+1))+150*num_headers]=(v[0:num_headers,1350:1500]).reshape(1,-1)
            x=x2
    elif num_headers==24 and pck_len==1500:
        
        x = da.getbytes(data, num_headers*pck_len)  #16X10*150
        x2=np.zeros((len(x),224*224))
        # print(len(x))
        for i, v in enumerate(x):
            v=v.reshape(-1,1500)
            for z in range(7):
                x2[i][(5376*z):(5376*z+5376)]=v[:,(200*z):(200*z+224)].reshape(1,-1)
            x2[i][37632:40032]=v[:,1400:1500].reshape(1,-1) 
        x=x2
    elif num_headers==20 and pck_len==1500:
        
        x = da.getbytes(data, num_headers*pck_len)  #16X10*150
        x2=np.zeros((len(x),224*224))
        # print(len(x))
        for i, v in enumerate(x):
            v=v.reshape(-1,1500)
            for z in range(7):
                x2[i][(4480*z):(4480*z+4480)]=v[:,(200*z):(200*z+224)].reshape(1,-1)
            x2[i][31360:33360]=v[:,1400:1500].reshape(1,-1) 
        x=x2
    x_std = standard_scaler.fit_transform(x)
    # step 4: get class labels y and then encode it into number
    # get class label data
    y = data['label'].values
    # encode the class label
    class_names = np.unique(y)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    indexs=np.unique(y)
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
    x_train, x_test, y_train, y_test = train_test_split(x_std, y, test_size=test_percentage, random_state=False)
    if test:
        return x_test, y_test,class_names,indexs
    else:
        return x_train, y_train,x_test, y_test

def get_data3(num_headers = 16,pck_len=1500,pngchang=160,t=1,niming_flag=1,test=False,dirs=[]):
    seed = 1
    quyiban=False
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
    if quyiban:
        data =data.sort_values(by=['time'])
        yiban=len(data)//2
        data=data.iloc[:yiban]
    data.sample(frac=1, random_state=seed).reset_index(drop=True)
    # step 3: get features (x) and scale the features
    # get x and convert it to numpy array
    standard_scaler = StandardScaler()
    if num_headers==16 and pck_len==72:
        x = da.getbytes(data, num_headers*pck_len)  #临时更改
        a=len(x)
        ze=np.zeros((a,60*60-num_headers*pck_len))  #alll
        x = np.hstack((x,ze))
    elif num_headers==16 and pck_len==54:    
        x=da.getbytes(data, num_headers*pck_len)
        a=len(x)
        ze=np.zeros((a,(60*60-num_headers*pck_len)))
        x = np.hstack((x,ze))
    elif t==1:
        x = da.getbytes(data, num_headers*pck_len)  #16X10*150
        x2=np.zeros((len(x),pngchang*pngchang))
        chang1=math.ceil(1500/(pngchang/num_headers)) #一行中不重复的数据长度150
        for i, v in enumerate(x):
            v=v.reshape(-1,pck_len)
            for z in range(int(pngchang/num_headers-1)):
                x2[i][(num_headers*pngchang*z):(num_headers*pngchang*z+num_headers*pngchang)]=v[:,(chang1*z):(chang1*z+pngchang)].reshape(1,-1)
            # x2[i][(num_headers*pngchang*(z+1)):num_headers*((pngchang*(z+1))+pck_len-(z+1)*chang1)]=(v[0:num_headers,((z+1)*chang1):pck_len]).reshape(1,-1)
            buchong=int(pngchang*pngchang/num_headers-pck_len-(pngchang-chang1)*(z+1))
            # print(buchong)
            x2[i][(num_headers*pngchang*(z+1)):(pngchang*pngchang)]=np.hstack((v[:,((z+1)*chang1):pck_len],v[:,0:buchong])).reshape(1,-1)
        x=x2
    # elif num_headers==16 and pck_len==1500:
    #     if t==1:
    #         x = da.getbytes(data, 24000)  #16X10*150
    #         x2=np.zeros((len(x),25600))
    #         for i, v in enumerate(x):
    #             v=v.reshape(-1,1500)
    #             for z in range(9):
    #                 x2[i][(2560*z):(2560*z+2560)]=v[:,(150*z):(150*z+160)].reshape(1,-1)
    #             x2[i][23040:25600]=np.hstack((v[:,1350:1500],v[:,0:10])).reshape(1,-1)
    #         x=x2
    #     elif t==10:
    #         x = da.getbytes(data, 24000)
    #         a=len(x)
    #         ze=np.zeros((a,1600))  #alll
    #         x = np.hstack((x,ze))
    elif (num_headers==4 or num_headers==8 or num_headers==12) and pck_len==1500:
        if t==1:
            x = da.getbytes(data, num_headers*pck_len)  #16X10*150
            x2=np.zeros((len(x),25600))
            for i, v in enumerate(x):
                v=v.reshape(-1,pck_len)
                for z in range(9):
                    x2[i][(num_headers*160*z):(num_headers*160*(z+1))]=v[:num_headers,(150*z):(150*z+160)].reshape(1,-1)
                x2[i][(num_headers*160*(z+1)):(num_headers*160*(z+1))+150*num_headers]=(v[0:num_headers,1350:1500]).reshape(1,-1)
            x=x2
    elif num_headers==24 and pck_len==1500:
        
        x = da.getbytes(data, num_headers*pck_len)  #16X10*150
        x2=np.zeros((len(x),224*224))
        # print(len(x))
        for i, v in enumerate(x):
            v=v.reshape(-1,1500)
            for z in range(7):
                x2[i][(5376*z):(5376*z+5376)]=v[:,(200*z):(200*z+224)].reshape(1,-1)
            x2[i][37632:40032]=v[:,1400:1500].reshape(1,-1) 
        x=x2
    elif num_headers==20 and pck_len==1500:
        
        x = da.getbytes(data, num_headers*pck_len)  #16X10*150
        x2=np.zeros((len(x),224*224))
        # print(len(x))
        for i, v in enumerate(x):
            v=v.reshape(-1,1500)
            for z in range(7):
                x2[i][(4480*z):(4480*z+4480)]=v[:,(200*z):(200*z+224)].reshape(1,-1)
            x2[i][31360:33360]=v[:,1400:1500].reshape(1,-1) 
        x=x2
    x_std = standard_scaler.fit_transform(x)

    y = data['label'].values

    return x_std, y

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
    PNG_chan=160  #灰度图尺寸
    niming_flag=1  #默认为1；0表示不匿名，1表示匿名不带方向，2表示匿名且标识方向,3表示用255进行匿名，不带方向
    # device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t=1   #对于161500,10表示老构图，1为新构图,默认为1
    dirs1={}
    deleteYY=False
    # dirs1[0]=['e:/1/federated_miss/client0/h5_'+str(NUM_HEADERS)+'_'+str(PCK_LEN)+'_8ipniming/',
    #           'e:/1/federated_miss/client0/h5_'+str(NUM_HEADERS)+'_'+str(PCK_LEN)+'_8ipniming2/']
    # dirs1[1]=['e:/1/federated_miss/client1/h5_'+str(NUM_HEADERS)+'_'+str(PCK_LEN)+'_8ipniming/']
    # dirs1[2]=['e:/1/federated_miss/client2/h5_'+str(NUM_HEADERS)+'_'+str(PCK_LEN)+'_8ipniming/']
    # dirs1[3]=['e:/1/federated_miss/client3/h5_'+str(NUM_HEADERS)+'_'+str(PCK_LEN)+'ipniming/']
    # dirs1[0]=['e:/1/client1/07050021/split/h5_'+str(NUM_HEADERS)+'_'+str(PCK_LEN)+'_8ipniming/',
    #           'e:/1/client1/07100002/split/h5_'+str(NUM_HEADERS)+'_'+str(PCK_LEN)+'_8ipniming/']
    # dirs1[1]=['e:/1/client2/07101940/split/h5_'+str(NUM_HEADERS)+'_'+str(PCK_LEN)+'_8ipniming/']
    # dirs1[2]=['e:/1/client3/07110005/split/h5_'+str(NUM_HEADERS)+'_'+str(PCK_LEN)+'_8ipniming/']
    # dirs1[3]=['e:/1/last/version819/split/h5_'+str(NUM_HEADERS)+'_'+str(PCK_LEN)+'_8ipniming/']
    # dirs1[0]=['e:/1/client1/07050021/split/h5_'+str(NUM_HEADERS)+'_'+str(PCK_LEN)+'noniming/',
    #           'e:/1/client1/07100002/split/h5_'+str(NUM_HEADERS)+'_'+str(PCK_LEN)+'noniming/']
    # dirs1[1]=['e:/1/client2/07101940/split/h5_'+str(NUM_HEADERS)+'_'+str(PCK_LEN)+'noniming/']
    # dirs1[2]=['e:/1/client3/07110005/split/h5_'+str(NUM_HEADERS)+'_'+str(PCK_LEN)+'noniming/']
    # dirs1[3]=['e:/1/last/version819/split/h5_'+str(NUM_HEADERS)+'_'+str(PCK_LEN)+'noniming/']
    dirs1[0]=['e:/1/last/last_'+str(NUM_HEADERS)+'_'+str(PCK_LEN)+'ipniming3/']
    dirs1[1]=['e:/1/last/version819/split/h5_'+str(NUM_HEADERS)+'_'+str(PCK_LEN)+'_8ipniming/']
    
    
    x_train, y_train,x_test, y_test,train_dataset,test_dataset,train_loader,val_loader ={},{},{},{},{},{},{},{}
    x_std,y={},{}
    for idx in range(len(dirs1)):
        x_std[idx], y[idx]=get_data3(num_headers = NUM_HEADERS,pck_len=PCK_LEN,pngchang=PNG_chan,
                                              t=t,niming_flag=niming_flag,dirs=dirs1[idx])
        if idx==0:
            y_all=y[idx]
        else:
            y_all= np.concatenate((y_all, y[idx]), axis=0)
    classnames=np.unique(y_all)
    classes_names=classnames
    indexs=np.arange(12)
    markdict={}
    i=0
    for classname in classnames:
        markdict[classname]=int(i)
        i=i+1
    print(markdict)
    # import pickle
    with open('markdict.txt','w') as file:
        file.write(str(markdict))
    for idx in range(len(dirs1)):
    #转换numpy为tensor
        for i in range(len(y[idx])):
            y[idx][i]=markdict[y[idx][i]]
        y[idx]=y[idx].astype(int)
        print(x_std[idx].shape)
        print(x_std[idx].shape[0])
        if deleteYY:
            deletelist=[]
            for c in range(x_std[idx].shape[0]):
                if y[idx][c]==11:
                    deletelist.append(c)
            print(len(deletelist))
            y[idx]=np.delete(y[idx],deletelist,axis=0)
            x_std[idx]=np.delete(x_std[idx],deletelist,axis=0)
        
        x_train[idx], x_test[idx], y_train[idx], y_test[idx] = train_test_split(x_std[idx], y[idx], test_size=0.2, random_state=True)
        np.savez('client'+'_'+str(idx)+'_'+str(NUM_HEADERS)+'_'+str(PCK_LEN)+'_'+str(PNG_chan)+'_niming'+str(niming_flag)+'_goutu'+str(t), x_train=x_train[idx], y_train=y_train[idx], x_test=x_test[idx],y_test=y_test[idx])  # c_array是数组c的命名
        data = np.load('client'+'_'+str(idx)+'_'+str(NUM_HEADERS)+'_'+str(PCK_LEN)+'_'+str(PNG_chan)+'_niming'+str(niming_flag)+'_goutu'+str(t)+'.npz')
        print('x_train : ', data['x_train'].shape)
        print('y_train : ', data['y_train'].shape)
        print('x_test : ', data['x_test'].shape)
        print('y_test : ', data['y_test'].shape)
        print(1)


import imp
import os
import sys
sys.path.insert(0,os.getcwd())
import argparse
import numpy as np
import copy
from numpy import mean
from tqdm import tqdm
from terminaltables import AsciiTable
from get_data import get_data2
from torch.utils.data import Dataset,TensorDataset
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
import time
import csv

from utils.dataloader import Mydataset, collate
from utils.train_utils import get_info, file2dict
from models.build import BuildNet
from core.evaluations import evaluate
from utils.inference import init_model

def get_metrics_output(eval_results, metrics_output,classes_names, indexs):
    f = open(metrics_output,'a', newline='')
    writer = csv.writer(f)
    
    """
    输出并保存Accuracy、Precision、Recall、F1 Score、Confusion matrix结果
    """
    p_r_f1 = [['Classes','Precision','Recall','F1 Score']]
    for i in range(len(classes_names)):
        data = []
        data.append(classes_names[i])
        data.append('{:.2f}'.format(eval_results.get('precision')[indexs[i]]))
        data.append('{:.2f}'.format(eval_results.get('recall')[indexs[i]]))
        data.append('{:.2f}'.format(eval_results.get('f1_score')[indexs[i]]))
        p_r_f1.append(data)
    TITLE = 'Classes Results'
    TABLE_DATA_1 = tuple(p_r_f1)
    table_instance = AsciiTable(TABLE_DATA_1,TITLE)
    #table_instance.justify_columns[2] = 'right'
    print()
    print(table_instance.table)
    writer.writerows(TABLE_DATA_1)
    writer.writerow([])
    print()

    TITLE = 'Total Results'    
    TABLE_DATA_2 = (
    ('Top-1 Acc', 'Top-5 Acc', 'Mean Precision', 'Mean Recall', 'Mean F1 Score'),
    ('{:.2f}'.format(eval_results.get('accuracy_top-1',0.0)), '{:.2f}'.format(eval_results.get('accuracy_top-5',100.0)), '{:.2f}'.format(mean(eval_results.get('precision',0.0))),'{:.2f}'.format(mean(eval_results.get('recall',0.0))),'{:.2f}'.format(mean(eval_results.get('f1_score',0.0)))),
    )
    table_instance = AsciiTable(TABLE_DATA_2,TITLE)
    #table_instance.justify_columns[2] = 'right'
    print(table_instance.table)
    writer.writerows(TABLE_DATA_2)
    writer.writerow([])
    print()


    writer_list     = []
    writer_list.append([' '] + [str(c) for c in classes_names])
    # for i in range(len(eval_results.get('confusion'))):
    #     writer_list.append([classes_names[i]] + [str(x) for x in eval_results.get('confusion')[i]])
    for i in range(len(classes_names)):
        writer_list.append([classes_names[i]] + [str(x) for x in eval_results.get('confusion')[i]])
    
    TITLE = 'Confusion Matrix'
    TABLE_DATA_3 = tuple(writer_list)
    table_instance = AsciiTable(TABLE_DATA_3,TITLE)
    print(table_instance.table)
    writer.writerows(TABLE_DATA_3)
    print()

def get_prediction_output(preds,targets,image_paths,classes_names,indexs,prediction_output):
    nums = len(targets)
    f = open(prediction_output,'a', newline='')
    writer = csv.writer(f)
    
    results = [['File', 'Pre_label', 'True_label', 'Success']]
    results[0].extend(classes_names)
    
    for i in range(nums):
        temp = [image_paths[i]]
        pred_label = classes_names[indexs[torch.argmax(preds[i]).item()]]
        true_label = classes_names[indexs[targets[i].item()]]
        success = True if pred_label == true_label else False
        class_score = preds[i].tolist()
        temp.extend([pred_label,true_label,success])
        temp.extend(class_score)
        results.append(temp)
        
    writer.writerows(results)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--device', help='device used for training. (Deprecated)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    args = parser.parse_args()
    return args

def main(): 
    args = parse_args()
    model_cfg,train_pipeline,val_pipeline,data_cfg,lr_config,optimizer_cfg = file2dict(args.config)

    """
    创建评估文件夹、metrics文件、混淆矩阵文件
    """
    NUM_HEADERS=16
    PCK_LEN=1500
    PNG_chan=160  #灰度图尺寸
    niming_flag=1  #默认为1；0表示不匿名，1表示匿名不带方向，2表示匿名且标识方向,3表示用255进行匿名，不带方向
    device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t=10   #10对于161500表示老构图，1为新构图,默认为1
    dirname = str(NUM_HEADERS)+'_'+str(PCK_LEN)+'_'+str(PNG_chan)+'_niming'+str(niming_flag)+'_goutu'+str(t)+time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    save_dir = os.path.join('eval_results',model_cfg.get('backbone').get('type'),dirname)
    os.makedirs(save_dir)
    metrics_output = os.path.join(save_dir,'metrics_output.csv')
    prediction_output = os.path.join(save_dir,'prediction_results.csv')
    
    """
    获取类别名以及对应索引、获取标注文件
    """
    # classes_map = 'datas/annotations.txt' 
    # test_annotations    = 'datas/test.txt'
    # classes_names, indexs = get_info(classes_map)
    # with open(test_annotations, encoding='utf-8') as f:
    #     test_datas   = f.readlines()
        
    """
    生成模型、加载权重
    """
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = BuildNet(model_cfg)
    if device != torch.device('cpu'):
        model = DataParallel(model,device_ids=[args.gpu_id])
    model = init_model(model, data_cfg, device=device, mode='eval')
    
    """
    制作测试集并喂入Dataloader
    """
    # val_pipeline = copy.deepcopy(train_pipeline)
    # test_dataset = Mydataset(test_datas, val_pipeline)
    # test_loader = DataLoader(test_dataset, shuffle=True, batch_size=data_cfg.get('batch_size'), num_workers=data_cfg.get('num_workers'), pin_memory=True, collate_fn=collate)
        #zxf
    x_test, y_test,classes_names, indexs=get_data2(num_headers = NUM_HEADERS,pck_len=PCK_LEN,pngchang=PNG_chan,t=t,niming_flag=niming_flag,test=True)
    #转换numpy为tensor
    if NUM_HEADERS==16 and PCK_LEN==54:
        # x_train=x_train.reshape(-1,1,16,54)
        x_test=x_test.reshape(-1,1,60,60)
    elif NUM_HEADERS==16 and PCK_LEN==72:
        # x_train=x_train.reshape(-1,1,72,72)
        x_test=x_test.reshape(-1,1,60,60)
    elif  PCK_LEN==60:
        # x_train=x_train.reshape(-1,1,60,60)
        x_test=x_test.reshape(-1,1,60,60)
    elif PCK_LEN==1500:
        x_test=x_test.reshape(-1,1,PNG_chan,PNG_chan)

    x_test=torch.from_numpy(x_test)
    y_test=torch.from_numpy(y_test)
    #修改数据格式
    
    x_test=x_test.type(torch.FloatTensor)
    y_test=y_test.type(torch.LongTensor)
    #封装数据
    # print(x_train[0])
    # train_dataset=TensorDataset(x_train,y_train)
    test_dataset=TensorDataset(x_test,y_test)
    # train_loader = DataLoader(train_dataset,  shuffle=False, batch_size=data_cfg.get('batch_size'), num_workers=data_cfg.get('num_workers'), pin_memory=True,
    # drop_last=True)
    test_loader = DataLoader(test_dataset,  shuffle=True, batch_size=data_cfg.get('batch_size'), 
                             num_workers=data_cfg.get('num_workers'), pin_memory=True, drop_last=True)
    """
    计算Precision、Recall、F1 Score、Confusion matrix
    """
    with torch.no_grad():
        preds,targets, image_paths = [],[],[]
        with tqdm(total=len(test_dataset)//data_cfg.get('batch_size')) as pbar:
            for _, batch in enumerate(test_loader):
                images, target = batch
                outputs = model(images.to(device),return_loss=False)
                preds.append(outputs)
                targets.append(target.to(device))
                image_paths.append("image_path")
                pbar.update(1)
                
    eval_results = evaluate(torch.cat(preds),torch.cat(targets),data_cfg.get('test').get('metrics'),data_cfg.get('test').get('metric_options'))
    
    get_metrics_output(eval_results,metrics_output,classes_names,indexs)
    get_prediction_output(torch.cat(preds),torch.cat(targets),image_paths, classes_names, indexs, prediction_output)                 

if __name__ == "__main__":
    main()

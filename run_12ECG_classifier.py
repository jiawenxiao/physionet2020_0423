#!/usr/bin/env python

import numpy as np
import joblib
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_12ECG_features(data,header_data):
    set_length=20000
    resample_interval=2
    data_num=np.zeros((1,12,set_length))
    data_external=np.zeros((1,3))
    length=data.shape[1]
    if length>=set_length:
        data_num[:,:,:]=data[:,:set_length]
    else:
        data_num[:,:,:length]=data
       
    resample_index=np.arange(0,set_length,resample_interval).tolist()
    data_num=data_num[:,:, resample_index]
    
    for lines in header_data:
        if lines.startswith('#Age'):
            age=lines.split(': ')[1].strip()
            if age=='NaN':
                age='60'     
        if lines.startswith('#Sex'):
            sex=lines.split(': ')[1].strip()
            
            
    length=data.shape[1]
    data_external[:,0]=float(age)/100
    data_external[:,1]=np.array(sex=='Male').astype(int) 
    data_external[:,2]=length/30000
    data_num=data_num/15000
    
    return data_num,data_external


def load_12ECG_model():
    model = torch.load('resnet_0420.pkl',map_location=device)
    return model

def run_12ECG_classifier(data,header_data,classes,model):
    
    num_classes = len(classes)
    current_label = np.zeros(num_classes, dtype=int)
    current_score = np.zeros(num_classes)
    # Use your classifier here to obtain a label and score for each class. 
    feats_reshape,feats_external = get_12ECG_features(data,header_data)
    
    feats_reshape = torch.tensor(feats_reshape,dtype=torch.float,device=device)
    feats_external = torch.tensor(feats_external,dtype=torch.float,device=device)
    
    
    pred = model.forward(feats_reshape,feats_external)
    pred =torch.sigmoid(pred)
    
    
    tmp_score = pred.squeeze().cpu().detach().numpy()    
    tmp_label = np.where(tmp_score>0.25,1,0)
    for i in range(num_classes):
        if np.sum(tmp_label)==0:
            max_index=np.argmax(tmp_score)
            tmp_label[max_index]=1
        if np.sum(tmp_label)>3:
            sort_index=np.argsort(tmp_score)
            min_index=sort_index[:6]
            tmp_label[min_index]=0 
    
    for i in range(num_classes):
        current_label[i] = np.array(tmp_label[i])
        current_score[i] = np.array(tmp_score[i])

    return current_label, current_score


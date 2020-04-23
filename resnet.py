# -*- coding: utf-8 -*-
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
from torchvision.models import resnet


class BasicBlock1d(nn.Module):


    def __init__(self, inplanes, planes, stride, size,downsample):
        super(BasicBlock1d, self).__init__()

        
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=size, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        
        self.conv2 = nn.Conv1d( planes, planes, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        
        self.downsample = nn.Sequential(
                nn.Conv1d(inplanes, planes ,kernel_size=size, stride=stride, bias=False),
                nn.BatchNorm1d(planes))
        self.dropout = nn.Dropout(.2)     
        self.sigmoid = nn.Sigmoid()


        self.globalAvgPool =nn.AdaptiveAvgPool1d(1)         
        self.fc1 = nn.Linear(in_features=planes, out_features=round(planes / 16))
        self.fc2 = nn.Linear(in_features=round(planes / 16), out_features=planes)    
        

    def forward(self, x):  
        
        x=x.squeeze(2)        
        residual = self.downsample(x)
        
        
        out = self.conv1(x)        
        out = self.bn1(out)
        out = self.relu(out)
        


        out = self.dropout(out)
        
        out = self.bn2(out)        
        out = self.conv2(out)

 
        #Squeeze-and-Excitation (SE)      
        original_out = out
        out = self.globalAvgPool(out) 
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1),1)
        out = out * original_out
        
        
        #resnet         
        out += residual
        out = self.relu(out)

        return out



class BasicBlock2d(nn.Module):

    def __init__(self, inplanes, planes, stride, size,downsample):
        super(BasicBlock2d, self).__init__()  
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(1,size), stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(1,1), stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes ,kernel_size=(1,size), stride=stride, bias=False),
                nn.BatchNorm2d(planes))
        

        self.dropout = nn.Dropout(.2)
        self.sigmoid = nn.Sigmoid()
        
        self.globalAvgPool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(in_features=planes, out_features=round(planes / 16))
        self.fc2 = nn.Linear(in_features=round(planes / 16), out_features=planes) 


    def forward(self, x):

        residual = self.downsample(x)
        
        
        out = self.conv1(x)        
        out = self.bn1(out)
        out = self.relu(out)


        out = self.dropout(out)
        
        out = self.bn2(out)        
        out = self.conv2(out)

          
       
    
    #Squeeze-and-Excitation (SE)   
        original_out=out
        out = self.globalAvgPool(out) 
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
      
        out = out.view(out.size(0), out.size(1),1,1)
        out = out * original_out

    #resnet           
        out += residual
        out = self.relu(out)

        return out

    
    
class ECGNet(nn.Module):
    def __init__(self, BasicBlock1d,BasicBlock2d, num_classes=9):
        super(ECGNet, self).__init__()
        
        self.sizes=[5,7,9]
        self.external = 3        
        

        self.relu = nn.ReLU(inplace=True)  
        
        self.conv1 =  nn.Conv2d(12,32, kernel_size=(1,50), stride=(1,2),padding=(0,0),bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.AvgPool = nn.AdaptiveAvgPool1d(1)
     
      
        self.layers=nn.Sequential()
        self.layers.add_module('layer_1',self._make_layer( BasicBlock2d,inplanes=32,planes=32,blocks=1,stride=(1,2),size=15))
        self.layers.add_module('layer_2',self._make_layer( BasicBlock2d,inplanes=32,planes=32,blocks=1,stride=(1,2),size=15))
        self.layers.add_module('layer_3',self._make_layer( BasicBlock2d,inplanes=32,planes=32,blocks=1,stride=(1,2),size=15))
        
        self.layers1_list=nn.ModuleList()
        self.layers2_list=nn.ModuleList()   
        
        
        for size in self.sizes:
            
   
            self.layers1=nn.Sequential()
            
            self.layers1.add_module('layer{}_1_1'.format(size),self._make_layer( BasicBlock2d,inplanes=32, planes=32,blocks=32,
                                                                                  stride=(1,1),size=size))

            
            
            self.layers2=nn.Sequential()   
            self.layers2.add_module('layer{}_2_1'.format(size),self._make_layer(BasicBlock1d,inplanes=32, planes=256,blocks=1,
                                                                                  stride=2,size=size))        

            self.layers2.add_module('layer{}_2_2'.format(size),self._make_layer(BasicBlock1d,inplanes=256, planes=256,blocks=1,
                                                                                  stride=2,size=size)) 
            
            self.layers2.add_module('layer{}_2_3'.format(size),self._make_layer(BasicBlock1d,inplanes=256, planes=256,blocks=1,
                                                                                  stride=2,size=size)) 
            
            self.layers2.add_module('layer{}_2_4'.format(size),self._make_layer(BasicBlock1d,inplanes=256, planes=256,blocks=1,
                                                                                  stride=2,size=size)) 
            

            self.layers1_list.append(self.layers1)
            self.layers2_list.append(self.layers2)
                     
        
        self.fc = nn.Linear(256*len(self.sizes)+self.external, num_classes)


    def _make_layer(self, block,inplanes, planes, blocks, stride ,size,downsample = None):
        layers = []
        for i in range(blocks):
            layers.append(block(inplanes, planes, stride, size,downsample))
        return nn.Sequential(*layers) 


    def forward(self, x0, fr):
        
        x0=x0.unsqueeze(2)

        x0 = self.conv1(x0)        
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.layers(x0)
        
        xs=[]
        for i in range(len(self.sizes)):
            
            x=self.layers1_list[i](x0)
            x=torch.flatten(x,start_dim=2,end_dim=3)
            x=self.layers2_list[i](x0)
            x= self.AvgPool(x)
            xs.append(x)
            

        out = torch.cat(xs,dim=2)
        out = out.view(out.size(0), -1)
        out = torch.cat([out,fr], dim=1)
        out = self.fc(out)

        return out


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ECGNet(BasicBlock1d,BasicBlock2d,**kwargs)
    return model

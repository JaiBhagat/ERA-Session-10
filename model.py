#Model 

import torch
import torch.nn as nn
import torch.nn.functional as F

dropout_value= 0.08

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1) 
        self.batch1= nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.batch2= nn.BatchNorm2d(128) 
        self.conv_res1 = nn.Conv2d(128, 128, 3, padding=1)
        self.batch_res1= nn.BatchNorm2d(128) 
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.batch3= nn.BatchNorm2d(256) 
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.batch4= nn.BatchNorm2d(512) 
        self.conv_res2 = nn.Conv2d(512, 512, 3, padding=1)
        self.batch_res2= nn.BatchNorm2d(512) 
        self.pool2 = nn.MaxPool2d(stride= 2, kernel_size= 4)
        self.flat= nn.Flatten()
        self.drop= nn.Dropout(dropout_value)
        self.dense= nn.Linear(2048, 10)




    def forward(self, x):
      x= self.drop(F.relu(self.batch1(self.conv1(x))))
      x= self.drop(F.relu(self.batch2(self.pool1(self.conv2(x)))))
      res_x1= F.relu(self.batch_res1(self.conv_res1(x)))
      res_x1= F.relu(self.batch_res1(self.conv_res1(res_x1)))
      f_x1=  torch.add(x, res_x1) # Can also be directly added 
      f_x1=self.drop( F.relu(self.batch3(self.pool1(self.conv3(f_x1)))))
      f_x1= self.drop(F.relu(self.batch4(self.conv4(f_x1))))
      res_x2= F.relu(self.batch_res2(self.conv_res2(f_x1)))
      res_x2= F.relu(self.batch_res2(self.conv_res2(res_x2)))
      f_x2=  torch.add(f_x1, res_x2)
      f_x2= self.pool2(f_x2)
      f_x2= self.drop(self.flat(f_x2))
      f_x2= self.dense(f_x2)
      f_x2 = f_x2.view(-1, 10)
      return F.log_softmax(f_x2)
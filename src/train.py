#!/usr/bin/env python
# coding: utf-8

# In[1]:


# data loader
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from torchvision import datasets, transforms as T
from tqdm import tqdm
import copy


# In[2]:


from image_dataset import ImageDataset 
from vggf32 import FCN32s


# In[3]:


############### Arguments ###############
batch_size = 4
num_epochs = 50
save_model_path = "./models/vgg-base.pth"
#########################################


# In[4]:


fcn32 = FCN32s()
model = models.vgg16(pretrained=True)
fcn32.copy_params_from_vgg16(model)


# In[5]:


train_dataset = ImageDataset("train")
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = ImageDataset("val")
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# In[6]:


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
fcn32 = fcn32.to(device)


# In[7]:


def cross_entropy2d(output, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = output.size()
    # log_p: (n, c, h, w)
    log_p = F.log_softmax(output, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
    if size_average:
        loss /= mask.data.sum()
    return loss


# In[30]:


def correct_2d(output, target):
    _, pred = torch.max(output, 1)
    cor = torch.sum(pred == label_tensor.data)
    batch_correct = cor / (512.0*512.0)
    return batch_correct


# In[8]:


optimizer = torch.optim.SGD(fcn32.parameters(), lr=0.003, momentum=0.9)


# In[9]:


val_loss_history = []

best_model_wts = copy.deepcopy(fcn32.state_dict())
best_loss = 99999

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    # train
    fcn32.train() 
    running_loss = 0.0
    epoch_correct = 0.0

    for image_tensor, label_tensor in tqdm(train_dataloader):
        image_tensor, label_tensor = image_tensor.to(device), label_tensor.to(device)
        optimizer.zero_grad()

        output = fcn32(image_tensor)
        loss = cross_entropy2d(output, label_tensor)
        correct = correct_2d(output, label_tensor)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        epoch_correct += correct.item()
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = epoch_correct / len(train_dataset)
    print("Epoch {} Training loss: {:4f}".format(epoch, epoch_loss))
    print("Epoch {} Training Acc: {:4f}".format(epoch, epoch_acc))

    # eval
    fcn32.eval() 
    val_loss = 0.0
    val_correct = 0.0
    for image_tensor, label_tensor in tqdm(val_dataloader):
        image_tensor, label_tensor = image_tensor.to(device), label_tensor.to(device)
        optimizer.zero_grad()

        with torch.no_grad():
            output = fcn32(image_tensor)
            loss = cross_entropy2d(output, label_tensor)
            val_loss += loss.item()
            correct = correct_2d(output, label_tensor)
            val_correct += correct.item()

    epoch_val_loss = val_loss / len(val_dataset)
    epoch_val_correct = val_correct / len(val_dataset)

    print("Epoch {} Val loss: {:4f}".format(epoch, epoch_val_loss))
    print("Epoch {} Val Acc: {:4f}".format(epoch, epoch_val_correct))

    if epoch_val_loss < best_loss:
        best_model_wts = copy.deepcopy(fcn32.state_dict())
        val_loss_history.append(epoch_val_loss)

    if epoch in (3, 25, 50):
        torch.save(fcn32.state_dict(), "./models/vgg-base_{}.pth".format(epoch))


# In[1]:


# load best model weight and save
fcn32.load_state_dict(best_model_wts)
torch.save(fcn32.state_dict(), save_model_path)


# In[ ]:





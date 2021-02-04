#!/usr/bin/env python
# coding: utf-8

# In[6]:


# data loader
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from torchvision import datasets, transforms as T
import matplotlib.pyplot as plt
import scipy.misc
from argparse import ArgumentParser


# In[2]:


from image_dataset import ImageDataset
from vggf32 import FCN32s
from vggf8 import FCN8s


# In[ ]:


def main(args):
    ############### Arguments ###############
    predict_mode = args.mode
    input_folder = args.input_path
    output_folder = args.output_path
    base_model_path = "./p2/p2_vgg_baseline.pth"
    improved_model_path = "./p2/p2_vgg_improved.pth"
    batch_size = 1
    #########################################
    test_set = ImageDataset("predict", predict_img_path=input_folder)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    if predict_mode == "baseline":
        model = FCN32s()
        save_model_path = base_model_path
    elif predict_mode == "improved":
        model = FCN8s()
        save_model_path = improved_model_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(torch.load(
        save_model_path, map_location=device))

    with torch.no_grad():
        model.eval()
        for image_tensor, img_id in test_loader:
            # predict
            image_tensor = image_tensor.to(device)
            output = model(image_tensor)
            _, pred = torch.max(output, 1)
            pred = pred[0]

            # save image
            pred = pred.cpu().numpy()
            pred_img = np.zeros((512, 512, 3))
            pred_img[np.where(pred == 0)] = [0, 255, 255]
            pred_img[np.where(pred == 1)] = [255, 255, 0]
            pred_img[np.where(pred == 2)] = [255, 0, 255]
            pred_img[np.where(pred == 3)] = [0, 255, 0]
            pred_img[np.where(pred == 4)] = [0, 0, 255]
            pred_img[np.where(pred == 5)] = [255, 255, 255]
            pred_img[np.where(pred == 6)] = [0, 0, 0]

            img_id = img_id[0]
            scipy.misc.imsave(
                "{}/{}_mask.png".format(output_folder, img_id), np.uint8(pred_img))


# In[ ]:


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--input_path", help="path of the test image directory",
                        dest="input_path", default="/shared_home/r08922168/dlcv/hw2-weifanhaha/hw2_data/p2_data/validation")
    parser.add_argument("--output_path", help="path where predict data saved",
                        dest="output_path", default="./pred")
    parser.add_argument("--mode", help="baseline or improved",
                        dest="mode", default='baseline')
    args = parser.parse_args()
    main(args)

import torch
import pickle
import torchvision
from torchvision import transforms
import torchvision.datasets as dset
from torchvision import transforms
from mydataset import OmniglotTrain, OmniglotTest
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
from model import Siamese
import time
import numpy as np
import gflags
import sys
from collections import deque
import PIL
import random,os
from PIL import Image
from IPython.display import Image

from PIL import Image
from torchvision.transforms import ToTensor
from matplotlib.pyplot import imshow


model = Siamese()
model.load_state_dict(torch.load('models/model-inter-19101.pt', map_location=torch.device('cpu')))

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

for i in range(2):
  c1 = random.choice(os.listdir("aug/valid")) #change dir name to whatever
  c2 = random.choice(os.listdir("aug/valid" + "/" +c1)) #change dir name to whatever
  c31 = random.choice(os.listdir("aug/valid" + "/" +c1 + "/" +c2)) #change dir name to whatever
  #print(c1, c2, c3)
  image1 = Image.open("aug/valid" + "/" +c1 + "/" +c2+ "/" +c31)
  img1 = ToTensor()(image1).unsqueeze(0) # unsqueeze to add artificial first dimension
  var_image1 = Variable(img1)

  c1 = random.choice(os.listdir("aug/valid")) #change dir name to whatever
  c2 = random.choice(os.listdir("aug/valid" + "/" +c1)) #change dir name to whatever
  c32 = random.choice(os.listdir("aug/valid" + "/" +c1 + "/" +c2)) #change dir name to whatever
  #print(c1, c2, c3)
  image2 = Image.open("aug/valid" + "/" +c1 + "/" +c2+ "/" +c32)
  img2 = ToTensor()(image2).unsqueeze(0) # unsqueeze to add artificial first dimension
  var_image2 = Variable(img2)

  output = model.forward(var_image1, var_image2).data.cpu().numpy()
  comb = get_concat_h(image1, image2)
  imshow(np.asarray(comb))
  print(c31, c32, output)

  c1 = random.choice(os.listdir("aug/valid")) #change dir name to whatever
  c2 = random.choice(os.listdir("aug/valid" + "/" +c1)) #change dir name to whatever
  c31 = random.choice(os.listdir("aug/valid" + "/" +c1 + "/" +c2)) #change dir name to whatever
  #print(c1, c2, c3)
  image1 = Image.open("aug/valid" + "/" +c1 + "/" +c2+ "/" +c31)
  img1 = ToTensor()(image1).unsqueeze(0) # unsqueeze to add artificial first dimension
  var_image1 = Variable(img1)

  c32 = random.choice(os.listdir("aug/valid" + "/" +c1 + "/" +c2)) #change dir name to whatever
  #print(c1, c2, c3)
  image2 = Image.open("aug/valid" + "/" +c1 + "/" +c2+ "/" +c32)
  img2 = ToTensor()(image2).unsqueeze(0) # unsqueeze to add artificial first dimension
  var_image2 = Variable(img2)

  output = model.forward(var_image1, var_image2).data.cpu().numpy()
  print(c31, c32, output)



# Introduction to Class activation mapping

This notebook is a first in a series where i try to review different methods for visualizing and explaining the decisions made by Deep Convolutional neural networks

The problem of interpretability is gaining popularity recently, as the field starting to mature, the accuracy in many vision tasks is practicly a solved problem (of course there is always room for improvement, but it has already surpassed human performance in some tasks like image recongnition ... ) now that we have production ready models that will work autonomously it's crucial to establish some regularisations concerning the latter, to be able to understand precicely the reason the model decided what it did, and insure ethics and to be more confident about the decisions

Many technics exist that tackle this problem:

- Gradient based visualisation

    - Deconvolution: this methode reverses the convolutions
    - Pure gradient: by computing gradient of the Error in respecet to the input image
    - Guided Backprop: something something
    - Class activation mapping: create a heatmap based on the importance of each feature 
        - GradCam
        - GradCam++
        - ScoreCam


- Perturbation based visualisations
    - LIME
- Score based visualisations
    - Layer wise relevence propagation
    - Deep Lift


```python
%load_ext tensorboard

import numpy as np # linear algebra library
import cv2 # for image processing methods 
import matplotlib.pyplot as plt # visualisation tool
from matplotlib import gridspec
import seaborn as sns
import pandas as pd
import os # for managing file paths and os operations
import sys # system specefic calls
import time 
import pdb # debug

from sklearn.metrics import confusion_matrix, classification_report

import torch # Deep learning framework
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim

import torchvision # for creating datasets, and pretrained models
from torch.utils.tensorboard import SummaryWriter # visualise the learning
from  torch.utils.data import DataLoader # parellel dataset loader
from torchvision import models, datasets, transforms
from torchviz import make_dot

import copy # to create deep copies
import pickle as pk # for serialization

```

    The tensorboard extension is already loaded. To reload it, use:
      %reload_ext tensorboard
    

# Preprocessing of the data

First of all, i choose a simple dataset to test the methods and make sure they work correctly (i can easily interpret the outputs and dont require expert knowledge to validate the results) this dataset also has the convinience of being separated into train validation and test sets, so a win for us.

we are going to use models pretrained on Imagenet, so to have more accurate results, we have to convert our images to be from the same distribution as Imagenet
by normalizing using the mean and std of the original train set used for Imagenet

for now we dont need image augmentation, so we just need to resize images to work with our model, pytorch makes it very convinient by efining a pipeline for transforms
the is applied to each set.


```python
data_dir = os.path.abspath('../../datasets/fruits')


# preprocessing piepeline image

normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])

img_transforms = {"train": transforms.Compose([transforms.Resize((224, 224)),
                                              transforms.ToTensor(),
                                              normalize]),
                  "val": transforms.Compose([transforms.Resize((224, 224)),
                                            transforms.ToTensor(),
                                            normalize]),
                  "test": transforms.Compose([transforms.Resize((224, 224)),
                                            transforms.ToTensor(),
                                            normalize])}

# load the sets in a dictionary for convienience
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), img_transforms[x])
                  for x in ['train', 'val', 'test']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                              batch_size=32,
                                              shuffle=True,
                                              num_workers=8,
                                              prefetch_factor=3,
                                              pin_memory=True)
              for x in ['train', 'val', 'test']}

# automaticly pick GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```


```bash
%%bash --out output --err error
nvidia-smi
```


```python
print("Using GPU :D ! \n" + output if device.type == 'cuda' else "Using CPU :( ! ")
```

    Using GPU :D ! 
    Sun Feb 14 20:43:46 2021       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 460.32.03    Driver Version: 460.32.03    CUDA Version: 11.2     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  GeForce GTX 106...  Off  | 00000000:01:00.0  On |                  N/A |
    |  0%   59C    P0    26W / 200W |    243MiB /  6077MiB |      2%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |    0   N/A  N/A      1448      G   /usr/lib/xorg/Xorg                160MiB |
    |    0   N/A  N/A      2481      G   /usr/bin/gnome-shell               35MiB |
    |    0   N/A  N/A      5484      G   /usr/lib/firefox/firefox            1MiB |
    |    0   N/A  N/A     17083      G   ...AAAAAAAAA= --shared-files       18MiB |
    +-----------------------------------------------------------------------------+
    
    


```python
# meta
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes
print(f"There is : {len(class_names)}")

plt.figure()
plt.yscale("log")
plt.bar(np.arange(0, 3), dataset_sizes.values())
plt.xticks(np.arange(0, 3), dataset_sizes.keys())
```

    There is : 36
    




    ([<matplotlib.axis.XTick at 0x7fd59208cd00>,
      <matplotlib.axis.XTick at 0x7fd59208ccd0>,
      <matplotlib.axis.XTick at 0x7fd59207a820>],
     [Text(0, 0, 'train'), Text(1, 0, 'val'), Text(2, 0, 'test')])




![png](output_6_2.png)


# Visualising the dataset

OK, now let's load a batch to see how the training set looks, we'll print the label of each image on top


```python
batch, labels = next(iter(dataloaders["train"]))


def plot_batch(btach, labels):

    fig, axes = plt.subplots(6, 6, figsize=(30, 20))
    axes = axes.reshape(-1)

    mean = np.array([0.485, 0.456, 0.406])
    std =  np.array([0.229, 0.224, 0.225])

    red = "#D32F2F"
    green = "#4CAF50"

    for i, (ax, img) in enumerate(zip(axes, batch)):

        img = img.numpy().transpose(1, 2, 0)*std + mean
        img[img < 0] = 0
        img[img > 1] = 1
        ax.set_title(class_names[labels[i]], fontweight="bold")
        ax.imshow(img)
        ax.set_axis_off()
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    
plot_batch(batch, labels)
```

    


![png](output_8_1.png)


# Defining the model

As a first experiemnt we'll content by using a simple VGG model pretrained on imagenet, and just removing it's last layer as we don't need that part
we introduce a Global Average Pool layer, which is basicly just a 1x1 convolution, it will considerably reduce our params count, and lets us classify direcly using the feature maps rather than flattening them then feeding them to an intermediate fully connected layer


```python
class VggCam(nn.Module):
    """
        This model uses a pretrained VGG as a feature extractor
        then passes it's output to a linear layer
    """
    
    def __init__(self):
        """
            Constructor of the model
        """
        super(VggCam, self).__init__()
        
        # Feature extractor
        #self.model = models.vgg16(pretrained=True)
        
        self.features = models.vgg16(pretrained=True).features
        
        
        for param in self.features.parameters():
            param.requires_grad = False
        
        
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, len(class_names), bias=False)
        
                                    
    def forward(self, X):
        """
            the forward pass of the model
        """
        out = self.features(X)
        out = self.gap(out)
        out = out.view(X.shape[0], -1)
        
        return self.fc(out)
```


```python
model = VggCam().to(device)
batch, labels = next(iter(dataloaders["train"]))
batch = batch.to(device)

mean = np.array([0.485, 0.456, 0.406])
std =  np.array([0.229, 0.224, 0.225])

writer = SummaryWriter("logs")
imgs = torchvision.utils.make_grid(batch.cpu()*std[None, :, None, None] + mean[None, :, None, None])
imgs += imgs.min()
imgs = torch.clamp(imgs, 0, 1)
print(imgs.min(), imgs.max())
writer.add_image('train_batch', imgs, 0)
writer.add_graph(model, batch)
writer.flush()
writer.close()
```

    

    tensor(0., dtype=torch.float64) tensor(1., dtype=torch.float64)
    


```python
%tensorboard --logdir logs
```


    Reusing TensorBoard on port 6006 (pid 7598), started 3:06:43 ago. (Use '!kill 7598' to kill it.)




      <iframe id="tensorboard-frame-1cc661cc5a95a1a1" width="100%" height="800" frameborder="0">
      </iframe>
      <script>
        (function() {
          const frame = document.getElementById("tensorboard-frame-1cc661cc5a95a1a1");
          const url = new URL("/", window.location);
          const port = 6006;
          if (port) {
            url.port = port;
          }
          frame.src = url;
        })();
      </script>
    


# Setup training loop

Very generic training loop, we define a function that takes the model in question, some criterion for our case it would be NLLLoss, an Optimizer Adam for our case ,a learning_rate scheduler to prevent local minima, and finnally Epochs (how many times we iterate through the full dataset)


```python
def train_model(model, criterion, optimizer, scheduler, num_epochs=25, verbose=True, writer_title="logs"):
    
    since = time.time()

    if(verbose):
        writer = SummaryWriter(writer_title)
    # store the best model state
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    with SummaryWriter() as writer:
        
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        #pdb.set_trace()
                        #riter.add_scalar('BatchLoss/'+phase, loss, epoch)
                        
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()


                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)


                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                #writer.add_scalar('EpochLoss/'+phase, epoch_loss, epoch)
                #writer.add_scalar('EpochAcc/'+phase, epoch_loss, epoch)
                
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    #best_model_wts = best_model_wts.cpu()
                    torch.save(best_model_wts, "weights.pk")

            print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
```

# Training

now that eveyrthing is setted up let's train out model


```python
torch.cuda.empty_cache()
!nvidia-smi
```

    Sun Feb 14 20:46:41 2021       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 460.32.03    Driver Version: 460.32.03    CUDA Version: 11.2     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  GeForce GTX 106...  Off  | 00000000:01:00.0  On |                  N/A |
    |  0%   56C    P8    14W / 200W |   1040MiB /  6077MiB |     49%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |    0   N/A  N/A      1448      G   /usr/lib/xorg/Xorg                140MiB |
    |    0   N/A  N/A      2481      G   /usr/bin/gnome-shell               35MiB |
    |    0   N/A  N/A      5484      G   /usr/lib/firefox/firefox            1MiB |
    |    0   N/A  N/A     17083      G   ...AAAAAAAAA= --shared-files       18MiB |
    |    0   N/A  N/A     26550      C   ...r 1/thesis/env/bin/python      817MiB |
    +-----------------------------------------------------------------------------+
    


```python
model = VggCam().to(device)

num_epochs = 25
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.01)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
criterion = nn.CrossEntropyLoss()
best_model = train_model(model, criterion, optimizer, lr_scheduler, num_epochs)
```

    Epoch 0/24
    ----------
    

    

    train Loss: 1.3744 Acc: 0.6339
    

    

    val Loss: 0.3885 Acc: 0.8946
    
    Epoch 1/24
    ----------
    

    

    train Loss: 0.6259 Acc: 0.8171
    

    

    val Loss: 0.2094 Acc: 0.9373
    
    Epoch 2/24
    ----------
    

    

    train Loss: 0.4640 Acc: 0.8590
    

    

    val Loss: 0.1483 Acc: 0.9516
    
    Epoch 3/24
    ----------
    

    

    train Loss: 0.3323 Acc: 0.9070
    

    

    val Loss: 0.1121 Acc: 0.9801
    
    Epoch 4/24
    ----------
    

    

    train Loss: 0.2486 Acc: 0.9271
    

    

    val Loss: 0.1330 Acc: 0.9601
    
    Epoch 5/24
    ----------
    

    

    train Loss: 0.2004 Acc: 0.9439
    

    

    val Loss: 0.1375 Acc: 0.9658
    
    Epoch 6/24
    ----------
    

    

    train Loss: 0.1833 Acc: 0.9486
    

    

    val Loss: 0.0996 Acc: 0.9658
    
    Epoch 7/24
    ----------
    

    

    train Loss: 0.1016 Acc: 0.9779
    

    

    val Loss: 0.0526 Acc: 0.9858
    
    Epoch 8/24
    ----------
    

    

    train Loss: 0.0880 Acc: 0.9824
    

    

    val Loss: 0.0511 Acc: 0.9858
    
    Epoch 9/24
    ----------
    

    

    train Loss: 0.0844 Acc: 0.9841
    

    

    val Loss: 0.0555 Acc: 0.9829
    
    Epoch 10/24
    ----------
    

    

    train Loss: 0.0810 Acc: 0.9844
    

    

    val Loss: 0.0495 Acc: 0.9886
    
    Epoch 11/24
    ----------
    

    

    train Loss: 0.0792 Acc: 0.9869
    

    

    val Loss: 0.0531 Acc: 0.9829
    
    Epoch 12/24
    ----------
    

    

    train Loss: 0.0778 Acc: 0.9872
    

    

    val Loss: 0.0500 Acc: 0.9858
    
    Epoch 13/24
    ----------
    

    

    train Loss: 0.0764 Acc: 0.9872
    

    

    val Loss: 0.0475 Acc: 0.9886
    
    Epoch 14/24
    ----------
    

    

    train Loss: 0.0710 Acc: 0.9897
    

    

    val Loss: 0.0476 Acc: 0.9886
    
    Epoch 15/24
    ----------
    

    

    train Loss: 0.0706 Acc: 0.9886
    

    

    val Loss: 0.0475 Acc: 0.9886
    
    Epoch 16/24
    ----------
    

    

    train Loss: 0.0704 Acc: 0.9897
    

    

    val Loss: 0.0474 Acc: 0.9886
    
    Epoch 17/24
    ----------
    

    

    train Loss: 0.0703 Acc: 0.9888
    

    

    val Loss: 0.0477 Acc: 0.9886
    
    Epoch 18/24
    ----------
    

    

    train Loss: 0.0701 Acc: 0.9886
    

    

    val Loss: 0.0476 Acc: 0.9886
    
    Epoch 19/24
    ----------
    

    

    train Loss: 0.0699 Acc: 0.9894
    

    

    val Loss: 0.0477 Acc: 0.9858
    
    Epoch 20/24
    ----------
    

    

    train Loss: 0.0698 Acc: 0.9891
    

    

    val Loss: 0.0473 Acc: 0.9886
    
    Epoch 21/24
    ----------
    

    

    train Loss: 0.0690 Acc: 0.9897
    

    

    val Loss: 0.0473 Acc: 0.9886
    
    Epoch 22/24
    ----------
    

    

    train Loss: 0.0690 Acc: 0.9897
    

    

    val Loss: 0.0473 Acc: 0.9886
    
    Epoch 23/24
    ----------
    

    

    train Loss: 0.0690 Acc: 0.9897
    

    

    val Loss: 0.0473 Acc: 0.9886
    
    Epoch 24/24
    ----------
    

    

    train Loss: 0.0689 Acc: 0.9897
    

    

    val Loss: 0.0473 Acc: 0.9886
    
    Training complete in 27m 29s
    Best val Acc: 0.988604
    

# Visualisation

let's visualize the CAM FINALLY


```python
def return_CAM(feature_conv, weight):

    size_upsample = (224, 224)
    nc, h, w = feature_conv.shape
    beforeDot =  feature_conv.reshape((nc, h*w))
    cam = np.matmul(weight, beforeDot)
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)

    return cv2.resize(cam_img, size_upsample, interpolation=cv2.INTER_CUBIC)


```


```python
batch, labels = next(iter(dataloaders["test"]))

batch = batch.to(device)
labels = labels.to(device)

preds = best_model(batch)

feature_maps = best_model.features(batch).cpu().detach().numpy()
weight = best_model.fc.weight.cpu().detach().numpy()
```

    


```python
def plot_report(y, yhat, labels):
    """
    
    """
    
    cm = confusion_matrix(y, y_hat)
    cr = classification_report(y, y_hat, output_dict=True)
    
    plt.figure(figsize=(20,10))
    plt.subplot(121)
    sns.heatmap(cm, annot=True)
    plt.subplot(122)
    sns.heatmap(pd.DataFrame(cr).iloc[:-1, :].T, annot=True)
    

def test(model, phase="test", verbose=True):
    
    model.eval()
    
    y_hat = []
    y = []
    running_loss = 0
    
    with torch.no_grad():
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(F.softmax(outputs, 1), 1)
            
            y_hat = y_hat + list(preds.cpu().detach().numpy())
            y = y + list(labels.cpu().detach().numpy())
            
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
        
        running_loss /= dataset_sizes[phase]
        
    return y_hat, y
```


```python
 y_hat, y = test(best_model, "test")    
```

    


```python
plot_report(y, y_hat, [])
```


![png](output_23_0.png)



```python
batch, labels = next(iter(dataloaders["test"]))

batch = batch.to(device)
labels = labels.to(device)

preds = best_model(batch)

feature_maps = best_model.features(batch).cpu().detach().numpy()
weights = best_model.fc.weight.cpu().detach().numpy()

```

    


```python
mean = np.array([0.485, 0.456, 0.406])
std =  np.array([0.229, 0.224, 0.225])
```


```python
m = feature_maps.max()/4
for i, img in enumerate(batch):
    
    
    
    fig = plt.figure(figsize=(15, 10)) 

    gs = gridspec.GridSpec(3, 5,wspace=0.0, hspace=0.0, width_ratios=[1, 1, 1, 1, 1],
                          top=0.8, bottom=0.2, left=0.17, right=0.845) 
    
    sorted_features = sorted(list(feature_maps[i]), key=lambda x: x.sum(), reverse=True)
    
#     fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(15, 10))
#     axes = axes.reshape(-1)

    img = np.clip(img.cpu().numpy().transpose(1, 2, 0)*std + mean, 0, 1)


    for j in range(3):
        for k in range(5):  
            ax = fig.add_subplot(gs[j, k])
            if(j == 0 and k == 0):
                ax.imshow(img)
            else:
                feature = (cv2.resize(sorted_features[j*4 + k], (224,224), interpolation=cv2.INTER_CUBIC) - sorted_features[j*4+k].min())/(sorted_features[j*4 + k].max()-sorted_features[j*4 + k].min())
                #m = feature.mean()
                #cam[cam<m] = 0
                feature = np.clip(feature, 0, 1)
                #img[cam == 0] = 0
                img = img*(feature[:, :, None])

                #img = np.clip(img, 0, 1)
                img = np.interp(img,(img.min(), img.max()), (0, 1))
                ax.imshow(img)
                
            ax.set_axis_off()
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

    plt.show()
```


![png](output_26_0.png)



![png](output_26_1.png)



![png](output_26_2.png)



![png](output_26_3.png)



![png](output_26_4.png)



![png](output_26_5.png)



![png](output_26_6.png)



![png](output_26_7.png)



![png](output_26_8.png)



![png](output_26_9.png)



![png](output_26_10.png)



![png](output_26_11.png)



![png](output_26_12.png)



![png](output_26_13.png)



![png](output_26_14.png)



![png](output_26_15.png)



![png](output_26_16.png)



![png](output_26_17.png)



![png](output_26_18.png)



![png](output_26_19.png)



![png](output_26_20.png)



![png](output_26_21.png)



![png](output_26_22.png)



![png](output_26_23.png)



![png](output_26_24.png)



![png](output_26_25.png)



![png](output_26_26.png)



![png](output_26_27.png)



![png](output_26_28.png)



![png](output_26_29.png)



![png](output_26_30.png)



![png](output_26_31.png)



```python
_, preds = torch.max(F.softmax(preds, 1), 1)
```


```python
preds[1].item()
```




    14




```python
fig, axes = plt.subplots(6, 6, figsize=(30, 20))
axes = axes.reshape(-1)

mean = np.array([0.485, 0.456, 0.406])
std =  np.array([0.229, 0.224, 0.225])

red = "#D32F2F"
green = "#4CAF50"


for i, (ax, img) in enumerate(zip(axes, batch)):
    
    cam = return_CAM(feature_maps[i], weights[preds[i].item()])
    img = img.cpu().detach().numpy().transpose(1, 2, 0)*std + mean
    img[img < 0] = 0
    img[img > 1] = 1
    ax.set_title(class_names[labels[i]], color=green if labels[i]==preds[i] else red, fontweight="bold")
    ax.imshow(img)
    ax.imshow(cam/255, alpha=0.6, cmap='jet')
    ax.set_axis_off()
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
```


![png](output_29_0.png)



```python
fig, axes = plt.subplots(6, 6, figsize=(30, 20))
axes = axes.reshape(-1)

mean = np.array([0.485, 0.456, 0.406])
std =  np.array([0.229, 0.224, 0.225])

red = "#D32F2F"
green = "#4CAF50"


for i, (ax, img) in enumerate(zip(axes, batch)):
    
    cam = return_CAM(feature_maps[i], weights[preds[i].item()])/255
    img = img.cpu().detach().numpy().transpose(1, 2, 0)*std + mean
    img[img < 0] = 0
    img[img > 1] = 1
    
    m = np.quantile(cam, 0.6)
    #cam[cam<m] = 0
    #img[cam == 0] = 0
    img = img*(cam[:, :, None])**(2)
    
    ax.set_title(class_names[labels[i]], color=green if labels[i]==preds[i] else red, fontweight="bold")
    ax.imshow(img)
    #ax.imshow(cam, alpha=0.6, cmap='jet')
    ax.set_axis_off()
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
```


![png](output_30_0.png)



```python

mean = np.array([0.485, 0.456, 0.406])
std =  np.array([0.229, 0.224, 0.225])

red = "#D32F2F"
green = "#4CAF50"


for i, img in enumerate(batch):
    
    fig= plt.figure(figsize=(10,10))
    
    ax = fig.gca()
    
    cam = return_CAM(feature_maps[i], weights[preds[i].item()])/255
    img = img.cpu().detach().numpy().transpose(1, 2, 0)*std + mean
    
    m = np.quantile(cam, 0.5)
    #cam[cam<m] = 0
    #img[cam == 0] = 0
    img = img*(cam[:, :, None])**2
    
    img[img < 0] = 0
    img[img > 1] = 1
    ax.set_title(class_names[labels[i]], color=green if labels[i]==preds[i] else red, fontweight="bold")
    ax.imshow(img)
    ax.set_axis_off()
```

    <ipython-input-217-257fd6fc3889>:10: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig= plt.figure(figsize=(10,10))
    


![png](output_31_1.png)



![png](output_31_2.png)



![png](output_31_3.png)



![png](output_31_4.png)



![png](output_31_5.png)



![png](output_31_6.png)



![png](output_31_7.png)



![png](output_31_8.png)



![png](output_31_9.png)



![png](output_31_10.png)



![png](output_31_11.png)



![png](output_31_12.png)



![png](output_31_13.png)



![png](output_31_14.png)



![png](output_31_15.png)



![png](output_31_16.png)



![png](output_31_17.png)



![png](output_31_18.png)



![png](output_31_19.png)



![png](output_31_20.png)



![png](output_31_21.png)



![png](output_31_22.png)



![png](output_31_23.png)



![png](output_31_24.png)



![png](output_31_25.png)



![png](output_31_26.png)



![png](output_31_27.png)



![png](output_31_28.png)



![png](output_31_29.png)



![png](output_31_30.png)



![png](output_31_31.png)



![png](output_31_32.png)


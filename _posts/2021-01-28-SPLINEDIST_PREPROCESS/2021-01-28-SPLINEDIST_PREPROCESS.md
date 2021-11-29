```python

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys
import time
import pytorch_lightning as pl
import torchvision
import random
from scipy.spatial.distance import cdist, directed_hausdorff
sys.path.append(os.path.join(os.getcwd(), '..'))
#from utils import *
from Models.unet import UNet
from Datasets.DSB18 import Nuclie_datamodule
%load_ext autoreload
%autoreload 2
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload
    

# Sample of the dataset
In spline dist, the data used is the rgb image, with the ground truth contour of the cells


```python
# intialize the dataloaders
datamodule = Nuclie_datamodule()

train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()

batch_x, batch_y = next(iter(train_loader))

#make grid of the unormalized batch_x using torchvision.utils.make_grid
grid = torchvision.utils.make_grid(batch_x)
masks = torchvision.utils.make_grid(batch_y)
mean, std = 0.5, 0.5
std_inv = 1 / (std + 1e-7)
mean_inv = -mean * std_inv
grid = torchvision.transforms.Normalize(mean=mean_inv, std=std_inv)(grid)

# and show it
plt.figure(figsize=(15, 15))
plt.imshow(grid.permute(1,2,0).clamp(0,1))
plt.show()

plt.figure(figsize=(15, 15))
plt.imshow(masks.permute(1,2,0).clamp(0,1))
plt.show()
```


    
![png](output_2_0.png)
    



    
![png](output_2_1.png)
    


# Contour generation from the binary masks




```python
def computeContours(img):
    contours, hierachy = cv2.findContours(img.astype(np.uint8), 
                                  mode=cv2.RETR_LIST, 
                                  method=cv2.CHAIN_APPROX_NONE)
    return contours
# draw a disk in an image using cv2
img = np.zeros((512, 512, 1), np.uint8)
cv2.circle(img, (256, 256), 100, (255, 255, 255), -1)
cv2.circle(img, (150, 150), 100, (255, 255, 255), -1)
#draw a rectangle in an image using cv2
cv2.rectangle(img, (300, 300), (400, 400), (255, 255, 255), -1)

plt.figure(figsize=(10, 10))
contours = computeContours(img)
plt.figure(figsize=(10, 10))
plt.imshow(img, cmap='gray')
plt.plot(contour[0][:,0,0], contour[0][:,0,1], 'r')
```




    [<matplotlib.lines.Line2D at 0x1f93addbe20>]




    <Figure size 720x720 with 0 Axes>



    
![png](output_5_2.png)
    


# Generaet object probabilities using binary masks


```python
# compute distance transform of img using opencv with the normalized euclidean distance metric
def compute_dist_transform(img):
    dist = cv2.distanceTransform(img, cv2.DIST_L2, 5)
    # normalize the distance transform
    dist = cv2.normalize(dist, None, 0, 1, cv2.NORM_MINMAX)
    return dist

dist = compute_dist_transform(img)
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.imshow(dist, cmap='gray')
plt.subplot(1, 2, 2)
# show histogram of the distance transform without 0
plt.hist(dist[dist>0].ravel(), bins=100)
plt.show()
```


    
![png](output_7_0.png)
    





```python
# run watershed on dist
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
from skimage import measure

thresh = img.reshape(512, 512)
local_maxi = peak_local_max(
    dist, indices=False, footprint=np.ones((3, 3)), labels=thresh)
markers = measure.label(local_maxi)
labels_ws = watershed(-dist, markers, mask=thresh)

# show markers and segmentation
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(np.clip(dist,0,1), cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(labels_ws)
plt.show()


```

    C:\Users\mbenimam\AppData\Local\Temp/ipykernel_25788/2356942329.py:8: FutureWarning: indices argument is deprecated and will be removed in version 0.20. To avoid this warning, please do not use the indices argument. Please see peak_local_max documentation for more details.
      local_maxi = peak_local_max(
    


    
![png](output_9_1.png)
    


# let's try every thing on a real exemple





```python
mask = np.uint8(batch_y[3].numpy().squeeze()>0)
dist = compute_dist_transform(mask)
# show image with historgram and fix the size of the histogram
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 6))
ax1.imshow(mask, cmap='gray')
ax1.set_title('Binary mask')
# show 

ax2.imshow(dist, cmap='gray')
ax2.set_title('Object probability')


contours = computeContours(mask)
ax3.imshow(mask, cmap='gray')
for contour in contours:
    ax3.plot(contour[:,0,0], contour[:,0,1], 'r')
ax3.set_title('Contours')

plt.show()

```


    
![png](output_12_0.png)
    


# Compute the distance bettween two contours



```python
def computeDistanceBetweenInstance(instance1, instance2, plot=False):

    # show two images containing contours 1 and 2
    contour1, contour2 = instance1.reshape(-1, 2), instance2.reshape(-1, 2)
    sampling_factor = min(len(contour1), len(contour2))

    # The two contours must have the same length
    index = np.floor(np.linspace(0, len(contour1)-1, sampling_factor)).astype(int)
    contour1 = contour1[index]

    index = np.floor(np.linspace(0, len(contour2)-1, sampling_factor)).astype(int)
    contour2 = contour2[index]
    #print(f"contour1: {len(contour1)}, contour2: {len(contour2)}")

    # translate the contours to the same origin
    contour1 = contour1 - contour1.mean(0)
    contour2 = contour2 - contour2.mean(0)
    # compute the bounding box of the contours

    # we must match every pixel of the contour to the corresponding pixel of the other contour
    # we use the euclidean distance to compute the distance between the contour pixels

    scikit_dist = cdist(contour1, contour2, metric='euclidean')
    scikit_dist_min = scikit_dist.min(axis=1)
    scikit_dist_min_args = scikit_dist.argmin(axis=1)
    dist_hausdorff = directed_hausdorff(contour1, contour2)[0]
    # show the distance matrix
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))



        ax1.plot(contour1[:,0], contour1[:,1], 'go-', label='contour1')
        ax1.plot(contour2[:,0], contour2[:,1], 'ro-', label='contour2')
        ax1.set_title(f"Contours with their closest mathcing, Dist={scikit_dist_min_args.sum()}, hausforf={dist_hausdorff}")
        # plot a line for each couple of pixels of index i and j in the indeces returned by scikit_dist_min_args
        for i, j in zip(scikit_dist_min_args, scikit_dist_min):
            ax1.plot([contour1[i,0], contour2[i,0]], [contour1[i,1], contour2[i,1]], 'b--')

        ax2.imshow(scikit_dist, cmap='gray')
        ax2.set_title('pairwise Euclidean distance between pixels')
        plt.show()

    return dist_hausdorff

computeDistanceBetweenInstance(contours[3], contours[6], plot=True)
```


    
![png](output_14_0.png)
    





    8.315808492973268




```python
import copy
# compute the distance matrix using the euclidean distance between pairs of instances
contours2 = copy.deepcopy(contours) # just a toy exemple of prediction
# contours1_bis = contours1.reshape(contours1.shape(0), -1)
# contours2_bis = contours2.reshape(contours2.shape(0), -1)
# dist = pairwise(contours1_bis, contours2_bis, metric=computeDistanceBetweenInstance, n_jobs=1)
dist_mat = np.zeros((len(contours), len(contours2)))

for i in range(len(contours)):
    for j in range(len(contours2)):
        dist_mat[i,j] = computeDistanceBetweenInstance(contours[i], contours2[j], plot=False)

# show the distance matrix
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig = plt.figure(figsize=(14, 5))
plt.imshow(dist_mat, cmap='jet')
plt.colorbar()
plt.title('pairwise distance between instances')
plt.show()

```


    
![png](output_15_0.png)
    



```python
dist_mat.argmin(axis=1)
```




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10], dtype=int64)




```python

```

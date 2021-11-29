```python
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 19:43:53 2021

@author: Mounib Benimam
"""
%load_ext autoreload
%autoreload 2
from utils import loadDataset, plot_report, test, plot_batch, unormalize, unormalize_img, plot_imgs, visualise_top_k             
from models.VisualisableModel import VggCam, DeconvolutionVisualisationModel
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
import cv2
import torch.nn.functional as F
from PIL import Image
from utils import normalize
from torchvision import transforms
```


```python
datasets, dataloaders, datasizes, classnames, _= loadDataset("fruits")
```


```python
plot_batch(dataloaders["train"], classnames, cols=8)
```


    
![png](output_2_0.png)
    



```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using GPU :D !" if device.type == 'cuda' else "Using CPU :( ! ")

model = VggCam(classnames)
model.load_from_file("models/VggCamFruitsWeights.pk")
model = model.to(device)
# yhat, y = test(model, dataloaders["test"], datasizes["test"])
# plot_report(y, yhat, classnames)
```

    Using GPU :D !
    


```python
batch, labels = next(iter(dataloaders["train"]))
```


```python
deconv = DeconvolutionVisualisationModel(model)
```


```python
b, c, h, w = batch.shape
m = deconv.generateMap(batch[0].reshape(1, c, h, w).to(device), labels[0], device)
```


```python
torch.cuda.empty_cache()
```


```python
for s in range(32):
    img = batch[s]
    visualise_top_k(deconv, batch[s], labels[s], device, classnames, 3, subplots=(2, 4), figsize=(20, 10))
```

    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    


    
![png](output_8_1.png)
    



    
![png](output_8_2.png)
    



    
![png](output_8_3.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](output_8_5.png)
    



    
![png](output_8_6.png)
    



    
![png](output_8_7.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](output_8_9.png)
    



    
![png](output_8_10.png)
    



    
![png](output_8_11.png)
    


    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    


    <Figure size 432x288 with 0 Axes>



    
![png](output_8_14.png)
    



    
![png](output_8_15.png)
    



    
![png](output_8_16.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](output_8_18.png)
    



    
![png](output_8_19.png)
    



    
![png](output_8_20.png)
    


    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    


    <Figure size 432x288 with 0 Axes>



    
![png](output_8_23.png)
    



    
![png](output_8_24.png)
    



    
![png](output_8_25.png)
    


    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    


    <Figure size 432x288 with 0 Axes>



    
![png](output_8_28.png)
    



    
![png](output_8_29.png)
    



    
![png](output_8_30.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](output_8_32.png)
    



    
![png](output_8_33.png)
    



    
![png](output_8_34.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](output_8_36.png)
    



    
![png](output_8_37.png)
    



    
![png](output_8_38.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](output_8_40.png)
    



    
![png](output_8_41.png)
    



    
![png](output_8_42.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](output_8_44.png)
    



    
![png](output_8_45.png)
    



    
![png](output_8_46.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](output_8_48.png)
    



    
![png](output_8_49.png)
    



    
![png](output_8_50.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](output_8_52.png)
    



    
![png](output_8_53.png)
    



    
![png](output_8_54.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](output_8_56.png)
    



    
![png](output_8_57.png)
    



    
![png](output_8_58.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](output_8_60.png)
    



    
![png](output_8_61.png)
    



    
![png](output_8_62.png)
    


    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    


    <Figure size 432x288 with 0 Axes>



    
![png](output_8_65.png)
    



    
![png](output_8_66.png)
    



    
![png](output_8_67.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](output_8_69.png)
    



    
![png](output_8_70.png)
    



    
![png](output_8_71.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](output_8_73.png)
    



    
![png](output_8_74.png)
    



    
![png](output_8_75.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](output_8_77.png)
    



    
![png](output_8_78.png)
    



    
![png](output_8_79.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](output_8_81.png)
    



    
![png](output_8_82.png)
    



    
![png](output_8_83.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](output_8_85.png)
    



    
![png](output_8_86.png)
    



    
![png](output_8_87.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](output_8_89.png)
    



    
![png](output_8_90.png)
    



    
![png](output_8_91.png)
    


    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    


    <Figure size 432x288 with 0 Axes>



    
![png](output_8_94.png)
    



    
![png](output_8_95.png)
    



    
![png](output_8_96.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](output_8_98.png)
    



    
![png](output_8_99.png)
    



    
![png](output_8_100.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](output_8_102.png)
    



    
![png](output_8_103.png)
    



    
![png](output_8_104.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](output_8_106.png)
    



    
![png](output_8_107.png)
    



    
![png](output_8_108.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](output_8_110.png)
    



    
![png](output_8_111.png)
    



    
![png](output_8_112.png)
    


    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    


    <Figure size 432x288 with 0 Axes>



    
![png](output_8_115.png)
    



    
![png](output_8_116.png)
    



    
![png](output_8_117.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](output_8_119.png)
    



    
![png](output_8_120.png)
    



    
![png](output_8_121.png)
    


    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    


    <Figure size 432x288 with 0 Axes>



    
![png](output_8_124.png)
    



    
![png](output_8_125.png)
    



    
![png](output_8_126.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](output_8_128.png)
    



    
![png](output_8_129.png)
    



    
![png](output_8_130.png)
    


    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    


    <Figure size 432x288 with 0 Axes>



    
![png](output_8_133.png)
    



    
![png](output_8_134.png)
    



    
![png](output_8_135.png)
    



    <Figure size 432x288 with 0 Axes>



```python
# feature_maps.shape
```


```python
torch.cuda.empty_cache()
```


```python
#!nvidia-smi
```

# Let's try with a higher resolution image
The convolutional neural network is size agnostic because we are using global average pooling, although the full image couldn't be loaded because of the lack of memory, this is why we used patches instead, that can latter be stiched together


```python
market = np.array(Image.open("market.jpg"))
h,w,c = market.shape
market_top_left=market[:h//2, :w//2, :]
market_top_right=market[:h//2, w//2:, :]
market_bottom_left=market[h//2:, :w//2, :]
market_bottom_right=market[h//2:, w//2:, :]
```


```python
market = transforms.ToTensor()(market_top_left)
market = normalize(market)
```


```python

```


```python
for market_patch in [market_bottom_left, market_bottom_right, market_top_left, market_top_right]:
    market = transforms.ToTensor()(market_patch)
    market = normalize(market)
    visualise_top_k(deconv, market, "kiwi", device, classnames, 5, (2, 4), figsize=(30 ,10))
```


    
![png](output_16_0.png)
    



    
![png](output_16_1.png)
    



    
![png](output_16_2.png)
    



    
![png](output_16_3.png)
    



    
![png](output_16_4.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](output_16_6.png)
    



    
![png](output_16_7.png)
    



    
![png](output_16_8.png)
    



    
![png](output_16_9.png)
    



    
![png](output_16_10.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](output_16_12.png)
    



    
![png](output_16_13.png)
    



    
![png](output_16_14.png)
    



    
![png](output_16_15.png)
    



    
![png](output_16_16.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](output_16_18.png)
    



    
![png](output_16_19.png)
    



    
![png](output_16_20.png)
    



    
![png](output_16_21.png)
    



    
![png](output_16_22.png)
    



    <Figure size 432x288 with 0 Axes>



```python
# market = transforms.ToTensor()(market_top_right)
# market = normalize(market)
# visualise_top_k(model, market, "kiwi", device, classnames, 10, (1, 5))
```


```python
# market = transforms.ToTensor()(market_bottom_left)
# market = normalize(market)
# visualise_top_k(model, market, "kiwi", device, classnames, 10, (1, 5))
```


```python

```

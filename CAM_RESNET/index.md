```python
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 19:43:53 2021

@author: Mounib Benimam
"""
%load_ext autoreload
%autoreload 2
from utils import loadDataset, plot_report, test, plot_batch, unormalize, unormalize_img, plot_imgs, visualise_top_k             
from models.VisualisableModel import VggCam, DeconvolutionVisualisationModel, ResnetCam
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

model = ResnetCam(classnames)
model.load_from_file("models/ResnetCamFruitsWeights.pk")
model = model.to(device)
yhat, y = test(model, dataloaders["test"], datasizes["test"])
plot_report(y, yhat, classnames)
```

    Using GPU :D !
    Using GPU :D !
    iteration 0
    iteration 1
    iteration 2
    iteration 3
    iteration 4
    iteration 5
    iteration 6
    iteration 7
    iteration 8
    iteration 9
    iteration 10
    iteration 11
    


    
![png](output_3_1.png)
    



```python
model2 = VggCam(classnames)
model2.load_from_file("models/VggCamFruitsWeights.pk")
model2 = model2.to(device)
yhat, y = test(model2, dataloaders["test"], datasizes["test"])
plot_report(y, yhat, classnames)
```

    Using GPU :D !
    iteration 0
    iteration 1
    iteration 2
    iteration 3
    iteration 4
    iteration 5
    iteration 6
    iteration 7
    iteration 8
    iteration 9
    iteration 10
    iteration 11
    


    
![png](output_4_1.png)
    



```python
batch, labels = next(iter(dataloaders["train"]))
```


```python
# deconv1 = DeconvolutionVisualisationModel(model)
# deconv2 = DeconvolutionVisualisationModel(model2)
```


```python
# b, c, h, w = batch.shape
# m = deconv.generateMap(batch[0].reshape(1, c, h, w).to(device), labels[0], device)
```


```python
torch.cuda.empty_cache()
```


```python
from IPython.core.display import HTML
```


```python
for s in range(32):
    img = batch[s]
    display(HTML("<h2>Resultat avec VGG16</h2>"))
    visualise_top_k(model2, batch[s], labels[s], device, classnames, 1, subplots=(1, 5), figsize=(20, 10))
    
    display(HTML("<h2>Resultat avec Resnet</h2>"))
    visualise_top_k(model, batch[s], labels[s], device, classnames, 1, subplots=(1, 5), figsize=(20, 10))
    
```


<h2>Resultat avec VGG16</h2>



    
![png](output_10_1.png)
    



<h2>Resultat avec Resnet</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_4.png)
    



<h2>Resultat avec VGG16</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_7.png)
    



<h2>Resultat avec Resnet</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_10.png)
    



<h2>Resultat avec VGG16</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_13.png)
    



<h2>Resultat avec Resnet</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_16.png)
    



<h2>Resultat avec VGG16</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_19.png)
    



<h2>Resultat avec Resnet</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_22.png)
    



<h2>Resultat avec VGG16</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_25.png)
    



<h2>Resultat avec Resnet</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_28.png)
    



<h2>Resultat avec VGG16</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_31.png)
    



<h2>Resultat avec Resnet</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_34.png)
    



<h2>Resultat avec VGG16</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_37.png)
    



<h2>Resultat avec Resnet</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_40.png)
    



<h2>Resultat avec VGG16</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_43.png)
    



<h2>Resultat avec Resnet</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_46.png)
    



<h2>Resultat avec VGG16</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_49.png)
    



<h2>Resultat avec Resnet</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_52.png)
    



<h2>Resultat avec VGG16</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_55.png)
    



<h2>Resultat avec Resnet</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_58.png)
    



<h2>Resultat avec VGG16</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_61.png)
    



<h2>Resultat avec Resnet</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_64.png)
    



<h2>Resultat avec VGG16</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_67.png)
    



<h2>Resultat avec Resnet</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_70.png)
    



<h2>Resultat avec VGG16</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_73.png)
    



<h2>Resultat avec Resnet</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_76.png)
    



<h2>Resultat avec VGG16</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_79.png)
    



<h2>Resultat avec Resnet</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_82.png)
    



<h2>Resultat avec VGG16</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_85.png)
    



<h2>Resultat avec Resnet</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_88.png)
    



<h2>Resultat avec VGG16</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_91.png)
    



<h2>Resultat avec Resnet</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_94.png)
    



<h2>Resultat avec VGG16</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_97.png)
    



<h2>Resultat avec Resnet</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_100.png)
    



<h2>Resultat avec VGG16</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_103.png)
    



<h2>Resultat avec Resnet</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_106.png)
    



<h2>Resultat avec VGG16</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_109.png)
    



<h2>Resultat avec Resnet</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_112.png)
    



<h2>Resultat avec VGG16</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_115.png)
    



<h2>Resultat avec Resnet</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_118.png)
    



<h2>Resultat avec VGG16</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_121.png)
    



<h2>Resultat avec Resnet</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_124.png)
    



<h2>Resultat avec VGG16</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_127.png)
    



<h2>Resultat avec Resnet</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_130.png)
    



<h2>Resultat avec VGG16</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_133.png)
    



<h2>Resultat avec Resnet</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_136.png)
    



<h2>Resultat avec VGG16</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_139.png)
    



<h2>Resultat avec Resnet</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_142.png)
    



<h2>Resultat avec VGG16</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_145.png)
    



<h2>Resultat avec Resnet</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_148.png)
    



<h2>Resultat avec VGG16</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_151.png)
    



<h2>Resultat avec Resnet</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_154.png)
    



<h2>Resultat avec VGG16</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_157.png)
    



<h2>Resultat avec Resnet</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_160.png)
    



<h2>Resultat avec VGG16</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_163.png)
    



<h2>Resultat avec Resnet</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_166.png)
    



<h2>Resultat avec VGG16</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_169.png)
    



<h2>Resultat avec Resnet</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_172.png)
    



<h2>Resultat avec VGG16</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_175.png)
    



<h2>Resultat avec Resnet</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_178.png)
    



<h2>Resultat avec VGG16</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_181.png)
    



<h2>Resultat avec Resnet</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_184.png)
    



<h2>Resultat avec VGG16</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_187.png)
    



<h2>Resultat avec Resnet</h2>



    <Figure size 432x288 with 0 Axes>



    
![png](output_10_190.png)
    



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

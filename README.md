# IRCNN

This repository includes codes and dataset for "IRCNN：An Irregular-Time-Distanced Recurrent Convolutional Neural Network for Change Detection in Satellite Time Series", which has been published in IEEE Geoscience and Remote Sensing Letters.    

Paper Link: https://ieeexplore.ieee.org/document/9721897  
Paper DOI: 10.1109/LGRS.2022.3154894

The materials in this repository are only for study and research, NOT FOR COMMERCIAL USE. **Please cite this paper if it is helpful for you**.
***

### Requirements： 
 ```
　cuda 10.0  
　numpy 1.19  
　pandas 1.1  
　python 3.6  
　pytorch 1.8.1  
  ```

### Data preparation  
The shape of the input data :  (T, B, C, H, W)  
　T is the length of Time  
　B is the Batch size  
　C is the Channels  
　H is the High of the image patch  
　W is the Width of the image patch  
 
### Train
python train.py  
Note there is a config file named Parameter.yaml  

### Test
python test.py  

### Data Download  
Remote sensing images (Landast7/8) and ground truth map of five study areas:  
Access from Baidu Cloud  
Link: https://pan.baidu.com/s/1UhE0N2GcybyjZdIYb8TlGQ  
Password: wdzm  
Access from Google Drive  
Link: https://drive.google.com/file/d/1tek-QOl1IWoebtJbyUyA0zKbo1x0V9wD/view?usp=sharing

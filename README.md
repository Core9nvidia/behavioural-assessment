# Real_time_emotion_detection
This project aims to determine the emotion on a person's face in real time into one of seven categories, using deep convolutional neural networks. The model is trained on the FER-2013 dataset which was published on International Conference on Machine Learning (ICML). This dataset consists of 35887 grayscale, 48x48 sized face images with seven emotions - angry, disgusted, fearful, happy, neutral, sad and surprised.


### Dataset
The original FER-2013 data set can be downloaded [here](https://drive.google.com/file/d/1X60B-uR3NtqPd4oosdotpbDgy8KOfUdr/view).

To train the model - 
Unzip the data into a folder containing dataset_prepare.py and run the python script. \
This will create four ```.npy``` files in the directory namely 'X_train', 'X_test', 'Y_train' and 'Y_test'. \
These are concatenated and shuffled numpy arrays of shapes - \
X_train - (#training_examples, 48, 48, 3) \
X_test - (#testing_examples, 48, 48, 3) \
Y_train - (#training_examples, 1) \
Y_test - (#testing_examples, 1) \
The arrays can be loaded by -  \
```python
import numpy as np
X_train = np.load('X_train.npy')
```
The Original Model was trained in Google colab, the notebook ```emotion_detection.ipynb``` contains the code for training.

### Model Summary
![sum](https://github.com/Varun221/Real_time_emotion_detection/blob/master/images/model_summary.png)
 
The model was trained for 30 epochs with results - 
Model: "DCNN"
_______________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 48, 48, 64)        4864      
_______________________
batchnorm_1 (BatchNormalizat (None, 48, 48, 64)        256       
_______________________
conv2d_2 (Conv2D)            (None, 48, 48, 64)        102464    
_______________________
batchnorm_2 (BatchNormalizat (None, 48, 48, 64)        256       
_______________________
maxpool2d_1 (MaxPooling2D)   (None, 24, 24, 64)        0         
_______________________
dropout_1 (Dropout)          (None, 24, 24, 64)        0         
_______________________
conv2d_3 (Conv2D)            (None, 24, 24, 128)       73856     
_______________________
batchnorm_3 (BatchNormalizat (None, 24, 24, 128)       512       
_______________________
conv2d_4 (Conv2D)            (None, 24, 24, 128)       147584    
_______________________
batchnorm_4 (BatchNormalizat (None, 24, 24, 128)       512       
_______________________
maxpool2d_2 (MaxPooling2D)   (None, 12, 12, 128)       0         
_______________________
dropout_2 (Dropout)          (None, 12, 12, 128)       0         
_______________________
conv2d_5 (Conv2D)            (None, 12, 12, 256)       295168    
_______________________
batchnorm_5 (BatchNormalizat (None, 12, 12, 256)       1024      
_______________________
conv2d_6 (Conv2D)            (None, 12, 12, 256)       590080    
_______________________
batchnorm_6 (BatchNormalizat (None, 12, 12, 256)       1024      
_______________________
maxpool2d_3 (MaxPooling2D)   (None, 6, 6, 256)         0         
_______________________
dropout_3 (Dropout)          (None, 6, 6, 256)         0         
_______________________
flatten (Flatten)            (None, 9216)              0         
_______________________
dense_1 (Dense)              (None, 128)               1179776   
_______________________
batchnorm_7 (BatchNormalizat (None, 128)               512       
_______________________
dropout_4 (Dropout)          (None, 128)               0         
_______________________
out_layer (Dense)            (None, 5)                 645       
=================================================================
Total params: 2,398,533
Trainable params: 2,396,485
Non-trainable params: 2,048
================================================================

![gg](https://user-images.githubusercontent.com/91504747/203370501-64fe3397-0afd-4a59-90cf-1336dee0af9a.jpeg)
![CNN](https://user-images.githubusercontent.com/91504747/203370777-677bf65d-5309-4e66-82d5-a9eed564d140.jpeg)


Further training resulted in overfitting, hence the training was stopped early. You can experiment with the model and its hyper params in the notebook.

The trained model is given in hdf5 format in ```models``` as well ```code``` folder.
You can load the model in your own script by - 
```python
import tensorflow as tf
model = tf.keras.models.load_model('<path_to_model>/my_model.h5')
```


### Algorithm
1. The face of the person in the feed is predicted by Haar Cascade's algorithm.
2. The Model then takes in the image and outputs a set of softmax scores for each emotion
3. The emotion with maximum softmax score is given as the person's emotion.

The final Result - \
![image](https://user-images.githubusercontent.com/91504747/203371112-2c5e3408-b23f-45b0-a45a-1e1f52ead9a3.png)


### References
The basic architecture of the model was inspired from the research paper, Emotion Recognition using Deep Convolutional Neural Networks by Enrique Correa, Arnoud Jonker, Michaël Ozo and Rob Stolk

Suggestions and Contributions are always welcome:)

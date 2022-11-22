import numpy as np
import os
import cv2
from tqdm.auto import tqdm


# emotion list contains the list of emotions
train_path = './Data/archive/train'
test_path = './Data/archive/test'

# this dictionary maps emotion to the index

emotions  = ['angry', 'happy', 'neutral', 'sad', 'surprise']

emotion_to_index = dict([(k,i) for i,k in enumerate(emotions)])
 

# image paths contain the path to train 
image_paths = []
for emotion in emotions:
    image_paths.append(os.path.join(train_path, emotion))
    
# test paths contain the path to test 
test_paths = []
for emotion in emotions:
    test_paths.append(os.path.join(test_path, emotion))
    
    

# copying the image arrays in the list    
print("\n Retrieving images from train...")
X = []
Y = []
for image_path in image_paths:
    images = os.listdir(image_path)
    num = len(images)
    print(f"\nFound {num} images in {image_path}")
    for i in tqdm(range(num)):
        path = os.path.join(image_path, images[i])
        img =  cv2.imread(path)
        X.append(img)
        Y.append(emotion_to_index[os.path.split(image_path)[1]])
        

# converting the list to array
X_train = np.array(X)
Y_train = np.array(Y)
shuffled_X = X_train.copy()
shuffled_Y = Y_train.copy()

# The arrays are shuffled
print("\n Shuffling train images ...")
indices = np.arange(X_train.shape[0])
np.random.shuffle(indices)
for i in range(X_train.shape[0]):
    shuffled_X[i, :, :, :] = X_train[indices[i], :, :, :]
    shuffled_Y[i] = Y_train[indices[i]]
    
Y_fin_train = np.reshape(shuffled_Y, (-1,1))
print("\n Saving the training arrays...")
np.save('Data/Y_train.npy', Y_fin_train)
np.save('Data/X_train.npy', shuffled_X)


# test images are retrieved in the form of a list
Xt = []
Yt = []
for test_path in test_paths:
    images = os.listdir(test_path)
    num = len(images)
    print(f"\nFound {num} images in {test_path}")
    for i in tqdm(range(num)):
        path = os.path.join(test_path, images[i])
        img =  cv2.imread(path)
        Xt.append(img)
        Yt.append(emotion_to_index[os.path.split(test_path)[1]])
        

X_ta = np.array(Xt)
Y_ta = np.array(Yt)
        
# shuffling test arrays
shuffled_Xt = X_ta.copy()
shuffled_Yt = Y_ta.copy()

print("\n Shuffling test arrays ...")
indices = np.arange(X_ta.shape[0])
np.random.shuffle(indices)
for i in range(X_ta.shape[0]):
    shuffled_Xt[i, :, :, :] = X_ta[indices[i], :, :, :]
    shuffled_Yt[i] = Y_ta[indices[i]]
        

        
Y_test = np.reshape(shuffled_Yt, (-1,1))
X_test = shuffled_Xt

# saving test arrays
print("\n Saving test arrays...")
np.save('Data/Y_test.npy', Y_test)
np.save('Data/X_test.npy', X_test)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 11:31:19 2020

@author: ali hussain
"""


from keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

test_datagen=ImageDataGenerator(rescale=1./255)

training_set=train_datagen.flow_from_directory('D:/mystuff/cnn/data_set/training_set',target_size=(64,64),batch_size=32,class_mode='binary')

test_set=test_datagen.flow_from_directory('D:/mystuff/cnn/data_set/test_set',target_size=(64,64),batch_size=32,class_mode='binary')

from keras.models import Sequential

from keras.layers import Convolution2D

from keras.layers import MaxPooling2D
 
from keras.layers import Flatten

from keras.layers import Dense


classifier=Sequential()

classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(32,3,3,activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

classifier.add(Dense(output_dim=128,activation='relu'))

classifier.add(Dense(output_dim=128,activation='relu'))

classifier.add(Dense(output_dim=1,activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit_generator(training_set,samples_per_epoch=8000,nb_epoch=25,validation_data=test_set,nb_val_samples=2000)

from keras.models import load_model
classifier.save('D:/mustuff/cnn/my_model_cnn.h5')

model=load_model('D:/mystuff/cnn/my_model_cnn.h5')

################################### error because not fit dataset
import numpy as np

from keras.preprocessing import image

test_image=image.load_img('D:/mystuff/cnn/data_set/single_prediction/dog.59.jpg',target_size=(64,64))

print(test_image)

test_image=image.img_to_array(test_image)
print(test_image)

print(test_image[0,0,:].size)

test_image=np.expand_dims(test_image,axis=0)
print(np.ndim(test_image))

print(np.shape(test_image))

result=model.predict(test_image)
print("Prediction ",result)

if result>0.5:
    result=1
elif result<0.5:
    result=0
else:
    print("Not Predictable")
    

print(result)

print(training_set.class_indices)

if (result==1):
    print("Given image is a Dog")
elif result==0:
    print("Given image is a Cat")
else:
    print("Given image is neither Dog nor Cat ")
    
    


"""
A local file was found, but it seems to be incomplete or outdated because the auto file hash does not match the original value of 8a61469f7ea1b51cbae51d4f78837e45 so we will re-download the data.
Downloading data from https://s3.amazonaws.com/img-datasets/mnist.npz
11493376/11490434 [==============================] - 30s 3us/step
WARNING: Logging before flag parsing goes to stderr.
W0124 15:21:38.912694  6488 deprecation_wrapper.py:119] From C:\Users\ali hussain\Anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:4070: The name tf.nn.max_pool is deprecated. Please use tf.nn.max_pool2d instead.

x_train shape: (60000, 28, 28, 1)
60000 train samples
10000 test samples
W0124 15:21:39.558722  6488 deprecation_wrapper.py:119] From C:\Users\ali hussain\Anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 60000 samples, validate on 10000 samples
Epoch 1/12
60000/60000 [==============================] - 82s 1ms/step - loss: 0.2665 - accuracy: 0.9172 - val_loss: 0.0603 - val_accuracy: 0.9819
Epoch 2/12
60000/60000 [==============================] - 85s 1ms/step - loss: 0.0932 - accuracy: 0.9726 - val_loss: 0.0422 - val_accuracy: 0.9861
Epoch 3/12
60000/60000 [==============================] - 86s 1ms/step - loss: 0.0691 - accuracy: 0.9797 - val_loss: 0.0373 - val_accuracy: 0.9878
Epoch 4/12
60000/60000 [==============================] - 86s 1ms/step - loss: 0.0555 - accuracy: 0.9836 - val_loss: 0.0336 - val_accuracy: 0.9887
Epoch 5/12
60000/60000 [==============================] - 86s 1ms/step - loss: 0.0497 - accuracy: 0.9848 - val_loss: 0.0323 - val_accuracy: 0.9891
Epoch 6/12
60000/60000 [==============================] - 86s 1ms/step - loss: 0.0429 - accuracy: 0.9872 - val_loss: 0.0370 - val_accuracy: 0.9900
Epoch 7/12
60000/60000 [==============================] - 85s 1ms/step - loss: 0.0386 - accuracy: 0.9884 - val_loss: 0.0315 - val_accuracy: 0.9890
Epoch 8/12
60000/60000 [==============================] - 85s 1ms/step - loss: 0.0357 - accuracy: 0.9890 - val_loss: 0.0314 - val_accuracy: 0.9908
Epoch 9/12
60000/60000 [==============================] - 87s 1ms/step - loss: 0.0319 - accuracy: 0.9901 - val_loss: 0.0275 - val_accuracy: 0.9904
Epoch 10/12
60000/60000 [==============================] - 109s 2ms/step - loss: 0.0305 - accuracy: 0.9904 - val_loss: 0.0278 - val_accuracy: 0.9909
Epoch 11/12
60000/60000 [==============================] - 110s 2ms/step - loss: 0.0303 - accuracy: 0.9907 - val_loss: 0.0245 - val_accuracy: 0.9922
Epoch 12/12
60000/60000 [==============================] - 112s 2ms/step - loss: 0.0273 - accuracy: 0.9914 - val_loss: 0.0261 - val_accuracy: 0.9921
Test loss: 0.026109962715069195
Test accuracy: 0.9921000003814697

runfile('C:/Users/ali hussain/.spyder-py3/cnn.py', wdir='C:/Users/ali hussain/.spyder-py3')
Found 6798 images belonging to 2 classes.
Found 1207 images belonging to 2 classes.
C:/Users/ali hussain/.spyder-py3/cnn.py:32: UserWarning: Update your `Conv2D` call to the Keras 2 API: `Conv2D(32, (3, 3), input_shape=(64, 64, 3..., activation="relu")`
  classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))
C:/Users/ali hussain/.spyder-py3/cnn.py:36: UserWarning: Update your `Conv2D` call to the Keras 2 API: `Conv2D(32, (3, 3), activation="relu")`
  classifier.add(Convolution2D(32,3,3,activation='relu'))
C:/Users/ali hussain/.spyder-py3/cnn.py:42: UserWarning: Update your `Dense` call to the Keras 2 API: `Dense(activation="relu", units=128)`
  classifier.add(Dense(output_dim=128,activation='relu'))
C:/Users/ali hussain/.spyder-py3/cnn.py:44: UserWarning: Update your `Dense` call to the Keras 2 API: `Dense(activation="relu", units=128)`
  classifier.add(Dense(output_dim=128,activation='relu'))
C:/Users/ali hussain/.spyder-py3/cnn.py:46: UserWarning: Update your `Dense` call to the Keras 2 API: `Dense(activation="sigmoid", units=1)`
  classifier.add(Dense(output_dim=1,activation='sigmoid'))
W0124 16:35:38.850757  6488 deprecation.py:323] From C:\Users\ali hussain\Anaconda3\lib\site-packages\tensorflow\python\ops\nn_impl.py:180: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
C:/Users/ali hussain/.spyder-py3/cnn.py:50: UserWarning: The semantics of the Keras 2 argument `steps_per_epoch` is not the same as the Keras 1 argument `samples_per_epoch`. `steps_per_epoch` is the number of batches to draw from the generator at each epoch. Basically steps_per_epoch = samples_per_epoch/batch_size. Similarly `nb_val_samples`->`validation_steps` and `val_samples`->`steps` arguments have changed. Update your method calls accordingly.
  classifier.fit_generator(training_set,samples_per_epoch=8000,nb_epoch=25,validation_data=test_set,nb_val_samples=2000)
C:/Users/ali hussain/.spyder-py3/cnn.py:50: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<keras.pre..., validation_data=<keras.pre..., steps_per_epoch=250, epochs=25, validation_steps=2000)`
  classifier.fit_generator(training_set,samples_per_epoch=8000,nb_epoch=25,validation_data=test_set,nb_val_samples=2000)
Epoch 1/25
250/250 [==============================] - 529s 2s/step - loss: 0.6733 - accuracy: 0.5793 - val_loss: 0.6716 - val_accuracy: 0.6257
Epoch 2/25
250/250 [==============================] - 446s 2s/step - loss: 0.6277 - accuracy: 0.6523 - val_loss: 0.7183 - val_accuracy: 0.6761
Epoch 3/25
250/250 [==============================] - 423s 2s/step - loss: 0.5875 - accuracy: 0.6882 - val_loss: 0.5414 - val_accuracy: 0.6866
Epoch 4/25
250/250 [==============================] - 350s 1s/step - loss: 0.5639 - accuracy: 0.7032 - val_loss: 0.6145 - val_accuracy: 0.7440
Epoch 5/25
250/250 [==============================] - 405s 2s/step - loss: 0.5198 - accuracy: 0.7421 - val_loss: 0.6112 - val_accuracy: 0.7665
Epoch 6/25
250/250 [==============================] - 400s 2s/step - loss: 0.4851 - accuracy: 0.7621 - val_loss: 0.3544 - val_accuracy: 0.7753
Epoch 7/25
250/250 [==============================] - 448s 2s/step - loss: 0.4638 - accuracy: 0.7780 - val_loss: 0.4146 - val_accuracy: 0.7672
Epoch 8/25
250/250 [==============================] - 417s 2s/step - loss: 0.4461 - accuracy: 0.7895 - val_loss: 0.4548 - val_accuracy: 0.7729
Epoch 9/25
250/250 [==============================] - 211s 844ms/step - loss: 0.4246 - accuracy: 0.7987 - val_loss: 0.5711 - val_accuracy: 0.7798
Epoch 10/25
250/250 [==============================] - 197s 789ms/step - loss: 0.4054 - accuracy: 0.8096 - val_loss: 0.3865 - val_accuracy: 0.7768
Epoch 11/25
250/250 [==============================] - 223s 892ms/step - loss: 0.3854 - accuracy: 0.8197 - val_loss: 0.3778 - val_accuracy: 0.7582
Epoch 12/25
250/250 [==============================] - 248s 992ms/step - loss: 0.3728 - accuracy: 0.8284 - val_loss: 0.4928 - val_accuracy: 0.7383
Epoch 13/25
250/250 [==============================] - 229s 916ms/step - loss: 0.3409 - accuracy: 0.8488 - val_loss: 0.4866 - val_accuracy: 0.7963
Epoch 14/25
250/250 [==============================] - 216s 866ms/step - loss: 0.3324 - accuracy: 0.8527 - val_loss: 0.2728 - val_accuracy: 0.7913
Epoch 15/25
250/250 [==============================] - 210s 840ms/step - loss: 0.3199 - accuracy: 0.8568 - val_loss: 0.5760 - val_accuracy: 0.7867
Epoch 16/25
250/250 [==============================] - 201s 806ms/step - loss: 0.3065 - accuracy: 0.8648 - val_loss: 0.3786 - val_accuracy: 0.7814
Epoch 17/25
250/250 [==============================] - 204s 815ms/step - loss: 0.2949 - accuracy: 0.8745 - val_loss: 1.0155 - val_accuracy: 0.7448
Epoch 18/25
250/250 [==============================] - 200s 801ms/step - loss: 0.2822 - accuracy: 0.8789 - val_loss: 0.8237 - val_accuracy: 0.7687
Epoch 19/25
250/250 [==============================] - 199s 797ms/step - loss: 0.2605 - accuracy: 0.8870 - val_loss: 0.4594 - val_accuracy: 0.7648
Epoch 20/25
250/250 [==============================] - 3141s 13s/step - loss: 0.2351 - accuracy: 0.9063 - val_loss: 0.5337 - val_accuracy: 0.7738
Epoch 21/25
250/250 [==============================] - 376s 2s/step - loss: 0.2329 - accuracy: 0.9063 - val_loss: 0.9640 - val_accuracy: 0.7722
Epoch 22/25
250/250 [==============================] - 398s 2s/step - loss: 0.2066 - accuracy: 0.9166 - val_loss: 0.9729 - val_accuracy: 0.7772
Epoch 23/25
250/250 [==============================] - 342s 1s/step - loss: 0.2102 - accuracy: 0.9128 - val_loss: 0.5991 - val_accuracy: 0.7655
Epoch 24/25
250/250 [==============================] - 443s 2s/step - loss: 0.1943 - accuracy: 0.9227 - val_loss: 0.6456 - val_accuracy: 0.7738
Epoch 25/25
250/250 [==============================] - 386s 2s/step - loss: 0.1791 - accuracy: 0.9292 - val_loss: 0.3335 - val_accuracy: 0.7579
Traceback (most recent call last):

  File "<ipython-input-3-e74d89804675>", line 1, in <module>
    runfile('C:/Users/ali hussain/.spyder-py3/cnn.py', wdir='C:/Users/ali hussain/.spyder-py3')

  File "C:\Users\ali hussain\Anaconda3\lib\site-packages\spyder_kernels\customize\spydercustomize.py", line 786, in runfile
    execfile(filename, namespace)

  File "C:\Users\ali hussain\Anaconda3\lib\site-packages\spyder_kernels\customize\spydercustomize.py", line 110, in execfile
    exec(compile(f.read(), filename, 'exec'), namespace)

  File "C:/Users/ali hussain/.spyder-py3/cnn.py", line 53, in <module>
    classifier.save('D:/mustuff/cnn/my_model_cnn.h5')

  File "C:\Users\ali hussain\Anaconda3\lib\site-packages\keras\engine\network.py", line 1152, in save
    save_model(self, filepath, overwrite, include_optimizer)

  File "C:\Users\ali hussain\Anaconda3\lib\site-packages\keras\engine\saving.py", line 449, in save_wrapper
    save_function(obj, filepath, overwrite, *args, **kwargs)

  File "C:\Users\ali hussain\Anaconda3\lib\site-packages\keras\engine\saving.py", line 540, in save_model
    with H5Dict(filepath, mode='w') as h5dict:

  File "C:\Users\ali hussain\Anaconda3\lib\site-packages\keras\utils\io_utils.py", line 191, in __init__
    self.data = h5py.File(path, mode=mode)

  File "C:\Users\ali hussain\Anaconda3\lib\site-packages\h5py\_hl\files.py", line 394, in __init__
    swmr=swmr)

  File "C:\Users\ali hussain\Anaconda3\lib\site-packages\h5py\_hl\files.py", line 176, in make_fid
    fid = h5f.create(name, h5f.ACC_TRUNC, fapl=fapl, fcpl=fcpl)

  File "h5py\_objects.pyx", line 54, in h5py._objects.with_phil.wrapper

  File "h5py\_objects.pyx", line 55, in h5py._objects.with_phil.wrapper

  File "h5py\h5f.pyx", line 105, in h5py.h5f.create

OSError: Unable to create file (unable to open file: name = 'D:/mustuff/cnn/my_model_cnn.h5', errno = 2, error message = 'No such file or directory', flags = 13, o_flags = 302)



"""













    
    
    





















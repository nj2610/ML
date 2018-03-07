# 1.Building the Convolution neural network

#importing the necessary keras packages
from keras.models import Sequential
from keras.layers import Convolution2D    #Used for convolution step
from keras.layers import MaxPooling2D     #Used for pooling step
from keras.layers import Flatten          #Used for flattening before sending to ANN
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

'''Arguements :
    1.nb_filters = no of filters to be used, 2.No oof rows of feature detector 3. No of columns in feature detector
    4.Input shape:dimension of image it has three arguements first and second is pixels third is channels 
    5.activation fn'''
# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

'''Additional layer is added for improving accuracy and to further improve accuracy we can increase input size.'''
# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
'''This is image augmentation and it is done to avoid overfitting.
it manipulates images in different forms so that we have variety to train.'''
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 2000)
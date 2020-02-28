from keras.models import Model, load_model, Sequential
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
from keras import optimizers , losses , activations , callbacks

HW_SIZE=(80,32)
MAX_THUMB_SIZE=(32,80)
NUM_TEAMS = 4

from ImageStandardizer import ImageStandardizer
class TeamRecognizer :

    def __init__( self, max_thumb_size, model=None ):
        self.max_thumb_size = max_thumb_size
        self.input_shape = (self.max_thumb_size[1], self.max_thumb_size[0], 3)    
        self.imageStandardizer = ImageStandardizer(self.max_thumb_size)
        self.class_indices = {}
        
        if model:
            self._model = model
            #TODO: 
            return
        
        self._model = self.basic_net()

        self._model.compile( loss=losses.binary_crossentropy , optimizer='rmsprop', metrics=['accuracy'])
        #self._model.compile(optimizer='sgd', loss=losses.binary_crossentropy, metrics=['accuracy'])
    def basic_net(self):
        cnn = Sequential()
        cnn.add(Conv2D(filters=32, 
                       kernel_size=(2,2), 
                       strides=(1,1),
                       padding='same',
                       input_shape=  self.input_shape,
                       data_format='channels_last'))
        cnn.add(Activation('relu'))
        cnn.add(MaxPooling2D(pool_size=(2,2),
                             strides=2))
        cnn.add(Conv2D(filters=64,
                       kernel_size=(2,2),
                       strides=(1,1),
                       padding='valid'))
        cnn.add(Activation('relu'))
        cnn.add(MaxPooling2D(pool_size=(2,2), strides=2))
        cnn.add(Flatten())        
        cnn.add(Dense(64))
        cnn.add(Activation('relu'))
        cnn.add(Dropout(0.1))
        cnn.add(Dense(NUM_TEAMS))
        cnn.add(Activation('softmax'))     
        return cnn
    
    def fit_generator(self, train_generator = None , validation_datagen=None, epochs=10, **kwargs  ):
        if not train_generator:
            train_datagen = ImageDataGenerator(
                rotation_range = 10,                  
                width_shift_range = 2,                  
                height_shift_range = 2,                  
                rescale = 1./255,                                   
                zoom_range = 0.01,                     
                horizontal_flip = True)
            train_generator = train_datagen.flow_from_directory(
                IMAGE_DIR,
                target_size=HW_SIZE,
                class_mode='categorical',
                batch_size = BATCH_SIZE)
            
        if not validation_datagen:
            validation_datagen = ImageDataGenerator(rescale = 1./255)    
            validation_generator = validation_datagen.flow_from_directory(
                VAL_DATA_DIR,
                target_size=HW_SIZE,
                class_mode='categorical',
                batch_size = BATCH_SIZE)
        import time
        start = time.time()
        cnn.fit_generator(
            train_generator,
            steps_per_epoch=NB_TRAIN_IMG//BATCH_SIZE,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=NB_VALID_IMG//BATCH_SIZE)
        end = time.time()
        print('Fiiting time:',(end - start)/60)
    def predict(self, crop_images):
        shape = (len(crop_images),) + HW_SIZE +(3,)
        res = np.empty(shape)
        i=0
        for crop_image, box in crop_images:
            res[i] = self.imageStandardizer.resize_array(crop_image)/255
            i+=1
            
        return np.argmax(self._model.predict(res), axis=-1)
            
    def save_model(self , file_path ='team_model.h5'):
        self._model.save(file_path )
    def load_model(self , file_path ='team_model.h5'):
        self._model = load_model(file_path)
        
  
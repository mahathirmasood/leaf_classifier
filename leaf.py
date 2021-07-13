from tensorflow.keras.layers import Conv2D,Dense,Flatten,MaxPool2D
from tensorflow.keras import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
train_data_dir = 'E:\leaf\Train\Train'
validation_data_dir = 'E:\leaf\Validation\Validation'
test_data_dir = 'E:\leaf\Test\Test'

data_gen = ImageDataGenerator(rescale=1./255)
train_gen = data_gen.flow_from_directory(train_data_dir,batch_size=16)
val_gen = data_gen.flow_from_directory(validation_data_dir,batch_size=16)
test_gen = data_gen.flow_from_directory(test_data_dir,batch_size=16)

def model():
    model = Sequential(name='leaf_classifier')

    model.add(Conv2D(32, kernel_size=(3, 3), padding='valid'))
    model.add(MaxPool2D(pool_size=(3, 3)))
    model.add(Conv2D(16,kernel_size=(3, 3), padding='valid'))
    model.add(MaxPool2D(pool_size=(3,3)))
    model.add(Conv2D(8, kernel_size=(3, 3), padding='valid'))
    model.add(MaxPool2D(pool_size=(3, 3)))


    model.add(Flatten())
    model.add(Dense(32,activation='sigmoid'))
    model.add(Dense(3,activation='softmax'))
    opt = optimizers.Adam(learning_rate=.0005)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
    return model
model = model()
callbacks = [EarlyStopping(patience=4),ModelCheckpoint(filepath='E:\leaf\model2',save_best_only=True)]
model.fit(train_gen,batch_size=16,epochs=20,validation_data=val_gen,callbacks=callbacks)
model.summary()
model.evaluate(test_gen)


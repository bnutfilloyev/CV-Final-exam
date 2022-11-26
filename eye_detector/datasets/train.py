import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import (Activation, Conv2D, Dense, Flatten, Input,
                          MaxPooling2D)
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

x_train = np.load('models/x_train.npy').astype(np.float32)
y_train = np.load('models/y_train.npy').astype(np.float32)
x_val = np.load('models/x_val.npy').astype(np.float32)
y_val = np.load('models/y_val.npy').astype(np.float32)
print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2
)

val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow(x=x_train, y=y_train, batch_size=256, shuffle=True)

val_generator = val_datagen.flow(x=x_val, y=y_val, batch_size=256, shuffle=False)

inputs = Input(shape=(26, 34, 1))

net = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(inputs)
net = MaxPooling2D(pool_size=2)(net)

net = Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')(net)
net = MaxPooling2D(pool_size=2)(net)

net = Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu')(net)
net = MaxPooling2D(pool_size=2)(net)

net = Flatten()(net)

net = Dense(512)(net)
net = Activation('relu')(net)
net = Dense(1)(net)
outputs = Activation('sigmoid')(net)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.summary()

model_name = 'models/model_batch_256_oct_7.h5'

model.fit_generator(
    train_generator, epochs=100, validation_data=val_generator,
    callbacks=[
        ModelCheckpoint(model_name, monitor='val_acc', save_best_only=True, mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=10, verbose=1, mode='auto', min_lr=1e-05)
    ]
)

model = load_model(model_name)

y_pred = model.predict(x_val / 255.)
y_pred_logical = (y_pred > 0.5).astype(np.int64)

print('test acc: %s' % accuracy_score(y_val, y_pred_logical))
print('test f1: %s' % f1_score(y_val, y_pred_logical))
print('test auc: %s' % roc_auc_score(y_val, y_pred))

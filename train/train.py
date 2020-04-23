# %%

import datetime
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Activation, Conv2D, Flatten, Dense, MaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
plt.style.use('dark_background')
import shutil
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# %% Load Dataset

x_train = np.load('dataset/x_train.npy').astype(np.float32)
y_train = np.load('dataset/y_train.npy').astype(np.float32)
x_val = np.load('dataset/x_val.npy').astype(np.float32)
y_val = np.load('dataset/y_val.npy').astype(np.float32)

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)

# %% Preview

plt.subplot(2, 1, 1)
plt.title(str(y_train[0]))
plt.imshow(x_train[0].reshape((26, 34)), cmap='gray')
plt.subplot(2, 1, 2)
plt.title(str(y_val[4]))
plt.imshow(x_val[4].reshape((26, 34)), cmap='gray')

# %% Data Augmentation

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2)

val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow(x=x_train, y=y_train, batch_size=32, shuffle=True)

val_generator = val_datagen.flow(x=x_val, y=y_val, batch_size=32, shuffle=False)

# %% Build Model

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
net = Dense(2)(net)
outputs = Activation('sigmoid')(net)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.summary()

start_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
model_file = 'models/ckpt-%s.h5' % (start_time)

# %% Train

model.fit_generator(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    callbacks=[
        ModelCheckpoint(model_file, monitor='val_acc', save_best_only=True, mode='max', verbose=1),
        ReduceLROnPlateau(
            monitor='val_acc', factor=0.2, patience=10, verbose=1, mode='auto', min_lr=1e-05)
    ])

# %%

shutil.copyfile(model_file, 'models/latest.h5')

# %% Confusion Matrix

model = load_model(model_file)

y_pred = model.predict(x_val / 255.)
y_pred_logical = (y_pred[:, 0] > 0.5).astype(np.int)
print(y_val.shape)
print('test acc: %s' % accuracy_score(y_val[:, 0], y_pred_logical))
cm = confusion_matrix(y_val[:, 0], y_pred_logical)
sns.heatmap(cm, annot=True)

# Distribution of Prediction

ax = sns.distplot(y_pred[:, 0], kde=False)

# %%

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential

from keras.layers import Dense, Flatten, Dropout, Activation, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D

# VGG 卷积神经网络

# in VGG 3*3卷积核,激活函数为ReLU
# filters:输出空间的维度, 即卷积核(滤波器)的个数
# strides:卷积核沿宽度和高度方向滑动的步长
# padding:补洞策略 same or valid
def conv_block(layer, filters, kernel_size=(3,3), strides=(1,1),padding='same',name=None):
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               padding=padding,
               kernel_initializer="he_normal",
               name=name)(layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

input_shape = (224, 224, 3)

#Instantiate an empty model
model = Sequential([
    # Stage1 两层64个3*3卷积核的卷积层, 一个池化层
    Conv2D(64, (3, 3), input_shape=input_shape, padding='same', activation='relu'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

    # Stage2 两层128个3*3卷积核的卷积层, 一个池化层
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

    # Stage3 三层256个3*3卷积核的卷积层, 一个池化层
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

    # Stage4 三层512个3*3卷积核的卷积层, 一个池化层
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

    # Stage4 三层512个3*3卷积核的卷积层, 一个池化层
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

    # FC Layers 三层全连接(第一层直接使用Flatten),最后一层使用softmax归一化输出1000分类
    Flatten(),
    Dense(4096, activation='relu'),
    Dense(4096, activation='relu'),
    Dense(1000, activation='softmax')
])

model.summary()

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy"])







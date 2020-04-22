from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Dropout, ZeroPadding2D
from keras.models import Sequential, Model
from keras.optimizers import Adam


def VGG19(patch_size, nclasses):
    inputs = Input((patch_size, patch_size, 3))
    conv1 = Conv2D(64, (3, 3), activation = 'relu', padding='same')(inputs)
    conv2 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(conv1)
    maxpool1 = MaxPooling2D((2, 2), strides = (2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same')(maxpool1)
    conv4 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same')(conv3)
    maxpool2 = MaxPooling2D((2, 2), strides = (2, 2))(conv4)

    conv5 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same')(maxpool2)
    conv6 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same')(conv5)
    conv7 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same')(conv6)
    conv8 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same')(conv7)
    maxpool3 = MaxPooling2D((2, 2), strides = (2, 2))(conv8)

    conv9 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same')(maxpool3)
    conv10 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same')(conv9)
    conv11 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same')(conv10)
    conv12 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same')(conv11)
    maxpool4 = MaxPooling2D((2, 2), strides = (2, 2))(conv12)

    conv13 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same')(maxpool4)
    conv14 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same')(conv13)
    conv15 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same')(conv14)
    conv16 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same')(conv15)
    maxpool5 = MaxPooling2D((2, 2), strides = (2, 2))(conv16)

    flat = Flatten()(maxpool5)
    fcn1 = Dense(4096, activation = 'relu')(flat)
    dropout1 = Dropout(0.5)(fcn1)
    fcn2 = Dense(4096, activation = 'relu')(dropout1)
    dropout2 = Dropout(0.5)(fcn2)
    out = Dense(nclasses, activation = 'softmax')(dropout2)

    model = Model(input = inputs, output = out)

    model.compile(optimizer=Adam(lr = 1e-4), loss = 'categorical_crossentropy',  metrics=['accuracy'])
    model.summary()

    return model



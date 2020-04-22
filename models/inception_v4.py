from keras.layers import Conv2D, MaxPool2D, concatenate, BatchNormalization, Activation, Input, AvgPool2D, Flatten, Dropout, Dense
from keras.models import Model

class Inception:
    def base(self, shape, dropout, classes):

        input = Input(shape = shape)
        net = self.Stem(input)

        for i in range(4):
            net = self.InceptionA(net)
        net = self.ReductionA(net)

        for i in range(7):
            net = self.InceptionB(net)
        net=  self.ReductionB(net)

        for i in range(3):
            net = self.InceptionC(net)

        net = AvgPool2D((14, 14))(net)

        net = Dropout(dropout)(net)
        net = Flatten()(net)

        out = Dense(classes, activation = 'sigmoid')(net)

        model = Model(input, out)

        model.summary()

        return model





    def conv_bn(self, net, filter, k_row, k_column, padding = 'same', strides = (1, 1), use_bias = False):
        net = Conv2D(filters=filter, kernel_size = (k_row, k_column), strides = strides, padding = padding, use_bias = use_bias)(net)
        net = BatchNormalization(axis=-1, momentum=0.9997, scale=False)(net)
        net = Activation('relu')(net)
        return net
    def Stem(self,  net):
        net = self.conv_bn(net, 32, 3, 3, strides = (2, 2), padding = 'valid')
        net = self.conv_bn(net, 32, 3, 3, padding = 'valid')
        net = self.conv_bn(net, 64, 3, 3)

        b1 = MaxPool2D((3, 3), strides = (2, 2))(net)
        b2 = self.conv_bn(net, 96, 3, 3, strides = (2, 2), padding = 'valid')

        net = concatenate([b1, b2], axis = -1)

        b1 = self.conv_bn(net, 64, 1, 1)
        b1 = self.conv_bn(b1, 96, 3, 3, padding = 'valid')

        b2 = self.conv_bn(net, 64, 1, 1)
        b2 = self.conv_bn(b2, 64, 1, 7)
        b2 = self.conv_bn(b2, 64, 7, 1)
        b2 = self.conv_bn(b2, 96, 3, 3, padding = 'valid')

        net = concatenate([b1, b2], axis = -1)

        b1 = self.conv_bn(net, 192, 3, 3, strides = (2, 2), padding = 'valid')
        b2 = MaxPool2D((3, 3), strides = (2, 2))(net)

        net = concatenate([b1,b2], axis = -1)

        return net
    def InceptionA(self, net):
        b1 = AvgPool2D((3, 3), strides = (1, 1), padding = 'same')(net)
        b1 = self.conv_bn(b1, 96, 1, 1)

        b2 = self.conv_bn(net, 96, 1, 1)

        b3 = self.conv_bn(net, 64, 1, 1)
        b3 = self.conv_bn(b3, 96, 3, 3)

        b4 = self.conv_bn(net, 64, 1, 1)
        b4 = self.conv_bn(b4, 96, 3, 3)
        b4 = self.conv_bn(b4, 96, 3, 3)

        net = concatenate([b1, b2, b3, b4], axis = -1)

        return net

    def ReductionA(self, net):
        b1 = MaxPool2D((3, 3), strides = (2, 2))(net)

        b2 = self.conv_bn(net, 384, 3, 3, strides=(2, 2), padding = 'valid')

        b3 = self.conv_bn(net, 192, 1, 1)
        b3 = self.conv_bn(b3, 224, 3, 3)
        b3 = self.conv_bn(b3, 256, 3, 3, strides = (2, 2), padding = 'valid')

        net = concatenate([b1, b2, b3], axis = -1)

        return net

    def InceptionB(self, net):
        b1 = AvgPool2D((3, 3), strides=(1, 1), padding = 'same')(net)
        b1 = self.conv_bn(b1, 128, 1, 1)

        b2 = self.conv_bn(net, 384, 1, 1)

        b3 = self.conv_bn(net, 192, 1, 1)
        b3 = self.conv_bn(b3, 224, 7, 1)
        b3 = self.conv_bn(b3, 256, 7, 1)

        b4 = self.conv_bn(net, 192, 1, 1)
        b4 = self.conv_bn(b4, 192, 7, 1)
        b4 = self.conv_bn(b4, 224, 1, 7)
        b4 = self.conv_bn(b4, 224, 7, 1)
        b4 = self.conv_bn(b4, 256, 1, 7)

        net = concatenate([b1, b2, b3, b4], axis = -1)

        return net

    def ReductionB(self, net):
        b1 = MaxPool2D((3, 3), strides = (2, 2))(net)

        b2 = self.conv_bn(net, 192, 1, 1)
        b2 = self.conv_bn(b2, 192, 3, 3, strides=(2, 2), padding ='valid')

        b3 = self.conv_bn(net, 256, 1, 1)
        b3 = self.conv_bn(b3, 266, 7, 1)
        b3 = self.conv_bn(b3, 320, 1, 7)
        b3 = self.conv_bn(b3, 320, 3, 3, strides = (2, 2), padding = 'valid')

        net = concatenate([b1, b2, b3], axis = -1)

        return net

    def InceptionC(self, net):
        b1 = AvgPool2D((3, 3), strides = (1, 1), padding = 'same')(net)
        b1 = self.conv_bn(b1, 256, 1, 1)

        b2 = self.conv_bn(net, 256, 1, 1)

        b3 = self.conv_bn(net, 384, 1, 1)
        b3_1 = self.conv_bn(b3, 256, 3, 1)
        b3_2 = self.conv_bn(b3, 256, 1, 3)

        b4 = self.conv_bn(net, 384, 1, 1)
        b4 = self.conv_bn(b4, 448, 3, 1)
        b4 = self.conv_bn(b4, 512, 1, 3)
        b4_1 = self.conv_bn(b4, 256, 1, 3)
        b4_2 = self.conv_bn(b4, 256, 3, 1)

        net = concatenate([b1, b2, b3_1, b3_2, b4_1, b4_2], axis = -1)
        return net








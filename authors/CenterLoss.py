


from keras.layers import Input, Dense, MaxPooling2D, Flatten, Activation, Embedding, Lambda
from keras.layers import Conv2D
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Layer
from keras import backend as K
from keras import backend
from keras import losses
import numpy as np
import itertools
backend.set_image_dim_ordering('th')



label_size = 7

class CenterLossLayer(Layer):
    def __init__(self, alpha=0.5, lambda_2=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.lambda_2 = lambda_2

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers', shape=(7,1024), initializer='uniform', trainable=False)
        super().build(input_shape)

    def pairwise_distance(self, pair):
        fea_k = self.centers[pair[0], :]
        fea_j = self.centers[pair[1], :]
        # 1x1
        return K.dot(fea_k, K.transpose(fea_j)) + 1

    # x[0] is N*1024 , X[1] is N*7 one hot,
    def call(self, x, mask=None):

        delta_centers = K.dot(K.transpose(x[1]), K.dot(x[1], self.centers) - x[0])
        center_counts = K.sum(K.transpose(x[1]), axis=1, keepdims=True) + 1
        delta_centers /= center_counts
        new_centers = self.centers - self.alpha * delta_centers
        self.add_update((self.centers, new_centers), x)

        # Center Loss calculate
        self.result = x[0] - K.dot(x[1], self.centers)
        self.result = 0.5 * K.sum(self.result**2, axis=1, keepdims=True)


        # N x 1
        return self.result

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)


def FineTune(Emb, Label, initial_learning_rate = 1e-3, lambda_c = 0.003, alpha = 0.5):
    Input(shape=())




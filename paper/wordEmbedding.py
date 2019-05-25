



from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, Dropout, Flatten, RepeatVector
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import CuDNNLSTM, CuDNNGRU, Bidirectional, LSTM
from keras.layers.merge import concatenate
from keras.layers import Dense
from util import setting

from keras import losses




EMB_DIM = setting.EMB_DIM

class EmbeddingModel:

    def __init__(self, path):
        self.path = path
        self.max_feature = setting.max_feature
        self.max_len = setting.max_len
        self.model = None
        self.MaxNorm = 0.0
        self.emb_mean = 0.0
        self.emb_std = 0.0

    def train(self, Features, size=EMB_DIM):
        # print(Features)
        print('============== prepare train ==============')
        self.model = Word2Vec(Features, size=size, window=5, min_count=5, workers=4)
        self.model.save(self.path)
        print('============== train end ==============')




    def load(self, name):
        self.model = Word2Vec.load(name)
        return self.model

    def project_embedding(self, tokens, idf=None):

        if self.model is None:
            self.load(self.path)

        vectors = []
        sum_weight = 0
        for token in tokens:
            if not token in self.model.wv:
                continue
            # print('pass')
            weight = 1
            if idf and token in idf:
                weight = idf[token]
            v = self.model.wv[token] * weight
            vectors.append(v)
            sum_weight += weight

        emb = np.sum(vectors, axis=0)
        emb /= sum_weight
        return emb

    def ConstructInput(self, X):
        tokenizer = Tokenizer(num_words=self.max_feature)
        tokenizer.fit_on_texts(list(X))
        X = tokenizer.texts_to_sequences(X)
        X = pad_sequences(X, maxlen=self.max_len)

        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))
        return X, word_index

    # 直接返回word2vec的向量
    def CNNLSTM_embedding(self, word_index):
        if self.model is None:
            self.load(self.path)
        nb_words = len(word_index) + 1
        # InWord2VecNumber = 0
        # embedding_matrix = np.zeros((nb_words+1, EMB_DIM))
        embedding_matrix = np.random.normal(0, 0.25, (nb_words  , EMB_DIM))
        for word, i in word_index.items():
            if word in self.model.wv:
                embedding_vector = self.model.wv[word]
                embedding_matrix[i] = embedding_vector
                # InWord2VecNumber = InWord2VecNumber + 1
        # print(InWord2VecNumber)
        return embedding_matrix

    # 用一种很简单的平均的方式来取Paper的特征，发现效果出奇地好
    def PapersEmbedding(self, X, embedding_matrix, idf=None, Index2Word=None):
        PapersMatrix = []
        for x in X:
            temp = []
            # print(x)
            for id in x:
                SumDegree = np.sum(embedding_matrix[id])
                # print('id:', id, ', SumDegree: ', SumDegree)
                # print(embedding_matrix[id-1])
                if SumDegree != 0 and id != 0 and idf is None and Index2Word is None:
                    temp.append(embedding_matrix[id])
                elif SumDegree != 0 and id != 0 and Index2Word.__contains__(id) and idf.__contains__(Index2Word[id]):
                    temp.append(embedding_matrix[id] * idf[Index2Word[id]])
                else:
                    pass

            if len(temp) != 0:
                temp = np.array(temp)
                PapersMatrix.append(list(temp.mean(axis=0)))
            else:
                # PapersMatrix.append(list(np.random.normal(0, 0.25, EMB_DIM)))
                PapersMatrix.append(list(np.zeros(shape=(EMB_DIM))))
        return PapersMatrix




    # Autoencoder

    def AutoEncoder(self, length, max_features,  embedding_martix, embedding_dim=EMB_DIM, latent_dim=100):

        # channel 1, 1X1的卷积
        Datainput = Input(shape=(length,))
        embedding = Embedding(input_dim=max_features, output_dim=embedding_dim, weights=[embedding_martix])(Datainput)

        conv1 = Conv1D(filters=25, kernel_size=1, activation='relu')(embedding)

        # # channel 2
        conv2_1 = Conv1D(filters=25, kernel_size=1, activation='relu')(embedding)
        conv2_2 = Conv1D(filters=25, kernel_size=3, activation='relu', padding='same')(conv2_1)

        # # channel 3
        conv3_1 = Conv1D(filters=25, kernel_size=1, activation='relu')(embedding)
        conv3_2 = Conv1D(filters=25, kernel_size=3, activation='relu', padding='same')(conv3_1)
        conv3_3 = Conv1D(filters=25, kernel_size=3, activation='relu', padding='same')(conv3_2)

        # channel 4
        maxpool4 = MaxPooling1D(pool_size=2, strides=1, padding='same')(embedding)
        conv4_1 = Conv1D(filters=25, kernel_size=1, activation='relu', padding='same')(maxpool4)

        LSTMAutoEncoderInputs = concatenate([conv1, conv2_2, conv3_3, conv4_1])
        print(LSTMAutoEncoderInputs)

        encoded = LSTM(latent_dim, name='LSTMINPUT', return_sequences=False)(LSTMAutoEncoderInputs)
        print(encoded)

        # # LSTM把length个单词压缩成了一个encode
        decoded = RepeatVector(length)(encoded)
        print(decoded)

        # 这时候要变回vocab_size
        decoded = LSTM(embedding_dim, return_sequences=True)(decoded)
        print(decoded)

        sequence_autoencoder = Model(Datainput, decoded)
        sequence_autoencoder.summary()

        def CNNLSTMAutoencoderLoss():
            # losses.binary_crossentropy()

            def GetEncoderInput(layername):
                return sequence_autoencoder.get_layer(layername).input

            # y_true我们是用不着的
            def loss(_y_true, y_pred):
                AutoEncoderInput = GetEncoderInput('LSTMINPUT')
                return losses.binary_crossentropy(AutoEncoderInput, y_pred)

            return loss

        sequence_autoencoder.compile(optimizer='adadelta', loss=CNNLSTMAutoencoderLoss())

        # # encoder = Model(LSTMAutoEncoderInputs, encoded)
        return sequence_autoencoder























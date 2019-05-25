

import tensorflow as tf
from gae.layers import GraphConvolution, InnerProductDecoder
from GAE.layers import Model

flags = tf.app.flags
FLAGS = flags.FLAGS


# 这里有4个输入
class DisambiguationGCNAE(Model):
    # num_features, Author的特征维度
    def __init__(self, placeholder, num_features, features_nonzero=None, **kwargs):
        super(DisambiguationGCNAE, self).__init(**kwargs)
        self.author_feature_inputs = placeholder['author_feature_inputs']
        self.D2A = placeholder['D2A']
        self.AuthorAdj = placeholder['AuthorAdj']
        self.DoucmentAdj = placeholder['DoucmentAdj']

        self.input_dim = num_features
        self.dropout = placeholder['dropout']
        self.features_nonzero = features_nonzero
        self.build()

    def _build(self):

        self.GCN1 = GraphConvolution(input_dim=self.author_feature_inputs,
                                output_dim=FLAGS.hidden1,
                                adj=self.AuthorAdj,
                                dropout=self.dropout,
                                logging = self.logging
                                )(self.inputs)

        # N * hidden2 (60)
        self.AuthorEnbedding = GraphConvolution(input_dim=FLAGS.hidden1,
                                output_dim=FLAGS.hidden2,
                                adj=self.AuthorAdj,
                                dropout=self.dropout,
                                act=lambda x: x,
                                logging=self.logging
                                )(self.GCN1)

        self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                                   act=lambda x: x,
                                                   logging=self.logging)(self.AuthorEnbedding)

        # self.document_feature_inputs = tf.matmul(D2A, )

        pass





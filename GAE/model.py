import tensorflow as tf
from GAE.layers import GraphConvolution, InnerProductDecoder

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


class GCNModelAE(Model):
    def __init__(self, placeholders, num_features, features_nonzero=None, **kwargs):
        super(GCNModelAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        '''
        self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden1,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)
        '''

        self.hidden1 = GraphConvolution(input_dim=self.input_dim,
                                        output_dim=FLAGS.hidden1,
                                        adj=self.adj,
                                        act=tf.nn.relu,
                                        dropout=self.dropout,
                                        logging=self.logging)(self.inputs)

        self.embeddings = GraphConvolution(input_dim=FLAGS.hidden1,
                                           output_dim=FLAGS.hidden2,
                                           adj=self.adj,
                                           act=lambda x: x,
                                           dropout=self.dropout,
                                           logging=self.logging)(self.hidden1)

        self.z_mean = self.embeddings

        self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                                   act=lambda x: x,
                                                   logging=self.logging)(self.embeddings)


class GCNModelVAE(Model):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero=None, isCenterLoss=False, **kwargs):
        super(GCNModelVAE, self).__init__(**kwargs)

        self.features = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.isCenterLoss = isCenterLoss
        self.build()

    def _build(self):
        '''
        self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden1,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)
        '''

        self.hidden1 = GraphConvolution(input_dim=self.input_dim,
                                        output_dim=FLAGS.hidden1,
                                        adj=self.adj,
                                        act=tf.nn.relu,
                                        dropout=self.dropout,
                                        logging=self.logging)(self.features)

        self.z_mean = GraphConvolution(input_dim=FLAGS.hidden1,
                                       output_dim=FLAGS.hidden2,
                                       adj=self.adj,
                                       act=lambda x: x,
                                       dropout=self.dropout,
                                       logging=self.logging)(self.hidden1)



        self.z_log_std = GraphConvolution(input_dim=FLAGS.hidden1,
                                          output_dim=FLAGS.hidden2,
                                          adj=self.adj,
                                          act=lambda x: x,
                                          dropout=self.dropout,
                                          logging=self.logging)(self.hidden1)

        self.z = self.z_mean + tf.random_normal([self.n_samples, FLAGS.hidden2]) * tf.exp(self.z_log_std)  # element-wise

        self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                                   act=lambda x: x,
                                                   # act=tf.nn.relu,
                                                   logging=self.logging)(self.z)

    def get_center_loss(self, features, labels, alpha, num_classes):
        """获取center loss及center的更新op

        Arguments:
            features: Tensor,表征样本特征,一般使用某个fc层的输出,shape应该为[batch_size, feature_length].
            labels: Tensor,表征样本label,非one-hot编码,shape应为[batch_size].
            alpha: 0-1之间的数字,控制样本类别中心的学习率,细节参考原文.
            num_classes: 整数,表明总共有多少个类别,网络分类输出有多少个神经元这里就取多少.

        Return：
            loss: Tensor,可与softmax loss相加作为总的loss进行优化.
            centers: Tensor,存储样本中心值的Tensor，仅查看样本中心存储的具体数值时有用.
            centers_update_op: op,用于更新样本中心的op，在训练时需要同时运行该op，否则样本中心不会更新
        """
        # 获取特征的维数，例如256维
        len_features = features.get_shape()[1]
        # 建立一个Variable,shape为[num_classes, len_features]，用于存储整个网络的样本中心，
        # 设置trainable=False是因为样本中心不是由梯度进行更新的
        centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0), trainable=False)
        # 将label展开为一维的，输入如果已经是一维的，则该动作其实无必要
        labels = tf.reshape(labels, [-1])

        # 根据样本label,获取mini-batch中每一个样本对应的中心值
        centers_batch = tf.gather(centers, labels)
        # 计算loss
        loss = tf.nn.l2_loss(features - centers_batch)

        # 当前mini-batch的特征值与它们对应的中心值之间的差
        diff = centers_batch - features

        # 获取mini-batch中同一类别样本出现的次数,了解原理请参考原文公式(4)
        unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])

        diff = diff / tf.cast((1 + appear_times), tf.float32)
        diff = alpha * diff

        centers_update_op = tf.scatter_sub(centers, labels, diff)

        return loss, centers, centers_update_op

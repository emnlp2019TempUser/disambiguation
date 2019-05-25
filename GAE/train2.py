
from __future__ import division
from __future__ import print_function

import os
import time
from os.path import join

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from GAE.optimizer import OptimizerAE, OptimizerVAE
from GAE.input_data import load_local_data
from GAE.model import GCNModelAE, GCNModelVAE
from GAE.preprocessing import preprocess_graph, construct_feed_dict, \
    sparse_to_tuple, normalize_vectors, gen_train_edges, cal_pos_weight

from util.cluster import clustering
from util.data_utils import load_json
from util.eval_utils import pairwise_precision_recall_f1, cal_f1
from util import setting, data_utils, input_data

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 500, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 200, 'Number of units in hidden layer 1.')  # 32
flags.DEFINE_integer('hidden2', 100, 'Number of units in hidden layer 2.')  # 16
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')

flags.DEFINE_string('model', 'gcn_vae', 'Model string.')
flags.DEFINE_string('name', 'hui_fang', 'Dataset string.')
# flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
flags.DEFINE_integer('is_sparse', 0, 'Whether input features are sparse.')

model_str = FLAGS.model
name_str = FLAGS.name
start_time = time.time()


def gae_for_na(name, isend = False):
    """
    train and evaluate disambiguation results for a specific name
    :param name:  author name
    :return: evaluation results
    """
    adj, features, labels, AuthorIds = load_local_data(name=name)
    # print(labels)

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    adj_train = gen_train_edges(adj)

    adj = adj_train

    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    num_nodes = adj.shape[0]
    input_feature_dim = features.shape[1]
    if FLAGS.is_sparse:  # TODO to test
        # features = sparse_to_tuple(features.tocoo())
        # features_nonzero = features[1].shape[0]
        features = features.todense()  # TODO
    else:
        features = normalize_vectors(features)

    # Define placeholders
    placeholders = {
        # 'features': tf.sparse_placeholder(tf.float32),
        'features': tf.placeholder(tf.float32, shape=(None, input_feature_dim)),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'labels': tf.placeholder(tf.int32, shape=(None), name='labels')
    }

    # Create model
    model = None
    if model_str == 'gcn_ae':
        model = GCNModelAE(placeholders, input_feature_dim)
    elif model_str == 'gcn_vae':
        model = GCNModelVAE(placeholders, input_feature_dim, num_nodes)
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()  # negative edges/pos edges
    print('positive edge weight', pos_weight)
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.nnz) * 2)

    # Optimizer
    with tf.name_scope('optimizer'):
        if model_str == 'gcn_ae':
            opt = OptimizerAE(preds=model.reconstructions,
                              labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                          validate_indices=False), [-1]),
                              pos_weight=pos_weight,
                              norm=norm)
        elif model_str == 'gcn_vae':
            print('check Label: ', tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                           validate_indices=False), [-1]))
            opt = OptimizerVAE(preds=model.reconstructions,
                               labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                           validate_indices=False), [-1]),
                               model=model, num_nodes=num_nodes,
                               pos_weight=pos_weight,
                               norm=norm)

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    def get_embs():
        feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(model.z_mean, feed_dict=feed_dict)  # z_mean is better
        return emb

    # print('labels: ', labels)
    # print('labels len: ', len(labels))
    # temp = labels
    # temp = sorted(temp)
    # labels = sorted(labels)
    # print('labels len: ', len(list(set(labels))))
    # print('temp: ', temp, ', temp len', len(temp))
    # print('label: ', labels, ', labels len', len(labels))



    # Train model
    for epoch in range(FLAGS.epochs):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Run single weight update
        outs = sess.run([opt.opt_op, opt.cost, opt.accuracy],
                        feed_dict=feed_dict)

        # Compute average loss
        avg_cost = outs[1]
        avg_accuracy = outs[2]

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
              "train_acc=", "{:.5f}".format(avg_accuracy),
              "time=", "{:.5f}".format(time.time() - t))

    emb = get_embs()


    if isend:
        n_clusters = len(set(labels))
        emb_norm = normalize_vectors(emb)
        clusters_pred = clustering(emb_norm, num_clusters=n_clusters)
        prec, rec, f1 =  pairwise_precision_recall_f1(clusters_pred, labels)

        print('n_clusters:', n_clusters)
        print(clusters_pred)
        print(labels)

        print(emb)

        print('pairwise precision', '{:.5f}'.format(prec),
              'recall', '{:.5f}'.format(rec),
              'f1', '{:.5f}'.format(f1))
        return [prec, rec, f1], num_nodes, n_clusters
    else:
        return emb, AuthorIds


def FirstTrain(AuthorName):
    gae_for_na(AuthorName)

def SecondTrain(AuthorName):
    gae_for_na(AuthorName + '_document', True)


if __name__ == "__main__":
    # FirstTrain("Hongbin Li")
    SecondTrain("Hongbin Li")






























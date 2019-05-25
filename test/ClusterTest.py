
from GAE.input_data import load_local_data
from GAE.preprocessing import preprocess_graph, construct_feed_dict, \
    sparse_to_tuple, normalize_vectors, gen_train_edges, cal_pos_weight
from util.cluster import clustering
from util.eval_utils import pairwise_precision_recall_f1, cal_f1
import numpy as np
from GAE.train2 import gae_for_na

def Cluster1Test(name):

    adj, features, labels, AuthorIds = load_local_data(name=name)

    print(features)
    print(features.shape)

    n_clusters = len(set(labels))
    emb_norm = normalize_vectors(features)
    clusters_pred = clustering(emb_norm, num_clusters=n_clusters)
    prec, rec, f1 = pairwise_precision_recall_f1(clusters_pred, labels)

    print('n_clusters:', n_clusters)
    print(clusters_pred)
    print(labels)

    print(features)

    print('pairwise precision', '{:.5f}'.format(prec),
          'recall', '{:.5f}'.format(rec),
          'f1', '{:.5f}'.format(f1))

def AuthorFeatureClusterTest(name):
    adj, features, labels, AuthorIds = load_local_data(name=name)

    print(features)
    print(features.shape)

    n_clusters = 62
    emb_norm = normalize_vectors(features)
    clusters_pred = clustering(emb_norm, num_clusters=n_clusters)
    prec, rec, f1 = pairwise_precision_recall_f1(clusters_pred, labels)

    print('n_clusters:', n_clusters)
    print(clusters_pred)
    print(list(labels))

    print(features)

    print('pairwise precision', '{:.5f}'.format(prec),
          'recall', '{:.5f}'.format(rec),
          'f1', '{:.5f}'.format(f1))


if __name__ == "__main__":
    AuthorName = 'Hongbin Li'
    # name = AuthorName + '_document'
    # AuthorFeatureClusterTest(AuthorName)
    gae_for_na(AuthorName, isend=True)





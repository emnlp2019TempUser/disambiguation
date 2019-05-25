

from GAE.input_data import load_local_data
from GAE.preprocessing import preprocess_graph, construct_feed_dict, \
    sparse_to_tuple, normalize_vectors, gen_train_edges, cal_pos_weight
from util.cluster import clustering
from util.eval_utils import pairwise_precision_recall_f1, cal_f1

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from GAE import train2


def PCAAnanlyse(emb):
    pca = PCA(n_components=2)
    X_new = pca.fit_transform(emb)
    plt.scatter(X_new[:, 0], X_new[:, 1], marker='o')
    plt.show()


def RawAnalyse():
    name = 'Hongbin Li'
    # name = AuthorName + '_document'
    adj, features, labels, AuthorIds = load_local_data(name=name)

    emb_norm = normalize_vectors(features)

    print(emb_norm)
    PCAAnanlyse(emb_norm)


def GCNAEAnalyse():
    name = 'Hongbin Li'
    emb, AuthorIds = train2.gae_for_na(name)
    PCAAnanlyse(emb)

def RawDocumentContentAnanlyse():
    name = 'Hongbin Li'
    name = name + '_document'
    adj, features, labels, AuthorIds = load_local_data(name=name)

    emb_norm = normalize_vectors(features)

    print(emb_norm)
    PCAAnanlyse(emb_norm)



if __name__ == "__main__":
    # RawAnalyse()
    GCNAEAnalyse()









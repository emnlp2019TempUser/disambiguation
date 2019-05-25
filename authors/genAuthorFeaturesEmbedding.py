
from paper import wordEmbedding
import json
from sklearn.feature_extraction.text import TfidfVectorizer

def main():

    print(" ======= start to generate Author Feature Embedding =======")

    with open('./data/authors/AuthorFeature.json', 'r') as fp:
        AuthorFeature = json.load(fp)
        fp.close()

    EmbPath = './data/model/TextFeatures.emb'
    Emb = wordEmbedding.EmbeddingModel(EmbPath)

    TempFeatures = AuthorFeature.values()
    Aids = AuthorFeature.keys()
    X, Word_index = Emb.ConstructInput(TempFeatures)



    embedding_matrix = Emb.CNNLSTM_embedding(Word_index)
    PapersEmbedding = list(Emb.PapersEmbedding(X, embedding_matrix))
    Author2PaperFeatures2 = dict(zip(Aids, PapersEmbedding))

    with open('./data/authors/AuthorFeatureEmb.json','w') as fp:
        json.dump(Author2PaperFeatures2, fp)
        fp.close()

    print(" =======  generate Author Feature Embedding end =======")

if __name__ == "__main__":
    main()


























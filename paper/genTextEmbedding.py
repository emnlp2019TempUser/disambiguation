
import json
from paper import wordEmbedding
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    print(" ======= start to generate Text embedding =======")
    EmbPath = './data/model/TextFeatures.emb'
    Emb = wordEmbedding.EmbeddingModel(EmbPath)

    with open('./data/papers/PapersFeatures.json', 'r') as fp:
        PapersFeatures = json.load(fp)
        Emb.train(PapersFeatures.values())

        # print(PapersFeatures.values())

        for key in PapersFeatures.keys():
            temp = ' '.join(PapersFeatures[key])
            PapersFeatures[key] = temp

        Corpus = PapersFeatures.values()
        # print(Corpus)



        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(Corpus)
        FeaturesName = vectorizer.get_feature_names()
        FeaturesNameLen = len(FeaturesName)
        AllFeatures = FeaturesName[int(FeaturesNameLen * 0.05):int(FeaturesNameLen * 0.95)]
        FeaturesSet = set(AllFeatures)
        VocabularyIDF = (vectorizer.idf_)[int(FeaturesNameLen * 0.05):int(FeaturesNameLen * 0.95)]
        Feature2IDF = dict(zip(AllFeatures, VocabularyIDF))

        # print(Feature2IDF)
        # print(Pid2PaperFeatures2)

        X, Word_index = Emb.ConstructInput(Corpus)
        Index2Word = {Word_index[k]:k for k in Word_index.keys()}
        # print(Index2Word)


        embedding_matrix = Emb.CNNLSTM_embedding(Word_index)
        PapersEmbedding = list(Emb.PapersEmbedding(X, embedding_matrix, idf=Feature2IDF, Index2Word=Index2Word))
        Pid2PaperEmbedding = dict(zip(PapersFeatures.keys(), PapersEmbedding))

        # print(PapersEmbedding)

        fp.close()

    with open('./data/papers/PaperFeatureEmb.json', 'w') as fp:
        json.dump(Pid2PaperEmbedding, fp)
        fp.close()

    print(" ======= generate Text embedding end =======")


if __name__ == "__main__":
    main()
























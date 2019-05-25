
from util import input_data, setting
import numpy as np

#使用AuthoEncoder对Paper的特征进行压缩
def FeatureCompression(authorname, usefulAuthorId2Emb):
    AuthorDisambiguationMateria = input_data.getAuthorDisambiguationMateria(authorname)
    Labels = []
    for key in AuthorDisambiguationMateria.keys():
        Labels = Labels + AuthorDisambiguationMateria[key]

    Labels = set(Labels)
    Aid2LabelIndex = {Aid:idx for idx, Aid in enumerate(Labels)}
    print(Aid2LabelIndex)

    AuthorFeatures = np.zeros(shape=(len(Labels), setting.AUTHOR_EMB_DIM))
    for aid in usefulAuthorId2Emb.keys():
        Emb = usefulAuthorId2Emb[aid]
        AuthorFeatures[aid] = Emb

    TempFeature = AuthorFeatures.flatten()
    print(TempFeature)

def extractPaperFeatureEmb():
    pass





if __name__ == "__main__":
    FeatureCompression('Hongbin Li', 1)




















































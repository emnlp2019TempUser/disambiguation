
import codecs
from os.path import join
import numpy as np
from GAE import train2
from util import input_data, data_utils, setting
from sklearn.feature_extraction.text import TfidfVectorizer

def main(Authorname):
    # PaperFeatureEmbdding
    PaperFeatureEmb = input_data.getPaperEmb()

    # AuthorFeatureEmbdding
    AuthorFeatureEmb = input_data.getAuthorEmb()

    # Document - Document Network
    DisambiguationAuthorIds = data_utils.getDisambiguationData(Authorname)
    AId2Pids = input_data.getAuthor2Pids()
    print(DisambiguationAuthorIds)
    print(len(set(DisambiguationAuthorIds)))

    DisambiguationPaperId = []
    for aid in DisambiguationAuthorIds:
        Pids = AId2Pids[aid]
        DisambiguationPaperId = DisambiguationPaperId + Pids

    DisambiguationPaperId = list(set(DisambiguationPaperId))
    print(DisambiguationPaperId)

    PaperFeatrues = input_data.getPaperFeatrues()
    DisambiguationPaperFeatures = {pid:' '.join(PaperFeatrues[pid]) for pid in DisambiguationPaperId}
    print(DisambiguationPaperFeatures)


    corpus = DisambiguationPaperFeatures.values()
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    FeaturesName = vectorizer.get_feature_names()
    FeaturesNameLen = len(FeaturesName)
    AllFeatures = FeaturesName[int(FeaturesNameLen*0.05):int(FeaturesNameLen*0.95)]
    FeaturesSet = set(AllFeatures)
    VocabularyIDF = (vectorizer.idf_)[int(FeaturesNameLen*0.05):int(FeaturesNameLen*0.95)]
    Feature2IDF = dict(zip(AllFeatures, VocabularyIDF))


    # print(AllFeatures)
    # print(set(AllFeatures))

    # Document Fetures

    Pid2Aids = input_data.getPid2AIds()
    AuthorEmb, AuthorIds = train2.gae_for_na(Authorname)
    # (1223, 64)
    print(AuthorEmb.shape)
    AuthorFeature = dict(zip(AuthorIds, AuthorEmb))

    # DisambiguationDimension = setting.EMB_DIM
    NumberAutuhor = 1
    DisambiguationDimension = NumberAutuhor*AuthorEmb.shape[1]

    DocumentFetures = np.zeros(shape=(len(DisambiguationPaperId), DisambiguationDimension))

    TempAuthorids = []
    # print(AuthorFeature)
    for id, pid in enumerate(DisambiguationPaperId):
        # temp = np.zeros(shape=(AuthorEmb.shape[1]))
        if len(Pid2Aids[pid]) == 0:
            print('======= Pid2Aids[pid] is zero!!!! =======')
        temp = []
        cnt = 1
        for index, aid in enumerate(Pid2Aids[pid]):
            aid = str(aid)

            # 尝试用未经过Author的领域特征的文本来表示
            if aid not in AuthorFeature :
                continue
            if aid in DisambiguationAuthorIds:
                temp = AuthorFeature[aid]
                break
            TempAuthorids.append(aid)
            # if cnt < NumberAutuhor:
            #     TempFeature = (10/(cnt+1)) * AuthorFeature[aid]
            #     temp += list(TempFeature)
            #     cnt = cnt + 1
        if sum(temp) != 0 and cnt == NumberAutuhor:
            # DocumentFetures[id] = np.array(PaperFeatureEmb[pid])
            DocumentFetures[id] = np.array(temp)
        else:
            DocumentFetures[id] = np.zeros(shape=(DisambiguationDimension))

    print(DocumentFetures.shape)
    print(DocumentFetures)

    Labels = data_utils.getDisambiguationDataLabel(DisambiguationPaperId, Authorname)

    # (346, 64)
    print(len(Labels))
    print(len(set(Labels)))

    wf = codecs.open(join(setting.materialData, "{}_document_pubs_content.txt".format(Authorname)), "w", encoding='utf-8')
    for id, pid in enumerate(DisambiguationPaperId):
        TempValues = DocumentFetures[id]
        if sum(TempValues) == 0:
            continue
        wf.write('{} '.format(pid))

        for FeatureValue in TempValues:
            wf.write('{} '.format(FeatureValue))
        wf.write('{}\n'.format(Labels[id]))
    wf.close()

    wf = codecs.open(join(setting.materialData, "{}_document_pubs_network.txt".format(Authorname)), "w", encoding='utf-8')
    DocumentNetwork = np.zeros((len(DisambiguationPaperId), len(DisambiguationPaperId)))
    for i in range(len(DisambiguationPaperId)):
        for j in range(i+1, len(DisambiguationPaperId), 1):
            if i == j: continue
            PidI = DisambiguationPaperId[i]
            PidJ = DisambiguationPaperId[j]
            DocumentIFeatures = set((' '.join(PaperFeatrues[PidI])).split(' ')) & FeaturesSet
            DocumentJFeatures = set(' '.join(PaperFeatrues[PidJ]).split(' ')) & FeaturesSet
            InterKeyWork = DocumentIFeatures & DocumentJFeatures
            RelateKeyWordIdf = [Feature2IDF[feature] for feature in InterKeyWork]
            # print(set((' '.join(PaperFeatrues[PidI])).split(' ')))
            # print(DocumentIFeatures)
            if sum(DocumentFetures[i]) == 0 or sum(DocumentFetures[j]) == 0:
                continue
            if sum(RelateKeyWordIdf) > 10:
                # print(InterKeyWork)
                wf.write('{} {}\n'.format(PidI, PidJ))
    wf.close()





    # check
    TempAuthorids = list(set(TempAuthorids))
    AuthorIds = list(set(AuthorIds))
    TempAuthorids = sorted(TempAuthorids)
    AuthorIds = sorted(AuthorIds)

    print(TempAuthorids)
    print(AuthorIds)



if __name__ == "__main__":
    Authorname = "Hongbin Li"
    main(Authorname)



















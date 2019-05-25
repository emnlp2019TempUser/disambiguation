
import codecs
from util import input_data, setting
from os.path import join
from collections import defaultdict
import json

def genAuthorGCNData(authorname):
    AuthorEmb = input_data.getAuthorEmb()
    AuthorName2Id = input_data.getAuthorName2Id()
    AId2Pids = input_data.getAuthor2Pids()
    Pid2AIds = input_data.getPid2AIds()



    DisambiguationAuthorId = []
    # 找出authorName有关的全部AuthorId
    for AuthorName in AuthorName2Id.keys():
        Res = AuthorName.split(':')
        Name = Res[0]
        if Name == authorname:
            aid = str(AuthorName2Id[AuthorName])
            DisambiguationAuthorId.append(aid)

    DisambiguationAuthorId = list(set(DisambiguationAuthorId))
    print('DisambiguationAuthorId len: ', len(DisambiguationAuthorId))

    RelatedPids = []
    RecentlyDisambiguationAuthorId = defaultdict(list)
    for aid in DisambiguationAuthorId:
            RelatedPids = RelatedPids + AId2Pids[aid]
            for temppid in AId2Pids[aid]:
                for tempaid in Pid2AIds[temppid]:
                    RecentlyDisambiguationAuthorId[tempaid].append(aid)
                    # for test the embedding result
                    # RecentlyDisambiguationAuthorId[tempaid].append(Aid2RawAId[aid])

    RelatedPids = list(set(RelatedPids))

    # print('RelatedPids: ', RelatedPids)



    # Gen pubs content
    GraphNetwork = defaultdict(list)

    for pid in RelatedPids:
        Aids = Pid2AIds[pid]
        tempLen = len(Aids)
        for i in range(tempLen):
            for j in range(i+1, tempLen, 1):
                GraphNetwork[Aids[i]].append(Aids[j])
                # wf.write("{} {}\n".format(Aids[i], Aids[j]))


    RelatedAid = []
    wf = codecs.open(join(setting.materialData, "{}_pubs_network.txt".format(authorname)), 'w', encoding='utf-8')

    # print('AuthorEmb keys:', AuthorEmb.keys())
    Flag = defaultdict(int)
    for key in GraphNetwork.keys():
        # print('key: ', key)
        if len(GraphNetwork[key]) > 10 and AuthorEmb.__contains__(key)  and sum(AuthorEmb[key]) != 0:
            Flag[key] = 1
        else:
            Flag[key] = 0



    for key in GraphNetwork.keys():
        values = GraphNetwork[key]
        if Flag[key] == 1:
            for v in values:
                if Flag[v] == 1:
                    wf.write("{} {}\n".format(key, v))
            RelatedAid.append(key)
    wf.close()


    RelatedAid = list(set(RelatedAid))



    wf.close()

    wf = codecs.open(join(setting.materialData, "{}_pubs_network_index.txt".format(authorname)), 'w', encoding='utf-8')

    for index, aid in enumerate(RelatedAid):
        wf.write('{} {}\n'.format(aid, aid))
    wf.close()


    wf = codecs.open(join(setting.materialData, "{}_pubs_content.txt".format(authorname)), 'w', encoding='utf-8')

    # 找出最近的合作过的张军的ID
    print('找出最近的合作过的ID')
    AllType = []
    for aid in RecentlyDisambiguationAuthorId.keys():
        RecentlyDisambiguationAuthorId[aid] = list(set(RecentlyDisambiguationAuthorId[aid]))
        AllType = AllType + list(set(RecentlyDisambiguationAuthorId[aid]))

    # print(RecentlyDisambiguationAuthorId)
    print(len(list(set(AllType))))

    with open(join(setting.disambiguationMaterialData, '{}_disambiguationMaterialData.json'.format(authorname)), 'w') as fp:
        json.dump(RecentlyDisambiguationAuthorId, fp)
        fp.close()


    # print(RelatedAid)
    for aid in RelatedAid:
        wf.write('{} '.format(aid))
        aid = str(aid)
        Embdding = AuthorEmb[aid]
        for value in Embdding:
            wf.write('{} '.format(value))
        wf.write('{}\n'.format(RecentlyDisambiguationAuthorId[aid][0]))
    wf.close()

def main(authorname):
    print(" ======= start to generate First layerout Matrix Data of  author {} =======".format(authorname))

    genAuthorGCNData(authorname)

    print(" ======= generate First layerout Matrix Data of  author {} end =======".format(authorname))



def test(authorname):
    AuthorName2Id = input_data.getAuthorName2Id()


    DisambiguationName = []
    DisambiguationAuthorId = []
    # 找出authorName有关的全部AuthorId
    for AuthorName in AuthorName2Id.keys():
        Res = AuthorName.split(':')
        Name = Res[0]
        if Name == authorname:
            aid = str(AuthorName2Id[AuthorName])
            DisambiguationAuthorId.append(aid)
            DisambiguationName.append(AuthorName)

    DisambiguationAuthorId = list(set(DisambiguationAuthorId))
    print('DisambiguationAuthorId len: ', len(DisambiguationAuthorId))
    print(DisambiguationName)

if __name__ == "__main__":
    main('Hongbin Li')
    # test('Hongbin Li')



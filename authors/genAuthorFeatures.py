
import json
from collections import defaultdict
from util import input_data

def main():
    print(" ======= start to generate Author Features =======")

    with open('./data/authors/AId2Pids.json', 'r') as fp:
        Aid2Pids = json.load(fp)

    with open('./data/papers/PapersFeatures.json', 'r') as fp:
        PapersFeatures = json.load(fp)

    Pid2Aids = input_data.getPid2AIds()


    AuthorFeature = defaultdict(list)
    for aid in Aid2Pids.keys():
        for pid in Aid2Pids[aid]:
            if Pid2Aids[pid].index(aid) > 1:
                continue
            AuthorFeature[aid] = AuthorFeature[aid] + PapersFeatures[pid]


    with open('./data/authors/AuthorFeature.json', 'w') as fp:
        json.dump(AuthorFeature, fp)
        fp.close()

    print(" ======= generate Author Features end =======")

if __name__ == "__main__":
    main()






















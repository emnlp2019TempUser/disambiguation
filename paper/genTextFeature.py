
import json
from paper import textFeatureUtil


def main():
    print(" ======= start to generate Text Feature =======")

    with open('./data/pubs_raw.json', 'r') as fp:

        Document = json.load(fp)
        Papers = []
        for key in Document.keys():
            Papers.append(Document[key])

        texts = []
        PapersFeatures = textFeatureUtil.extract_paper_features(Papers)
        PapersFeatures = dict(zip(Document.keys(), PapersFeatures))
        with open('./data/papers/PapersFeatures.json','w') as PapersFeaturesfp:
            json.dump(PapersFeatures, PapersFeaturesfp)
            PapersFeaturesfp.close()

    print(" ======= generate Text Feature end =======")




























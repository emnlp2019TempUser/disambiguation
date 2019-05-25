import codecs
import json
from os.path import join
from util import setting

def getAuthor2Pids():
    AId2Pids = load_json(setting.AuthorDataPath, 'AId2Pids.json')
    return AId2Pids

def getPid2AIds():
    Pid2AIds = load_json(setting.PaperDataPath, 'PId2AIds.json')
    return Pid2AIds

def getAuthorEmb():
    AuthorFeatureEmb = load_json(setting.AuthorDataPath, 'AuthorFeatureEmb.json')
    return AuthorFeatureEmb

def getAuthorDisambiguationMateria(authorname):
    AuthorDisambiguationMateria = load_json(setting.disambiguationMaterialData, '{}_disambiguationMaterialData.json'.format(authorname))
    return AuthorDisambiguationMateria

def getPaperEmb():
    PaperFeatureEmb = load_json(setting.PaperDataPath, 'PaperFeatureEmb.json')
    return PaperFeatureEmb

def getRawData():
    RawData = load_json(setting.DataPath, "pubs_raw.json")
    return RawData

def getAuthorName2Id():
    AuthorName2Id = load_json(setting.AuthorDataPath, "AuthorName2Id.json")
    return AuthorName2Id

def getAuthorName2AuthorId():
    AuthorName2AuthorId = load_json(setting.AuthorDataPath, "AuthorName2AuthorId.json")
    return AuthorName2AuthorId

def getAid2RawAid():
    Aid2RawAid = load_json(setting.AuthorDataPath, "Aid2RawAid.json")
    return Aid2RawAid

def getPaperFeatrues():
    paperFeatures = load_json(setting.PaperDataPath, "PapersFeatures.json")
    return paperFeatures

def load_json(rfdir, rfname):
    with codecs.open(join(rfdir, rfname), 'r', encoding='utf-8') as rf:
        return json.load(rf)

def dump_json(obj, wfpath, wfname, indent=None):
    with codecs.open(join(wfpath, wfname), 'w', encoding='utf-8') as wf:
        json.dump(obj, wf, ensure_ascii=False, indent=indent)

from collections import defaultdict
from util import setting, input_data


print("===== prepare authorid and pid  ===== ")
rawPapers = input_data.load_json(setting.DataPath, 'pubs_raw.json')

AuthorName2AuthorId = defaultdict(str)
Aid2RawAid = defaultdict(str)
AuthorSet = set()
for pid in rawPapers:
    authors = rawPapers[pid]['authors']
    for author in authors:
        if author.__contains__('org'):
            AuthorSet.add(author['name'] + ':' + author['org'])
            AuthorName2AuthorId[author['name'] + ':' + author['org']] = author['id']

# 726804
# print(len(AuthorSet))
# 406402
# print(len(RawDataAuthorData))

AuthorName2Id = {AuthorName:Id for Id,AuthorName in enumerate(AuthorSet)}
Aid2RawAid = {Id:AuthorName2AuthorId[AuthorName] for Id,AuthorName in enumerate(AuthorSet) }

input_data.dump_json(AuthorName2Id, setting.AuthorDataPath, 'AuthorName2Id.json')

# 生成Authorid 2 Pid
# 生成Pid2Author

AuthorId2Pids = defaultdict(list)
Pid2AuthorIds = defaultdict(list)

for pid in rawPapers:
    authors = rawPapers[pid]['authors']
    for author in authors:
        if author.__contains__('org'):
            AuthorId = AuthorName2Id[author['name'] + ':' + author['org']]
            AuthorId2Pids[AuthorId].append(pid)
            Pid2AuthorIds[pid].append(str(AuthorId))

print("===== prepare AId2Pids.json  ===== ")

input_data.dump_json(AuthorId2Pids, setting.AuthorDataPath, 'AId2Pids.json')

print("===== prepare PId2AIds.json  ===== ")

input_data.dump_json(Pid2AuthorIds, setting.PaperDataPath, 'PId2AIds.json')

print("===== prepare AuthorName2AuthorId.json  ===== ")

input_data.dump_json(AuthorName2AuthorId, setting.AuthorDataPath, 'AuthorName2AuthorId.json')

print("===== prepare Aid2RawAid.json  ===== ")

input_data.dump_json(Aid2RawAid, setting.AuthorDataPath, 'Aid2RawAid.json')

print("===== prepare end ===== ")



















from util import input_data

raw_data = input_data.getRawData()


AllAuthors = []
for pid in raw_data:
    paper = raw_data[pid]
    for author in paper['authors']:
        AllAuthors.append(author['id'])

print(len(set(AllAuthors)))
# print(len(raw_data))



from util import input_data, data_utils

JsonMap = input_data.getAid2RawAid()

Aid1 = '597577'
Aid2 = '212554'

print('Aid1: ', JsonMap[Aid1])
print('Aid2: ', JsonMap[Aid2])


AuthorName2Id = input_data.getAuthorName2Id()
Id2Authorname = {AuthorName2Id[authorname]:authorname for authorname in AuthorName2Id.keys()}


print('author1: ', Id2Authorname[int(Aid1)])
print('author2: ', Id2Authorname[int(Aid2)])


DisambiguationalAuthors = data_utils.getDisambiguationData('Hongbin Li')
print(DisambiguationalAuthors)
print(Aid2 in DisambiguationalAuthors)

## get specified document

D1 = '5b5433e7e1cd8e4e15fc4f02'
D2 = '5b5433f4e1cd8e4e15189218'
Documents = input_data.getRawData()
print(Documents[D1])
print(Documents[D2])


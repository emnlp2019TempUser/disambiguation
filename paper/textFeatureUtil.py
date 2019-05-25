
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
import re


punct = set(u''':!)#,.:;?.]}¢'"、。〉》」』-〕〗>〞︰︱︳©﹐､﹒
﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､￠
々‖•·ˇˉ―′’”([{£¥'"‵〈《「『〔〖（［｛￡￥〝︵︷︹︻
︽︿﹁﹃﹙﹛﹝（｛“‘_…/''')
stemmer = nltk.stem.PorterStemmer()


def stem(word):
    return stemmer.stem(word)

def clean_stop_words(sentenceWords):
    filter_sentence = [w for w in sentenceWords if w not in stopwords.words('english')]
    return filter_sentence

def clean_sentence(text, stemming=True):
    # print('clean_sentence pass')
    for token in punct:
        text = text.replace(token, "")

    text = re.sub('[0-9]{1,}', '', text)
    words = text.split()
    if stemming:
        stemming_words = []
        for w in words:
            stemming_words.append(stem(w))
        words = clean_stop_words(stemming_words)
    # words = clean_numbers(words)
    return words

def clean_abstract(abstract):
    blob = TextBlob(abstract)
    keys = blob.noun_phrases
    for idx, key in enumerate(keys):
        keys[idx] = str(key)
    return keys

def extract_common_features(paper):
    # print(paper['title'])
    # print(paper['title'].lower())
    titleFeatures = clean_sentence(paper['title'].lower())
    keywords = []
    venues = []

    if 'keyword' in paper.keys():
        keywords = clean_sentence(paper['keyword'].lower())
        # keywords = transform_feature(clean_sentence(paper['keyword'].lower()), f_name='keyword')

    # if 'venue' in paper.keys():
    #     venues = clean_sentence(paper['venue'].lower())
        # venues = transform_feature(clean_sentence(paper['venue'].lower()), f_name='venue')

    abstract = []
    if 'abstract' in paper.keys():
        # abstract = clean_sentence((paper['abstract'].lower()))
        abstract = clean_abstract(paper['abstract'].lower())

    return titleFeatures + keywords + venues + abstract

from collections import defaultdict
def extract_paper_features(papers):
    Features = []
    for pid, paper in enumerate(papers):
        # if pid % 1000 == 0:
        #     print('pid:' , pid)
        Feature = extract_common_features(paper)
        Features.append(Feature)
    return Features

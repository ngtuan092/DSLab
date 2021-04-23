from collections import defaultdict
from os import listdir
from os.path import isfile
import re
import numpy as np

def gather_20newsgroups_data():
    path = "../datasets/20news-bydate/"
    dirs = [path + dirname +
            '/' for dirname in listdir(path) if not isfile(path + dirname)]

    train_dir, test_dir = dirs if "train" in dirs[0] else dirs[::-1]
    list_newgroups = [dirname for dirname in listdir(train_dir)]
    list_newgroups.sort()
    fp = open("../datasets/20news-bydate/stop_words.txt")
    stop_words = fp.read().splitlines()
    fp.close()
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()

    def collect_data_from(parent_dir, newsgroup_list):
        data = []
        for group_id, newsgroup in enumerate(newsgroup_list):
            label = group_id
            dir_path = parent_dir + "/" + newsgroup + "/"
            files = [(filename, dir_path + filename)
                     for filename in listdir(dir_path) if isfile(dir_path + filename)]
            files.sort()
            for filename, filepath in files:
                with open(filepath, "r") as f:
                    text = f.read().lower()
                    words = [stemmer.stem(word) for word in re.split(
                        r'\W+', text) if word not in stop_words]
                    content = ' '.join(words)
                    delimiter = "<fff>"
                    assert(len(content.splitlines()) == 1)
                    data.append(str(label) + delimiter +
                                filename + delimiter + content)
        return data
    train_data = collect_data_from(train_dir, list_newgroups)
    test_data = collect_data_from(test_dir, list_newgroups)
    fulldata = train_data + test_data
    with open("../datasets/20news-bydate/20news-train-processed.txt", "w") as f:
        f.write("\n".join(train_data))
    with open("../datasets/20news-bydate/20news-test-processed.txt", "w") as f:
        f.write("\n".join(test_data))
    with open("../datasets/20news-bydate/20news-full-processed.txt", "w") as f:
        f.write("\n".join(fulldata))


def generate_vocabulary(datapath):
    def compute_idf(df, corpus_size):
        assert df > 0
        return np.log10(corpus_size * 1. / df)
    with open(datapath) as f:
        lines = f.read().splitlines()
    doc_count = defaultdict(int)
    corpus_size = len(lines)

    for line in lines:
        feature = line.split("<fff>")
        text = feature[-1]
        words = list(set(text.split()))
        for word in words:
            doc_count[word] += 1

    words_idfs = [(word, compute_idf(document_freq, corpus_size))
                  for word, document_freq in zip(doc_count.keys(), doc_count.values()) if document_freq > 10 and not word.isdigit()]
    words_idfs.sort(key=lambda entry: -entry[-1])
    print(f'Vocabulary size: {len(words_idfs)}')
    with open('../datasets/20news-bydate/words_idfs.txt', 'w') as f:
        f.write('\n'.join([word + "<fff>" + str(idf)
                           for word, idf in words_idfs]))


def get_tf_idf(data_path):
    delimiter = "<fff>"
    with open("../datasets/20news-bydate/words_idfs.txt") as f:
        word_idfs = [(line.split(delimiter)[0], float(line.split(delimiter)[1]))
                     for line in f.read().splitlines()]
        word_IDs = dict([(word, index)
                         for index, (word, idf) in enumerate(word_idfs)])
        idfs = dict(word_idfs)

    with open(data_path, "r") as f:
        documents = [(int(line.split(delimiter)[0]), int(line.split(delimiter)[
                      1]), line.split(delimiter)[2]) for line in f.read().splitlines()]
    data_tf_idf = []
    for document in documents:
        label, doc_id, text = document
        words = [word for word in text.split() if word in idfs]
        word_set = list(set(words))
        max_term_freq = max([words.count(word) for word in word_set])
        word_tf_idfs = []
        sum_square = 0.0
        for word in word_set:
            term_freq = words.count(word)
            tf_idf_value = (term_freq * 1. / max_term_freq) * idfs[word]
            word_tf_idfs.append((word_IDs[word], tf_idf_value))
            sum_square += tf_idf_value ** 2
        words_tf_idfs_normalized = [str(index) + ':' + str(tf_idf_value / np.sqrt(
            sum_square)) for index, tf_idf_value in word_tf_idfs]
        sparse_rep = ' '.join(words_tf_idfs_normalized)
        data_tf_idf.append((label, doc_id, sparse_rep))
    with open("../datasets/20news-bydate/tf_idf.txt", "w") as p:
        p.write('\n'.join([str(label) + delimiter + str(doc_id) + delimiter +
                           sparse_rep for label, doc_id, sparse_rep in data_tf_idf]))


if __name__ == '__main__':
    gather_20newsgroups_data()
    generate_vocabulary("../datasets/20news-bydate/20news-full-processed.txt")
    get_tf_idf("../datasets/20news-bydate/20news-full-processed.txt")
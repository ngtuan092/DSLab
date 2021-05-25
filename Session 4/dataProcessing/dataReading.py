from collections import defaultdict
from os import listdir
from os.path import isfile
from re import split


def gen_data_and_vocab():
    def collect_data_from(parent_path, newsgroup_list, word_count=None):
        """lấy data có label trong newsgroup_list từ parent_path 
            word_count là một dictionary để lưu số lần xuất hiện của word trong datasets
        """
        data=[]
        for group_id, newsgroup in enumerate(newsgroup_list):
            dir_path = parent_path + '/' + newsgroup + '/' 
            files = [(file, dir_path + file) for file in listdir(dir_path) if isfile(dir_path + file)]
            files.sort()
            label = group_id
            print(f'Processing: {group_id} - {newsgroup}')
            for file, filepath in files:
                with open(filepath) as f:
                    text = f.read().lower() # chuyển hết về lower case
                    words = split('\W+', text) # phân tích văn bản thành các word
                    if word_count is not None:
                        for word in words:
                            word_count[word] += 1
                    content = ' '.join(words)

                assert len(content.splitlines()) == 1
                data.append(f'{str(label)}<fff>{file}<fff>{content}')
        return data

    word_count = defaultdict(int)  # đếm tần suất xuất hiện của word trong data
    # đường dẫn đến folder chứa train và test datasets
    path = '../datasets/20news-bydate/'

    # lấy train path và test path
    parts = [path + dir_name +
             '/' for dir_name in listdir(path) if not isfile(path + dir_name)]
    train_path, test_path = parts if 'train' in parts[0] else parts[::-1]

    newsgroup_list = [newsgroup for newsgroup in listdir(train_path)]
    newsgroup_list.sort()
    # lấy training data, word_count
    train_data = collect_data_from(
        parent_path=train_path, newsgroup_list=newsgroup_list, word_count=word_count)
    vocab = [word for word, freq in zip(
        word_count.keys(), word_count.values()) if freq > 10]  # tập hợp các từ có tần suất xuất hiện không quá bé trong datasets
    vocab.sort()

    with open('../datasets/w2v/vocab_raw.txt', 'w') as file:
        file.write('\n'.join(vocab))

    test_data = collect_data_from(
        parent_path=test_path, newsgroup_list=newsgroup_list)
    
    with open('../datasets/w2v/20news-train-raw.txt', 'w') as file:
        file.write('\n'.join(train_data))
    
    with open('../datasets/w2v/20news-test-raw.txt', 'w') as file:
        file.write('\n'.join(test_data))

if __name__ == '__main__':
    gen_data_and_vocab()
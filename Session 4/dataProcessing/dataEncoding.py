def encode_data(data_path, vocab_path):
    """
    mỗi văn bản được mã hoá bằng cách nhận biết sự có mặt của các word có trong văn bản và thay thế nó bằng ID tương ứng của nó.
    """
    with open(vocab_path) as file:
        vocab = dict([word, word_ID + 2]
                     for word_ID, word in enumerate(file.read().splitlines()))
        unknown_ID = 0
        padding_ID = 1
        # vocab[0] cho unknowing words, vocab[1] cho padding words
    delimiter = '<fff>'
    with open(data_path) as file:
        documents = [(line.split(delimiter)[0], line.split(
            delimiter)[1], line.split(delimiter)[2]) for line in file.read().splitlines()]
    encoded_data = []
    MAX_SENTENCE_LENGTH = 500  # ????
    for document in documents:
        label, doc_id, text = document
        words = text.split(' ')[:MAX_SENTENCE_LENGTH]
        sentence_length = len(words)
        encoded_text = []
        for word in words:
            if word in vocab:
                encoded_text.append(str(vocab[word]))
            else:
                # thêm các từ chưa biết
                encoded_text.append(str(unknown_ID))
        if len(words) < MAX_SENTENCE_LENGTH:
            num_padding = MAX_SENTENCE_LENGTH - len(words)
            for _ in range(num_padding):
                encoded_text.append(str(padding_ID))
        encoded_data.append(label + delimiter + doc_id + delimiter +
                            str(sentence_length) + delimiter + ' '.join(encoded_text))
        # thêm các từ rỗng

    # save in 'w2v/...train-encoded.txt' hoặc '...test-encoded.txt'
    dir_name = '/'.join(data_path.split('/')[:-1])
    filename = '-'.join(data_path.split('/')
                        [-1].split('-')[:-1]) + '-encoded.txt'

    with open(dir_name + '/' + filename, 'w') as f:
        f.write('\n'.join(encoded_data))


if __name__ == '__main__':
    encode_data('../datasets/w2v/20news-train-raw.txt',
                '../datasets/w2v/vocab-raw.txt')
    encode_data('../datasets/w2v/20news-test-raw.txt',
                '../datasets/w2v/vocab-raw.txt')

import numpy as np
import random


class DataReader:
    def __init__(self, data_path, batch_size, vocab_size):
        self._batch_size = batch_size
        self._data = []
        self._labels = []
        self._sequence_lengths = []
        with open(data_path, 'r') as f:
            d_lines = f.read().splitlines()

        for line in d_lines:
            feature = line.split('<fff>')
            label = int(feature[0])
            doc_id = int(feature[1])
            sequence_lengths = int(feature[2])
            tokens = feature[3].split()
            self._data.append(tokens)
            self._labels.append(label)
            self._sequence_lengths.append(sequence_lengths)

        self._data = np.array(self._data)
        self._labels = np.array(self._labels)
        self._sequence_lengths = np.array(self._sequence_lengths)

        # batch information
        self._batch_id = 0
        self._num_epoch = 0

    def next_batch(self):
        start = self._batch_id * self._batch_size
        end = start + self._batch_size
        self._batch_id += 1
        if end > len(self._data):
            end = len(self._data)
            start = end - self._batch_size
            self._num_epoch += 1
            self._batch_id = 0
            indices = list(range(len(self._data)))
            random.seed(2021)
            random.shuffle(indices)
            self._data = self._data[indices]
            self._labels = self._labels[indices]
            self._sequence_lengths = self._sequence_lengths[indices]

        final_tokens = np.array([token[-1]
                                 for token in self._data[start: end]])

        return self._data[start: end], self._labels[start: end], self._sequence_lengths[start:end], final_tokens

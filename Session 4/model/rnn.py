import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import shape
import tensorflow.compat.v1 as tf1
import numpy as np
MAX_DOC_LENGTH = 500
NUM_CLASSES = 20


class RNN:
    def __init__(self, vocab_size, embedding_size, lstm_size, batch_size):
        self._vocab_size = vocab_size
        self._embedding_size = embedding_size  # hidden layer 's size
        self._lstm_size = lstm_size
        self._batch_size = batch_size

        self._data = tf1.placeholder(
            tf.int32, shape=[batch_size, MAX_DOC_LENGTH])
        self._labels = tf1.placeholder(tf.int32, shape=[batch_size, ])
        self._sentence_lengths = tf1.placeholder(
            tf.int32, shape=[batch_size, ])  # Để làm gì?
        self._final_tokens = tf1.placeholder(tf.int32, shape=[batch_size, ])

    def build_graph(self):
        embeddings = self.embedding_layer(self._data)  # first layer
        lstm_outputs = self.lstm_layer(embeddings=embeddings)  # output layer
        weights = tf1.get_variable(
            name='final-layer-weights',
            shape=(self._lstm_size, NUM_CLASSES),
            initializer=tf.random_normal_initializer(seed=2021)
        )

        biases = tf1.get_variable(
            name='final-layer-biases',
            shape=(NUM_CLASSES, ),
            initializer=tf.random_normal_initializer(seed=2021)
        )

        logits = tf.matmul(lstm_outputs, weights) + biases
        labels_one_hot = tf.one_hot(
            indices=self._labels,
            depth=NUM_CLASSES,
            dtype=tf.float32
        )

        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels_one_hot, logits=logits)

        loss = tf.reduce_mean(loss)

        probs = tf.nn.softmax(logits)

        predicted_labels = tf.argmax(probs, axis=1)
        # giảm chiều thừa của predicted labels
        predicted_labels = tf.squeeze(predicted_labels)
        return predicted_labels, loss

    def embedding_layer(self, indices):
        # đọc lại và viết lại
        pretrained_vectors = []
        pretrained_vectors.append(np.zeros(self._embedding_size))
        # bằng 0 cho unknowing word bởi unknowing word ít quan trọng ko ảnh hưởng đến mục tiêu
        # không cần thiết phải random init
        np.random.seed(2021)
        for _ in range(self._vocab_size + 1):
            pretrained_vectors.append(np.random.normal(
                loc=0.0, scale=1., size=self._embedding_size))

        pretrained_vectors = np.array(pretrained_vectors)
        self._embedding_matrix = tf1.get_variable(
            name='embedding',
            shape=(self._vocab_size + 2, self._embedding_size),
            initializer=tf.constant_initializer(pretrained_vectors)
        )
        return tf.nn.embedding_lookup(self._embedding_matrix, indices)

    def lstm_layer(self, embeddings):
        # đọc lại và viết lại
        lstm_cell = tf1.nn.rnn_cell.BasicLSTMCell(self._lstm_size)  # cell init
        zero_state = tf.zeros(shape=(self._batch_size, self._lstm_size))
        initial_state = tf1.nn.rnn_cell.LSTMStateTuple(zero_state, zero_state)

        lstm_inputs = tf.unstack(
            tf.transpose(embeddings, perm=[1, 0, 2])
        )
        lstm_outputs, last_state = tf1.nn.static_rnn(
            cell=lstm_cell,
            inputs=lstm_inputs,
            initial_state=initial_state,
            sequence_length=self._sentence_lengths
        )
        lstm_outputs = tf.unstack(
            tf.transpose(lstm_outputs, perm=[1, 0, 2]),
        )
        lstm_outputs = tf.concat(
            lstm_outputs,
            axis=0
        )
        mask = tf.sequence_mask(
            lengths=self._sentence_lengths,
            maxlen=MAX_DOC_LENGTH,
            dtype=tf.float32
        )
        mask = tf.concat(tf.unstack(mask, axis=0), axis=0)
        mask = tf.expand_dims(mask, -1)
        lstm_outputs = mask * lstm_outputs
        lstm_outputs_split = tf.split(
            lstm_outputs, num_or_size_splits=self._batch_size)
        lstm_outputs_sum = tf.reduce_sum(lstm_outputs_split, axis=1)
        lstm_outputs_average = lstm_outputs_sum / \
            tf.expand_dims(tf.cast(self._sentence_lengths, tf.float32), -1)
        return lstm_outputs_average

    def trainer(self, loss, learning_rate):
        train_op = tf1.train.AdamOptimizer(learning_rate).minimize(loss)
        return train_op

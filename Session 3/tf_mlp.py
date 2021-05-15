import tensorflow as tf
import tensorflow.compat.v1 as tf1
import numpy as np
import random
NUM_CLASSES = 20


class MLP:
    def __init__(self, vocab_size, hidden_size):
        self._vocab_size = vocab_size
        self._hidden_size = hidden_size

    def build_graph(self):
        self._X = tf1.placeholder(tf.float32, shape=[None, self._vocab_size])
        self._real_Y = tf1.placeholder(tf.int32, shape=[None, ])

        # first layers initialization
        weights_1 = tf1.get_variable(name='weights_input_hidden', shape=(
            self._vocab_size, self._hidden_size), initializer=tf.random_normal_initializer())
        biases_1 = tf1.get_variable(name='biases_input_hidden', shape=(
            self._hidden_size), initializer=tf.random_normal_initializer())

        # ouput layers initialization
        weights_2 = tf1.get_variable(name='weight_output_hidden', shape=(
            self._hidden_size, NUM_CLASSES), initializer=tf.random_normal_initializer())
        biases_2 = tf1.get_variable(name='biases_output_hidden', shape=(
            NUM_CLASSES), initializer=tf.random_normal_initializer())

        # build computation graph
        hidden = tf.matmul(self._X, weights_1) + biases_1
        hidden = tf.sigmoid(hidden)

        logits = tf.matmul(hidden, weights_2) + biases_2
        labels_one_hot = tf.one_hot(
            indices=self._real_Y, depth=NUM_CLASSES, dtype=tf.float32)
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels_one_hot, logits=logits)
        loss = tf.reduce_mean(loss)

        # predicted labels
        probs = tf.nn.softmax(logits=logits)
        predicted_labels = tf.argmax(probs, axis=1)
        predicted_labels = tf.squeeze(predicted_labels)
        return predicted_labels, loss

    def trainer(self, loss, learning_rate):
        train_op = tf1.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(loss)
        return train_op


class Data_reader:
    def __init__(self, data_path, batch_size, vocab_size):
        self._batch_size = batch_size
        self._data = []
        self._labels = []
        with open(data_path, 'r') as f:
            d_lines = f.read().splitlines()

        for line in d_lines:
            feature = line.split('<fff>')
            label = int(feature[0])
            doc_id = int(feature[1])
            vector = [0. for _ in range(vocab_size)]
            tokens = feature[2].split()
            for token in tokens:
                indices = token.split(':')
                index, value = int(indices[0]), float(indices[1])
                vector[index] = value
            self._data.append(vector)
            self._labels.append(label)

        self._data = np.array(self._data)
        self._labels = np.array(self._labels)

        # batch information
        self._batch_id = 0
        self._num_epoch = 0

    def next_batch(self):
        start = self._batch_id * self._batch_size
        end = start + self._batch_size
        self._batch_id += 1
        if end + self._batch_size > len(self._data):
            end = len(self._data)
            self._num_epoch += 1
            self._batch_id = 0
            indices = list(range(len(self._data)))
            random.seed(2021)
            random.shuffle(indices)
            self._data = self._data[indices]
            self._labels = self._labels[indices]

        return self._data[start: end], self._labels[start: end]


def save_parameters(name, value, epoch):
    name = name.replace(':', '-colon-')
    filename = f'saved_paras/{name}-epoch-{epoch}.txt'
    if len(value.shape) == 1:
        stringForm = ';'.join([str(number) for number in value])
    else:
        stringForm = '\n'.join([';'.join(
            [str(number) for number in value[row]]) for row in range(value.shape[0])])
    with open(filename, 'w') as fp:
        fp.write(stringForm)


def restore_value(name, epoch):
    name = name.replace(':', '-colon-')
    filename = f'saved_paras/{name}-epoch-{epoch}.txt'
    with open(filename, 'r') as fp:
        lines = fp.read().splitlines()
    if len(lines) == 1:
        value = [float(number) for number in lines[0].split(';')]
    else:
        value = [[float(number) for number in lines[index].split(';')]
                 for index in range(len(lines))]
    return np.array(value)


def train():
    with tf1.Session() as sess:
        train_data_reader = Data_reader(
            data_path='datasets/20news-bydate/train_tf_idf.txt', batch_size=50, vocab_size=vocab_size)
        sess.run(tf1.global_variables_initializer())
        step, MAX_STEP = 0, 10000
        while step < MAX_STEP:
            train_data, train_labels = train_data_reader.next_batch()
            plabels_eval, loss_eval, _ = sess.run(
                [predicted_labels, loss, trainer_op],
                feed_dict={
                    mlp._X: train_data,
                    mlp._real_Y: train_labels
                })

            step += 1
            print('Step: {}, loss: {}'.format(step, loss_eval),
                  f"True predictions: {np.sum(np.equal(plabels_eval, train_labels))}",
                  f'Wrong prediction: {len(train_labels) - np.sum(np.equal(plabels_eval, train_labels))}')
        trainable_variables = tf1.trainable_variables()
        for variable in trainable_variables:
            save_parameters(
                name=variable.name,
                value=variable.eval(),
                epoch=train_data_reader._num_epoch
            )


def test(epoch):
    with tf1.Session() as sess:
        test_data_reader = Data_reader(
            data_path='datasets/20news-bydate/test_tf_idf.txt', vocab_size=vocab_size, batch_size=50)
        trainable_variables = tf1.trainable_variables()
        for variable in trainable_variables:
            saved_value = restore_value(name=variable.name, epoch=epoch)
            assign_op = variable.assign(saved_value)
            sess.run(assign_op)
        num_true_preds = 0
        while True:
            test_data, test_labels = test_data_reader.next_batch()
            test_plabels_eval = sess.run(predicted_labels,
                                         feed_dict={
                                             mlp._X: test_data,
                                             mlp._real_Y: test_labels
                                         })
            matches = np.equal(test_plabels_eval, test_labels)
            print(np.sum(matches) / len(test_labels), len(test_labels))
            num_true_preds += np.sum(matches.astype(float))

            if test_data_reader._batch_id == 0:
                break
        print(f"Epoch: {epoch}")
        print(
            f'Accuracy on test data: {num_true_preds / len(test_data_reader._data)}')


if __name__ == '__main__':
    # initialization
    tf1.disable_eager_execution()
    with open('datasets/20news-bydate/words_idfs.txt') as f:
        vocab_size = len(f.read().splitlines())

    mlp = MLP(vocab_size=vocab_size, hidden_size=50)
    predicted_labels, loss = mlp.build_graph()
    trainer_op = mlp.trainer(loss=loss, learning_rate=0.1)
    train()
    test(44)

import numpy as np


def load_data(datapath):
    def sparse_to_dense(sparse_r_d, vocab_size):
        r_d = [0. for _ in range(vocab_size)]
        indices_tfidfs = sparse_r_d.split()
        for index_tfidfs in indices_tfidfs:
            index, tfidfs = index_tfidfs.split(':')
            index = int(index)
            tfidf = float(tfidfs)
            r_d[index] = tfidf
        return np.array(r_d)
    data = []
    labels = []
    with open(datapath, "r") as fp:
        dlines = fp.read().splitlines()
    with open('../datasets/20news-bydate/words_idfs.txt') as fp:
        vocab_size = len(fp.read().splitlines())
    for line in dlines:
        feature = line.split('<fff>')
        r_d = sparse_to_dense(feature[2], vocab_size=vocab_size)
        label = int(feature[0])
        data.append(r_d)
        labels.append(label)

    return np.array(data), np.array(labels)

def compute_accuracy(predicted_y, expected_y):
    matches = np.equal(predicted_y, expected_y)
    accuracy = np.sum(matches.astype(float)) / expected_y.size
    return accuracy

def clustering_with_Kmeans():
    data, labels = load_data(
        datapath='../datasets/20news-bydate/full_tf_idf.txt')
    from sklearn.cluster import KMeans
    from scipy.sparse import csr_matrix
    X = csr_matrix(data)
    print("=========")
    kmeans = KMeans(n_clusters=20, init='random', n_init=5, tol=1e-3, random_state=2021).fit(X)
    labels = kmeans.labels_
    
def classifying_with_kernel_svms():
    train_X, train_Y = load_data("../datasets/20news-bydate/train_tf_idf.txt");
    from sklearn.svm import SVC
    classifier = SVC(
        C=50.0,
        kernel='rbf',
        gamma=0.1,
        tol=0.001,
        verbose=True
    )
    classifier.fit(train_X, train_Y)
    test_X, test_Y = load_data("../datasets/20news-bydate/test_tf_idf.txt")
    predicted_Y = classifier.predict(test_X)
    accuracy = compute_accuracy(predicted_Y, test_Y)
    print(f'Accuracy: {accuracy}')

def classifying_with_linear_svms():
    train_X, train_Y = load_data("../datasets/20news-bydate/train_tf_idf.txt")
    from sklearn.svm import LinearSVC
    classifier = LinearSVC(
        C=10.0,
        tol=0.01,
        verbose=True
    )
    classifier.fit(train_X, train_Y)
    test_X, test_Y = load_data("../datasets/20news-bydate/test_tf_idf.txt")
    predicted_Y = classifier.predict(test_X)
    accuracy = compute_accuracy(predicted_Y, test_Y)
    print(f'Accuracy: {accuracy}')

if __name__ == "__main__":
    clustering_with_Kmeans()
    classifying_with_linear_svms()
    classifying_with_kernel_svms()
import numpy as np
from eval import load_params
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

#def tsne_plot(model):
def tsne_plot(embeddings,vocab):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for i in range(vocab.size):
        tokens.append(embeddings[i])
        labels.append(vocab[i])

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    #plt.savefig("plot.png")
    plt.show()

def get_embeddings(w1, one_hot_matrix):
    embeddings = []
    for x in one_hot_matrix:
        embeddings.append(np.matmul(x, w1))
    return np.array(embeddings)


if __name__ == '__main__':
    dataset_location = input("Please enter the relative path to your dataset directory, please put slash in the end (Default is \"data/\" , press Enter for default): ")

    if dataset_location == '':
        dataset_location = "data/"
    # Load data
    vocab = np.load(dataset_location + 'vocab.npy')
    one_hot_matrix = np.diag(np.diag(np.ones([250, 250])))
    w1, w2_1, w2_2, w2_3, w3, b1, b2 = load_params()
    embeddings = get_embeddings(w1, one_hot_matrix)
    tsne_plot(embeddings,vocab)
    print("")

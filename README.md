# Next Word Prediction with Multi Layer Perceptron

In this project, I train a neural language model using a multi-layer perceptron given below. This network receives 3 consecutive words as the input and aims to predict the next word. I train this model using cross-entropy loss function, which corresponds to maximizing the probability of the target word.

![alt text](https://github.com/guraybaydur/next-word-prediction/blob/main/architecture.png)

The network consists of a 16 dimensional embedding layer, a 128 dimensional hidden layer and one output layer. The input consists of a sequence of 3 consecutive words, provided as integer valued indices representing a word in our 250-word dictionary. I need to convert each word to it’s one-hot representation and feed it to the embedding layer which will be 250 × 16 dimensional. Hidden layer will have sigmoid loss function and the output layer is a softmax over the 250 words in our dictionary. After the embedding layer, 3 word embeddings are concatenated and fed to the hidden layer.

## Data

Train/Validation/Test splits are provided to you in the project folder. Words in the dictionary can be found in vocab.npy file.

## Visualization

I also created a 2-D plot of the embeddings using t-SNE which maps nearby 16 dimensional embeddings close to each other in the 2-D space. I used of the shelf t-SNE functions. tsne.py file is the file where you load model parameters, return the learned embeddings, and plot t-SNE. 

## Steps to run the codes

Prereqs
1) Please put all *.py files in the same directory

Guidelines
1) For all .py file executions you can specify the location of the dataset(it could be relative or absolute). You can skip with default location which is "data/" via pressing Enter. If you want to override, please put forward slash to the end of the relative path .
2) Run main.py file to train the network. Training accuracy, validation accuracy as well as their respective losses are reported here. Please choose minimum batch size, epoch and learning rate. You can skip with defaults via pressing Enter.
3) Run eval.py file to load the params from model.pk and observe the test accuracy. You will also see the 3 word predictions in the project tasks. If you want to test the prediction for another 3 words you can try it out in the end.
4) Run tsne.py to see the similarity between word embeddings via plot.


## License

This project is licensed under the [afl-3.0] License - see the LICENSE.md file for details



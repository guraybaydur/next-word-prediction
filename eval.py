from main import *

def load_params():
    model = pickle.load(open("model.pk", 'rb'))
    w1 = model["w1"]
    w2_1 = model["w2_1"]
    w2_2 = model["w2_2"]
    w2_3 = model["w2_3"]
    w3 = model["w3"]
    b1 = model["b1"]
    b2 = model["b2"]
    return w1, w2_1, w2_2, w2_3, w3, b1, b2

def predict_word(word1,word2,word3,vocab):
    word1_index = np.where(vocab == word1)[0][0]
    word2_index = np.where(vocab == word2)[0][0]
    word3_index = np.where(vocab == word3)[0][0]

    word1_encoded = get_one_hot_encoding(word1_index, vocab.size)
    word2_encoded = get_one_hot_encoding(word2_index, vocab.size)
    word3_encoded = get_one_hot_encoding(word3_index, vocab.size)

    w1, w2_1, w2_2, w2_3, w3, b1, b2 = load_params()

    # word prediction
    x1_test = np.array(word1_encoded)
    x2_test = np.array(word2_encoded)
    x3_test = np.array(word3_encoded)

    p_test, h2_test, a_test, h1_test, e1_test, e2_test, e3_test, y_head_index_test = forward_propagation(x1_test,
                                                                                                         x2_test,
                                                                                                         x3_test, w1,
                                                                                                         w2_1, w2_2,
                                                                                                         w2_3, w3,
                                                                                                         b1, b2)
    predicted_word = vocab[y_head_index_test]
    return predicted_word

if __name__ == '__main__':
    dataset_location = input("Please enter the relative path to your dataset directory, please put slash in the end (Default is \"data/\" , press Enter for default): ")

    if dataset_location == '':
        dataset_location = "data/"

    vocab = np.load(dataset_location + 'vocab.npy')

    w1, w2_1, w2_2, w2_3, w3, b1, b2 = load_params()

    test_inputs = np.load(dataset_location +'test_inputs.npy')
    test_targets = np.load(dataset_location +'test_targets.npy')

    test_inputs_encoded = encode_input(test_inputs, vocab.size)
    test_targets_encoded = encode_input(test_targets, vocab.size)

    x1 = test_inputs_encoded[:, 0, :]
    x2 = test_inputs_encoded[:, 1, :]
    x3 = test_inputs_encoded[:, 2, :]

    p, h2, a, h1, e1, e2, e3, y_head_index = forward_propagation(x1, x2, x3, w1, w2_1, w2_2, w2_3, w3,
                                                                 b1, b2)

    target_indices_validation = np.argmax(test_targets_encoded, axis=1)
    predicted_indices_validation = y_head_index
    test_accuracy = (np.sum(target_indices_validation == predicted_indices_validation) /
                           test_targets_encoded.shape[0]) * 100
    # calculate loss
    test_loss = cross_entropy_loss(test_targets_encoded, p)
    print("\nTest accuracy: " + str(test_accuracy) )
    print("Test loss: " + str(test_loss))
    print("")

    city_index = np.where(vocab == "city")[0][0]
    of_index = np.where(vocab == "of")[0][0]
    new_index = np.where(vocab == "new")[0][0]
    life_index = np.where(vocab == "life")[0][0]
    in_index = np.where(vocab == "in")[0][0]
    the_index = np.where(vocab == "the")[0][0]
    he_index = np.where(vocab == "he")[0][0]
    is_index = np.where(vocab == "is")[0][0]

    city_encoded = get_one_hot_encoding(city_index,vocab.size)
    of_encoded = get_one_hot_encoding(of_index ,vocab.size)
    new_encoded = get_one_hot_encoding(new_index ,vocab.size)
    life_encoded = get_one_hot_encoding(life_index,vocab.size)
    in_encoded = get_one_hot_encoding(in_index ,vocab.size)
    the_encoded = get_one_hot_encoding(the_index ,vocab.size)
    he_encoded = get_one_hot_encoding(he_index ,vocab.size)
    is_encoded = get_one_hot_encoding(is_index ,vocab.size)

    #city of new prediction
    x1_test = np.array(city_encoded)
    x2_test = np.array(of_encoded)
    x3_test = np.array(new_encoded)

    p_test, h2_test, a_test, h1_test, e1_test, e2_test, e3_test, y_head_index_test = forward_propagation(x1_test, x2_test, x3_test, w1, w2_1, w2_2, w2_3, w3,
                                                                 b1, b2)
    predicted_word = vocab[y_head_index_test]

    print("city of new " + predicted_word)
    # life in the prediction
    x1_test = np.array(life_encoded)
    x2_test = np.array(in_encoded)
    x3_test = np.array(the_encoded)

    p_test, h2_test, a_test, h1_test, e1_test, e2_test, e3_test, y_head_index_test = forward_propagation(x1_test,
                                                                                                         x2_test,
                                                                                                         x3_test, w1,
                                                                                                         w2_1, w2_2,
                                                                                                         w2_3, w3,
                                                                                                         b1, b2)

    predicted_word = vocab[y_head_index_test]

    print("life in the " + predicted_word)
    # he is the prediction
    x1_test = np.array(he_encoded)
    x2_test = np.array(is_encoded)
    x3_test = np.array(the_encoded)

    p_test, h2_test, a_test, h1_test, e1_test, e2_test, e3_test, y_head_index_test = forward_propagation(x1_test,
                                                                                                         x2_test,
                                                                                                         x3_test, w1,
                                                                                                         w2_1, w2_2,
                                                                                                         w2_3, w3,
                                                                                                         b1, b2)

    predicted_word = vocab[y_head_index_test]
    print("he is the " + predicted_word + "\n")


    answer = input("Do you want to predict a new word? (Yes or yes or y or Y / otherwise type anything):  ")

    if answer == "Yes" or answer == "yes" or answer == "y" or answer == "Y":
        words = input("Please enter 3 consecutive words to predict (please put spaces between them): ")
        word1, word2, word3 = words.split(" ")
        predicted_word = predict_word(word1,word2,word3,vocab)
        print(word1 + " " + word2 + " " + word3 + " " +predicted_word)

    print("End of eval.py")

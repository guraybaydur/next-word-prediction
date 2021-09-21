# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import pickle
from Network import *


def get_one_hot_encoding(index, size):
    one_hot_vector = np.zeros(size)
    one_hot_vector[index] = 1
    return one_hot_vector


def encode_input(input, size):
    if len(input.shape) == 2:
        input_encoded = []
        for i in range(0, len(input)):
            input_encoded.append([])
            for j in range(0, len(input[i])):
                input_encoded[i].append(get_one_hot_encoding(input[i][j], size))
        input_encoded = np.array(input_encoded)
    elif len(input.shape) == 1:
        input_encoded = []
        for i in range(0, len(input)):
            input_encoded.append(get_one_hot_encoding(input[i], size))
        input_encoded = np.array(input_encoded)
    else:
        raise Exception("There is a problem with dimensions!!!")
    return input_encoded

def cross_entropy_loss(y, p):
    #cost = -(1.0 / p.shape[1]) * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    cost = -(1.0 / p.shape[0]) * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    # return (1 / y_head.shape[1]) * (-1) * np.sum(np.multiply(y, np.log(y_head)))
    return cost

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Welcome to CMPE 597 Sp. Tp. Deep Learning Spring 2021 Project I")
    dataset_location = input("Please enter the relative path to your dataset directory, please put slash in the end (Default is \"data/\" , press Enter for default): ")

    if dataset_location == '':
        dataset_location = "data/"

    mbs = input("Please enter the minibatch size for training (Default is 64, press Enter for default): ")

    # Minibatch size
    if mbs == '':
        mbs = 64
    else:
        mbs = int(mbs)

    # epoch
    epoch = input("Please enter the number of epochs (Default is 5, press Enter for default): ")
    if epoch == '':
        epoch = 5
    else:
        epoch = int(epoch)

    # learning rate
    learning_rate = input("Please enter the learning_rate (Default is 0.01, press Enter for default): ")
    if learning_rate == '':
        learning_rate = 0.01
    else:
        learning_rate = int(learning_rate)

    # Load data
    print("Loading vocab...")
    vocab = np.load(dataset_location + "vocab.npy")

    print("Loading train and validation datasets... This might take some time... Thank you for your patience!")
    train_inputs = np.load(dataset_location + 'train_inputs.npy')
    train_targets = np.load(dataset_location + 'train_targets.npy')
    #test_inputs = np.load('data/test_inputs.npy')
    #test_targets = np.load('data/test_targets.npy')
    valid_inputs = np.load(dataset_location + 'valid_inputs.npy')
    valid_targets = np.load(dataset_location + 'valid_targets.npy')
    print("Completed Loading train and validation datasets...")

    print("One Hot Encoding train and validation datasets... This might take some time... Thank you for your patience!")
    train_inputs_encoded = encode_input(train_inputs, vocab.size)
    #np.save("data/train_inputs_encoded.npy", train_inputs_encoded)
    train_targets_encoded = encode_input(train_targets, vocab.size)
    #np.save("data/train_targets_encoded.npy", train_targets_encoded)

    valid_inputs_encoded = encode_input(valid_inputs, vocab.size)
    #np.save("data/valid_inputs_encoded.npy", valid_inputs_encoded)
    valid_targets_encoded = encode_input(valid_targets, vocab.size)
    print("Completed One Hot Encoding train and validation datasets...")
    #np.save("data/valid_targets_encoded.npy", valid_targets_encoded)

    # train_inputs_encoded = np.load("data/train_inputs_encoded.npy")
    # train_targets_encoded = np.load("data/train_targets_encoded.npy")
    # test_inputs_encoded = np.load("data/train_inputs_encoded.npy")
    # test_targets_encoded = np.load("data/train_targets_encoded.npy")
    # valid_inputs_encoded = np.load("data/valid_inputs_encoded.npy")
    # valid_targets_encoded = np.load("data/valid_targets_encoded.npy")

    # Shuffle data
    np.random.seed(42)
    # shuffler = np.random.permutation(len(train_inputs_encoded))
    # train_inputs_shuffled = train_inputs_encoded[shuffler]
    # train_targets_shuffled = train_targets_encoded[shuffler]
    # shuffler2 = np.random.permutation(len(valid_inputs_encoded))
    # valid_inputs_shuffled = valid_inputs_encoded[shuffler2]
    # valid_targets_shuffled = valid_targets_encoded[shuffler2]

    size = train_targets_encoded.shape[0]


    # Divide into minibatches
    num_of_minibatches = math.floor(size / mbs) + 1
    # minibatches = []
    # target_minibatches = []
    #
    # for i in range(0, num_of_minibatches):
    #     if i != num_of_minibatches - 1:
    #         minibatch = train_inputs_shuffled[i * mbs:i * mbs + mbs]
    #         minibatches.append(np.array(minibatch))
    #         target_minibatch = train_targets_shuffled[i * mbs:i * mbs + mbs]
    #         target_minibatches.append(np.array(target_minibatch))
    #     else:
    #         minibatch = train_inputs_shuffled[i * mbs:size]
    #         minibatches.append(np.array(minibatch))
    #         target_minibatch = train_targets_shuffled[i * mbs:size]
    #         target_minibatches.append(np.array(target_minibatch))
    #
    # target_minibatches = np.array(target_minibatches,dtype=object)
    # minibatches = np.array(minibatches,dtype=object)
    #print('a')
    # Divide dataset into mini batches
    # Shuffle data
    # shuffler = np.random.permutation(len(train_inputs_encoded))
    # train_inputs_encoded_shuffled = train_inputs_encoded[shuffler]
    # test_inputs_encoded_shuffled = test_inputs_encoded[shuffler]

    print("Initializing weights and biases...")
    # Initialize the weights with gaussian distribution
    mu, sigma = 0, 1  # mean and standard deviation
    w1 = np.random.normal(mu, sigma, (250, 16))
    w2_1 = np.random.normal(mu, sigma, (16, 128))
    w2_2 = np.random.normal(mu, sigma, (16, 128))
    w2_3 = np.random.normal(mu, sigma, (16, 128))
    w3 = np.random.normal(mu, sigma, (128, 250))

    # zero bias initialization
    b1 = np.zeros(128)
    b2 = np.zeros(250)
    # b1 = np.empty(shape=(250, size))
    # b2 = np.empty(shape=(250, size))

    # b1.fill(0)
    # b2.fill(0)

    """
    # Plot w1 weight distribution
    count, bins, ignored = plt.hist(w1, 30, density=True)
    plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
             np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)),
             linewidth=2, color='r')
    plt.show()
    """
    print("Starting to train...")
    # Optimization part
    start_time = time.time()


    filename = "model.pk"
    for i in range(epoch):
        # Shuffle Data
        shuffler = np.random.permutation(len(train_inputs_encoded))
        train_inputs_shuffled = train_inputs_encoded[shuffler]
        train_targets_shuffled = train_targets_encoded[shuffler]

        # Divide into minibatches
        minibatches = []
        target_minibatches = []
        for k in range(0, num_of_minibatches):
            if k != num_of_minibatches - 1:
                minibatch = train_inputs_shuffled[k * mbs:k * mbs + mbs]
                minibatches.append(np.array(minibatch))
                target_minibatch = train_targets_shuffled[k * mbs:k * mbs + mbs]
                target_minibatches.append(np.array(target_minibatch))
            else:
                minibatch = train_inputs_shuffled[k * mbs:size]
                minibatches.append(np.array(minibatch))
                target_minibatch = train_targets_shuffled[k * mbs:size]
                target_minibatches.append(np.array(target_minibatch))

        target_minibatches = np.array(target_minibatches, dtype=object)
        minibatches = np.array(minibatches, dtype=object)

        for j in range(num_of_minibatches):
            #print(j)
            #loss, y_head, z_2, z_1, z_0, g_1 = forward_propagation(minibatches[j], w1, w2, w3, b1, b2, target_minibatches[j])

            x1 = minibatches[j][:, 0, :]
            x2 = minibatches[j][:, 1, :]
            x3 = minibatches[j][:, 2, :]
            # Forward prop for single datapoint
            p, h2, a, h1, e1, e2, e3, y_head_index = forward_propagation(x1, x2, x3, w1, w2_1, w2_2, w2_3, w3,
                                                                         b1, b2)
            # y_head = get_one_hot_encoding(y_head_index,size)
            # predicted_words = vocab[y_head_index]

            # calculate loss
            #loss = cross_entropy_loss(target_minibatches[j], p)

            # backward prop
            dL_dw3, dL_db2, dL_dw2_1, dL_dw2_2, dL_dw2_3, dL_db1, dL_dw1 = backward_propagation(p, target_minibatches[j], a, w3, w2_1,
                                                                                                w2_2, w2_3, x1,
                                                                                                x2, x3, h1, e1,
                                                                                                e2, e3)
            w1 = w1 - learning_rate * dL_dw1
            w2_1 = w2_1 - learning_rate * dL_dw2_1
            w2_2 = w2_2 - learning_rate * dL_dw2_2
            w2_3 = w2_3 - learning_rate * dL_dw2_3
            w3 = w3 - learning_rate * dL_dw3
            b1 = b1 - learning_rate * dL_db1
            b2 = b2 - learning_rate * dL_db2

        # Training Accuracy calculation
        x1 = train_inputs_shuffled[:, 0, :]
        x2 = train_inputs_shuffled[:, 1, :]
        x3 = train_inputs_shuffled[:, 2, :]

        p, h2, a, h1, e1, e2, e3, y_head_index = forward_propagation(x1, x2, x3, w1, w2_1, w2_2, w2_3, w3,
                                                                     b1, b2)

        target_indices = np.argmax(train_targets_shuffled, axis=1)
        predicted_indices = y_head_index
        training_accuracy = (np.sum(target_indices == predicted_indices) / size) * 100
        # calculate loss
        loss = cross_entropy_loss(train_targets_shuffled, p)
        print("Epoch: " + str(i+1) + " Training accuracy: " + str(training_accuracy) + " Batch Size: " + str(mbs))
        print("loss: " + str(loss))
        #print("\n")

        # Validation Accuracy calculation
        x1 = valid_inputs_encoded[:, 0, :]
        x2 = valid_inputs_encoded[:, 1, :]
        x3 = valid_inputs_encoded[:, 2, :]
        p, h2, a, h1, e1, e2, e3, y_head_index = forward_propagation(x1, x2, x3, w1, w2_1, w2_2, w2_3, w3,
                                                                     b1, b2)
        target_indices_validation = np.argmax(valid_targets_encoded, axis=1)
        predicted_indices_validation = y_head_index
        validation_accuracy = (np.sum(target_indices_validation == predicted_indices_validation) / valid_targets_encoded.shape[0]) * 100
        # calculate loss
        validation_loss = cross_entropy_loss(valid_targets_encoded, p)
        print("Epoch: " + str(i+1) + " Validation accuracy: " + str(validation_accuracy) + " Batch Size: " + str(mbs))
        print("validation loss: " + str(validation_loss))
        #print("\n")


        #
        # Training Accuracy calculation
        # loss, y_head, z_2, z_1, z_0, g_1 = forward_propagation(train_inputs_shuffled, w1, w2, w3, b1, b2,
        #                                                        train_targets_shuffled)
        # target_indices = np.argmax(train_targets_shuffled, axis=1)
        # predicted_indices = np.argmax(y_head.transpose(), axis=1)
        # training_accuracy = (np.sum(target_indices == predicted_indices) / size) * 100
        # print("Epoch: " + str(i+1) + " Training accuracy: " + str(training_accuracy) + " Batch Size: " + str(mbs))
        # print("loss: " + str(loss))
        # #print("\n")
        #
        # #Validation Accuracy calculation
        # validation_loss, y_head_validation, _, _, _, _ = forward_propagation(valid_inputs_encoded, w1, w2, w3, b1, b2,
        #                                                        valid_targets_encoded)
        # target_indices_validation = np.argmax(valid_targets_encoded, axis=1)
        # predicted_indices_validation = np.argmax(y_head_validation.transpose(), axis=1)
        # validation_accuracy = (np.sum(target_indices_validation == predicted_indices_validation) / size) * 100
        # print("Epoch: " + str(i+1) + " Validation accuracy: " + str(validation_accuracy) + " Batch Size: " + str(mbs))
        # print("validation_loss: " + str(validation_loss))
        # #print("\n")

        # for j in range(num_of_minibatches):

        # loss, y_head, z_2, z_1, z_0, g_1 = forward_propagation(minibatches[j], w1, w2, w3, b1, b2,
        #                                                        target_minibatches[j])
        # L_wrt_w1_transpose, L_wrt_w2_transpose, L_wrt_w3_transpose, L_wrt_b1, L_wrt_b2 = backward_propagation(
        #     minibatches[j], w1, w2, w3, b1, b2,
        #     target_minibatches[j], y_head, z_2, z_1, z_0, g_1)

        # w1 = w1 - learning_rate * L_wrt_w1_transpose.transpose()
        # w2 = w2 - learning_rate * L_wrt_w2_transpose.transpose()
        # w3 = w3 - learning_rate * L_wrt_w3_transpose.transpose()
        # b1 = b1 - learning_rate * L_wrt_b1.transpose()
        # b2 = b2 - learning_rate * L_wrt_b2
        # b1[:, j * mbs:(j + 1) * mbs] = b1[:, j * mbs:(j + 1) * mbs] - learning_rate * L_wrt_b1.transpose()
        # b2[:, j * mbs:(j + 1) * mbs] = b2[:, j * mbs:(j + 1) * mbs] - learning_rate * L_wrt_b2

        # else:
        #     # print('k')
        #     #b1_for_last_batch = b1[:, j * mbs:j * mbs + size - ((num_of_minibatches - 1) * mbs)]
        #     #b2_for_last_batch = b2[:, j * mbs:j * mbs + size - ((num_of_minibatches - 1) * mbs)]
        #     loss, y_head, z_2, z_1, z_0, g_1 = forward_propagation(minibatches[j], w1, w2, w3, b1,
        #                                                            b2,
        #                                                            target_minibatches[j])
        #     L_wrt_w1_transpose, L_wrt_w2_transpose, L_wrt_w3_transpose, L_wrt_b1, L_wrt_b2 = backward_propagation(
        #         minibatches[j], w1, w2, w3, b1, b2, target_minibatches[j], y_head,
        #         z_2, z_1, z_0, g_1)
        #
        #     w1 = w1 - learning_rate * L_wrt_w1_transpose.transpose()
        #     w2 = w2 - learning_rate * L_wrt_w2_transpose.transpose()
        #     w3 = w3 - learning_rate * L_wrt_w3_transpose.transpose()
        #     b1 = b1 - learning_rate * L_wrt_b1.transpose()
        #     b2 = b2 - learning_rate * L_wrt_b2
        #     #b1[:, j * mbs:j * mbs + size - ((num_of_minibatches - 1) * mbs)] = b1[:, j * mbs:j * mbs + size - (
        #     #            (num_of_minibatches - 1) * mbs)] - learning_rate * L_wrt_b1.transpose()
        #     #b2[:, j * mbs:j * mbs + size - ((num_of_minibatches - 1) * mbs)] = b2[:, j * mbs:j * mbs + size - (
        #     #            (num_of_minibatches - 1) * mbs)] - learning_rate * L_wrt_b2


        #############################
        # divide x to x1 x2 x3
        #x1 = train_inputs_shuffled[:, 0, :]
        #x2 = train_inputs_shuffled[:, 1, :]
        #x3 = train_inputs_shuffled[:, 2, :]
        #y = train_targets_shuffled

        ## Forward prop for single datapoint
        #p, h2, a, h1, e1, e2, e3, y_head_index = forward_propagation(x1[0], x2[0], x3[0], w1, w2_1, w2_2, w2_3, w3, b1, b2)

        ##y_head = get_one_hot_encoding(y_head_index,size)
        #predicted_word = vocab[y_head_index]

        ## calculate loss
        #loss = cross_entropy_loss(y[0],p)

        #dL_dw3, dL_db2, dL_dw2_1, dL_dw2_2, dL_dw2_3 ,dL_db1, dL_dw1 = backward_propagation(p, y[0], a, w3, w2_1, w2_2, w2_3, x1[0], x2[0], x3[0], h1, e1, e2, e3)
        #w1 = w1 - learning_rate * dL_dw1
        #w2_1 = w2_1 - learning_rate * dL_dw2_1
        #w2_2 = w2_2 - learning_rate * dL_dw2_2
        #w2_3 = w2_3 - learning_rate * dL_dw2_3
        #w3 = w3 - learning_rate * dL_dw3
        #b1 = b1 - learning_rate * dL_db1
        #b2 = b2 - learning_rate * dL_db2
        ##############################3

        print("")


    print("Training Completed!!!")
    print("Saving model as pickle file...")
    pickle.dump({"w1": w1, "w2_1": w2_1, "w2_2": w2_2, "w2_3": w2_3, "w3": w3, "b1": b1, "b2": b2}, open(filename, 'wb'))
    print("Completed Saving model as pickle file...")
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Num of Epochs: " + str(epoch), ' Batch size:' + str(mbs), ' Time elapsed: ' + str(time.time() - start_time),
          ' Sample Size: ' + str(size))

    # loss, y_head, z_2, z_1, z_0, g_1 = forward_propagation(minibatches[0], w1, w2, w3, b1, b2, target_minibatches[0])
    # L_wrt_w1_transpose, L_wrt_w2_transpose, L_wrt_w3_transpose, L_wrt_b1, L_wrt_b2 = backward_propagation(
    #    minibatches[0], w1, w2, w3, b1, b2, target_minibatches[0], y_head, z_2, z_1, z_0, g_1)
    #
    # learning_rate = 0.01
    # w1 = w1 - learning_rate * L_wrt_w1_transpose.transpose()
    # w2 = w2 - learning_rate * L_wrt_w2_transpose.transpose()
    # w3 = w3 - learning_rate * L_wrt_w3_transpose.transpose()
    # b1 = b1 - learning_rate * L_wrt_b1.transpose()
    # b2 = b2 - learning_rate * L_wrt_b2
    #
    # loss2, y_head, z_2, z_1, z_0, g_1 = forward_propagation(minibatches[0], w1, w2, w3, b1, b2, target_minibatches[0])
    # L_wrt_w1_transpose, L_wrt_w2_transpose, L_wrt_w3_transpose, L_wrt_b1, L_wrt_b2 = backward_propagation(
    #    minibatches[0], w1, w2, w3, b1, b2, target_minibatches[0], y_head, z_2, z_1, z_0, g_1)
    #
    # print("loss:" + str(loss))
    # print("loss2:" + str(loss2))


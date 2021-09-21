import numpy as np


def sigmoid(x):
    # return 1 / (1 + np.exp(-x))
    return np.where(x < 0, np.exp(x) / (1 + np.exp(x)), 1 / (1 + np.exp(-x)))


def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum()


def stable_softmax(x):
    z = x - np.max(x, axis=-1, keepdims=True)
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=-1, keepdims=True)
    softmax = numerator / denominator
    return softmax





def forward_propagation(x1,x2,x3, w1, w2_1,w2_2, w2_3, w3, b1, b2):

    # multiply each xi with w1
    e1 = np.matmul(x1,w1)
    e2 = np.matmul(x2,w1)
    e3 = np.matmul(x3,w1)

    # concat multiplication results for all xi, in other words, overall word embedding layer output
    if len(e1.shape) == 1:
        e = np.concatenate((e1, e2, e3))
    else:
        e = np.concatenate((e1, e2, e3),axis=1)

    # w2 matrix creation from w2_i's
    w2 = np.concatenate((w2_1, w2_2, w2_3))

    # First hidden layer h1
    h1 = np.matmul(e,w2) + b1

    # Sigmoid activation of first layer
    a = sigmoid(h1)

    # Second hidden layer
    h2 = np.matmul(a,w3) + b2

    # Softmax activation of second layer
    p = stable_softmax(h2)

    if len(e1.shape) == 1:
        y_head_index = np.argmax(p)
    else:
        y_head_index = np.argmax(p,axis=1)

    # z_1_without_b1 = np.matmul(w2.transpose(), z_0)
    # z_1 = z_1_without_b1 + b1[:, None]
    #
    # g_1 = sigmoid(z_1)
    #
    # z_2_without_b2 = np.matmul(w3.transpose(), g_1)
    # z_2 = z_2_without_b2 + b2[:, None]
    #
    # # y_head = softmax(z_2)
    # y_head = stable_softmax(z_2)
    #
    # loss = cross_entropy_loss(y.transpose(), y_head)

    # print('b')
    return p, h2, a, h1, e1, e2, e3, y_head_index


def backward_propagation(p, y, a, w3, w2_1, w2_2, w2_3, x1, x2, x3, h1, e1, e2, e3):
    # # calculate gradient of softmax
    # grad_of_softmax = y_head - y.transpose()
    #
    # # calculate gradient of w3 transpose
    # z_2_wrt_w3_transpose = g_1.transpose()
    # L_wrt_w3_transpose = np.matmul(grad_of_softmax, z_2_wrt_w3_transpose)
    #
    # # calculate gradient of b2
    # z_2_wrt_b2 = 1
    # # L_wrt_b2 = grad_of_softmax * z_2_wrt_b2
    # L_wrt_b2 = grad_of_softmax.mean(axis=1) * z_2_wrt_b2
    # # print('x')
    #
    # # calculate gradient of w2 transpose
    # L_wrt_g_1 = np.dot(grad_of_softmax.transpose(), w3.transpose())
    # g_1_wrt_z_1 = sigmoid(z_1) * (1 - sigmoid(z_1))
    # L_wrt_z_1 = L_wrt_g_1 * g_1_wrt_z_1.transpose()
    # z_1_wrt_w2_transpose = z_0.transpose()
    # L_wrt_w2_transpose = np.dot(L_wrt_z_1.transpose(), z_1_wrt_w2_transpose)
    #
    # # calculate gradient of b1
    # z_2_wrt_b1 = 1
    # L_wrt_b1 = L_wrt_z_1.transpose().mean(axis=1) * z_2_wrt_b1
    # # L_wrt_b1 = L_wrt_z_1 * z_2_wrt_b1
    #
    # # calculate gradient of w1 transpose
    # mbs = int(w2.shape[0] / 3)
    # w2_1 = w2[:mbs, :]
    # w2_2 = w2[mbs:2 * mbs, :]
    # w2_3 = w2[2 * mbs:3 * mbs, :]
    #
    # L_wrt_z_0_1 = np.dot(L_wrt_z_1, w2_1.transpose())
    # z_0_wrt_w1_transpose_1 = x[:, 0, :]
    # L_wrt_w1_transpose_1 = np.dot(L_wrt_z_0_1.transpose(), z_0_wrt_w1_transpose_1)
    #
    # L_wrt_z_0_2 = np.dot(L_wrt_z_1, w2_2.transpose())
    # z_0_wrt_w1_transpose_2 = x[:, 1, :]
    # L_wrt_w1_transpose_2 = np.dot(L_wrt_z_0_2.transpose(), z_0_wrt_w1_transpose_2)
    #
    # L_wrt_z_0_3 = np.dot(L_wrt_z_1, w2_3.transpose())
    # z_0_wrt_w1_transpose_3 = x[:, 2, :]
    # L_wrt_w1_transpose_3 = np.dot(L_wrt_z_0_3.transpose(), z_0_wrt_w1_transpose_3)
    #
    # L_wrt_w1_transpose = L_wrt_w1_transpose_1 + L_wrt_w1_transpose_2 + L_wrt_w1_transpose_3
    # return L_wrt_w1_transpose, L_wrt_w2_transpose, L_wrt_w3_transpose, L_wrt_b1, L_wrt_b2

    # Compute gradient of L wrt h2
    dL_dh2 = p - y

    # Compute gradient of w3
    if len(a.shape) == 1:
        dL_dw3 = np.matmul(np.array([a]).T,np.array([dL_dh2]))
    else:
        dL_dw3 = np.matmul(a.T,dL_dh2)

    # Compute gradient of b2
    dL_db2 = dL_dh2.mean(axis=0)

    # Compute gradient of w2_i's
    dh2_da = w3

    da_dh1 = sigmoid(h1) * (1-sigmoid(h1))

    dh1_dw2_1 = e1
    dh1_dw2_2 = e2
    dh1_dw2_3 = e3

    dL_da = np.matmul(dL_dh2,dh2_da.T)
    dL_dh1 = dL_da * da_dh1

    if len(dh1_dw2_1.shape) == 1:
        dL_dw2_1 = np.matmul(np.array([dh1_dw2_1]).T,np.array([dL_dh1]))
        dL_dw2_2 = np.matmul(np.array([dh1_dw2_2]).T, np.array([dL_dh1]))
        dL_dw2_3 = np.matmul(np.array([dh1_dw2_3]).T, np.array([dL_dh1]))
    else:
        dL_dw2_1 = np.matmul(dh1_dw2_1.T, dL_dh1)
        dL_dw2_2 = np.matmul(dh1_dw2_2.T, dL_dh1)
        dL_dw2_3 = np.matmul(dh1_dw2_3.T, dL_dh1)

    # Compute gradient of b1
    dL_db1 = dL_dh1.mean(axis=0)

    # Compute gradient of w1
    dL_de1 = np.matmul(dL_dh1,w2_1.T)

    if len(x1.shape) == 1:
        dL_dw1_1 = np.matmul(np.array([x1]).T,np.array([dL_de1]))
    else:
        dL_dw1_1 = np.matmul(x1.T, dL_de1)

    dL_de2 = np.matmul(dL_dh1, w2_2.T)

    if len(x2.shape) == 1:
        dL_dw1_2 = np.matmul(np.array([x2]).T, np.array([dL_de2]))
    else:
        dL_dw1_2 = np.matmul(x2.T, dL_de2)

    dL_de3 = np.matmul(dL_dh1, w2_3.T)

    if len(x3.shape) == 1:
        dL_dw1_3 = np.matmul(np.array([x3]).T, np.array([dL_de3]))
    else:
        dL_dw1_3 = np.matmul(x3.T, dL_de3)

    dL_dw1 = dL_dw1_1 + dL_dw1_2 + dL_dw1_3

    return dL_dw3, dL_db2, dL_dw2_1, dL_dw2_2, dL_dw2_3 ,dL_db1, dL_dw1
from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    num_class = W.shape[1]
    for i in xrange(num_train):
        score = X[i] @ W
        score = np.exp(score - np.max(score))
        loss_i = -np.log(score[y[i]] / np.sum(score))
        loss = loss + loss_i
        for j in xrange(num_class):
            if j == y[i]:
                dW[:, j] += (-1 + score[j] / np.sum(score)) * X[i]
            else:
                dW[:, j] += score[j] / np.sum(score) * X[i]

    loss = loss / num_train
    loss = loss + 0.5 * reg * np.sum(W * W)
    dW = dW / num_train
    dW = dW + reg * W

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    # num_class=W.shape[1]
    score = X @ W
    score = np.exp(score - np.max(score, axis=1, keepdims=True))
    loss = np.sum(-np.log(score[range(num_train), y] / np.sum(score, axis=1)))
    loss = loss / num_train
    loss = loss + 0.5 * reg * np.sum(W * W)

    dS = score / np.sum(score, axis=1, keepdims=True)
    dS[range(num_train), y] = dS[range(num_train), y] - 1
    dW = X.T @ dS
    dW = dW / num_train
    dW = dW + reg * W

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

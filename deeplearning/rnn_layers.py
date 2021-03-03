import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    N, D = x.shape
    N, H = prev_h.shape
    next_h = np.zeros(shape = (N, H))
    for i in range(0, N) :
        prev_i = prev_h[i, :]
        x_i = x[i, :]
        prev_i_forward = prev_i @ Wh
        x_i_forward = x_i @ Wx
        new_forward = prev_i_forward + x_i_forward + b
        next_h[i, :] = new_forward
    a_next = next_h
    next_h = np.tanh(next_h)
    cache = (x, prev_h, Wx, Wh, b, a_next)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    # Set the vars
    N, H = dnext_h.shape
    x, prev_h, Wx, Wh, b, a_next = cache
    N, D = x.shape
    # First, derive wrt tanh
    d_tanh = dnext_h * (1 - ((np.tanh(a_next))** 2))
    # now backprop through the +
    db = np.sum(d_tanh, axis = 0)
    # dx = d_tanh (N*H) @ Wx.T (H, D)
    dx = d_tanh @  Wx.T
    # dprev_h = d_tanh (n*h) @ Wh.T (H, H)
    dprev_h = d_tanh @ Wh.T
    #dWx = sum over N dim d_tanh[i, :] (h*1) @ x[i, :] (1 * d)
    dWx = np.zeros((D, H))
    for i in range(0, N) :
        x_i = x[i, :].reshape((D, 1))
        d_tanh_i = d_tanh[i, :].reshape((1, H))
        dWx += (x_i @ d_tanh_i)
    #dWx = sum over N dim d_tanh[i, :] (h*1) @ prev[i, :] (1 * h)
    dWh = np.zeros((H,H))
    for i in range(0, N) :
        prev_i = prev_h[i, :].reshape((H, 1))
        d_tanh_i = d_tanh[i, :].reshape((1, H))
        dWh += (prev_i @ d_tanh_i)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    N, T, D = x.shape
    N, H = h0.shape
    curr_h = h0
    curr_x = np.zeros((N,D))
    h = np.zeros((N, T, H))
    cache = {}
    for t in range(0, T) :
        curr_x = x[:, t, :]
        next_h, cache[t] = rnn_step_forward(curr_x, curr_h, Wx, Wh, b)
        curr_h = next_h
        h[:, t, :] = curr_h
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H)

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    N, T, H = dh.shape
    x1, prev_h1, Wx1, Wh1, b1, a_next1 = cache[0]
    D, H = Wx1.shape
    dh_curr = np.zeros((N, H))
    dx = np.zeros((N, T, D))
    dWx = np.zeros((D, H))
    dWh = np.zeros((H, H))
    db = np.zeros(H)
    for t in range(T-1, -1, -1) :
        dh_curr += dh[:, t, :]
        cache_t = cache[t]
        dx_curr, dprev_h_curr, dWx_curr, dWh_curr, db_curr = rnn_step_backward(dh_curr, cache_t)
        dx[:, t, :] = dx_curr
        dWx += dWx_curr
        dWh += dWh_curr
        db += db_curr
        dh_curr = dprev_h_curr
    dh0 = dh_curr
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using NumPy's array indexing.           #
    ##############################################################################
    N, T = x.shape
    V, D = W.shape
    out = np.zeros((N, T, D))
    for n in range(0, N) :
        for t in range(0, T) :
            out[n, t, :] = W[x[n, t], :]
    cache = (x, W)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that Words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    x, W = cache
    V, D = W.shape
    dW = np.zeros((V, D))
    np.add.at(dW, x, dout)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    N, D = x.shape
    N, H = prev_h.shape
    next_c = np.zeros((N, H))
    next_h = np.zeros((N, H))
    f_s = np.zeros((N,H))
    i_s = np.zeros((N,H))
    g_s = np.zeros((N,H))
    o_s = np.zeros((N,H))
    for i in range(0, N) :
        prev_c_i = prev_c[i, :]
        #process prev_h
        prev_h_i = prev_h[i, :] # 1 X H
        x_i = x[i, :] # 1 x D
        exploded_prev_h_i = (prev_h_i @ Wh)
        exploded_x_i = (x_i @ Wx)
        exploded_vec = (b + exploded_prev_h_i + exploded_x_i).reshape(4*H)
        # assigning the parts of the vector
        ihat = exploded_vec[0:H]
        fhat = exploded_vec[H:(2*H)]
        ohat = exploded_vec[(2*H):(3*H)]
        ghat = exploded_vec[(3*H):(4*H)]
        f_vec = sigmoid(fhat)
        i_vec = sigmoid(ihat)
        g_vec = np.tanh(ghat)
        o_vec = sigmoid(ohat)
        f_s[i, :] = f_vec
        i_s[i, :] = i_vec
        g_s[i, :] = g_vec
        o_s[i, :] = o_vec
        next_a = (prev_c_i * f_vec) + (g_vec * i_vec)
        next_c[i, :] = next_a
        next_a_nonlin_i = np.tanh(next_a) * o_vec
        next_h[i, :] = next_a_nonlin_i
    cache = (x, prev_h, prev_c, Wx, Wh, b, f_s, i_s, g_s, o_s, next_h)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    x, prev_h, prev_c, Wx, Wh, b, f_s, i_s, g_s, o_s, next_h = cache
    N, D = x.shape
    N, H = prev_h.shape
    next_c_nonlin = next_h / o_s
    # Initial grads
    dnext_c_nonlin = (1 - (next_c_nonlin ** 2))
    dnext_c_nonlin_prod = dnext_h * o_s * dnext_c_nonlin + dnext_c
    # initialize
    dx = np.zeros((N, D))
    dprev_h = np.zeros((N, H))
    dprev_c = np.zeros((N, H))
    dWx = np.zeros((D, 4*H))
    dWh = np.zeros((H, 4*H))
    db = np.zeros(4*H)
    # With respect to i,f,o,g
    dfhat = (f_s)*(1 - f_s)*prev_c*dnext_c_nonlin_prod # sigmoid
    dihat = (i_s)*(1 - i_s)*g_s*dnext_c_nonlin_prod #sig
    dohat = (o_s)*(1 - o_s)*dnext_h*next_c_nonlin #sig
    dghat = (1 - (g_s ** 2))*dnext_c_nonlin_prod*i_s #tanh
    dexploded_vec = np.concatenate((dihat,dfhat, dohat, dghat), axis = 1)
    # dx and dprev_h
    dx = dexploded_vec @ Wx.T # (N, 4H)(4H, D)
    dprev_h = dexploded_vec @ Wh.T # (N, 4H) (4H, H)
    dprev_c = f_s * dnext_c_nonlin_prod
    db = np.sum(dexploded_vec, axis = 0)
    dWx = x.T @ dexploded_vec # (D, N) @ (N, 4H)
    dWh = prev_h.T @ dexploded_vec # (H, N) @ (N, 4H)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    caches = {}
    N, T, D = x.shape
    N, H = h0.shape
    h = np.zeros((N, T, H))
    curr_h = h0
    curr_c = np.zeros((N, H))
    for t in range(0, T) :
        x_t = x[:, t, :]
        next_h, next_c, caches[t] = lstm_step_forward(x_t, curr_h, curr_c, Wx, Wh, b)
        curr_h = next_h
        curr_c = next_c
        h[:, t, :] = curr_h
    cache = caches
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    N, T, H = dh.shape
    x1 = cache[0][0]
    N, D = x1.shape
    dx = np.zeros((N, T, D))
    dh0 = np.zeros((N, H))
    dWx = np.zeros((D, 4*H))
    dWh = np.zeros((H, 4*H))
    db = np.zeros((4*H))
    dh_curr = np.zeros((N, H))
    dc_curr = np.zeros((N, H))
    for t in range(T-1, -1, -1) :
        dh_curr += dh[:, t, :]
        dx_c, dprev_h, dprev_c, dWx_c, dWh_c, db_c = lstm_step_backward(dh_curr, dc_curr, cache[t])
        dh_curr = dprev_h
        dx[:, t, :] = dx_c
        dWx += dWx_c
        dWh += dWh_c
        db += db_c
        dc_curr = dprev_c
    dh0 = dh_curr
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx

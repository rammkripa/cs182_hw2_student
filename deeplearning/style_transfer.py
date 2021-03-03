import numpy as np

import torch
import torch.nn.functional as F


def content_loss(content_weight, content_current, content_target):
    """
    Compute the content loss for style transfer.

    Inputs:
    - content_weight: Scalar giving the weighting for the content loss.
    - content_current: features of the current image; this is a PyTorch Tensor of shape
      (1, C_l, H_l, W_l).
    - content_target: features of the content image, Tensor with shape (1, C_l, H_l, W_l).

    Returns:
    - scalar content loss
    """
    ##############################################################################
    #                               YOUR CODE HERE                               #
    ##############################################################################
    a, c, h, w = content_current.shape
    content_target_reshaped = content_target.reshape((c, h*w))
    content_current_reshaped = content_current.reshape((c, h*w))
    return content_weight * torch.sum((content_target_reshaped - content_current_reshaped)**2)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################


def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.

    Inputs:
    - features: PyTorch Variable of shape (N, C, H, W) giving features for
      a batch of N images.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)

    Returns:
    - gram: PyTorch Variable of shape (N, C, C) giving the
      (optionally normalized) Gram matrices for the N input images.
    """
    ##############################################################################
    #                               YOUR CODE HERE                               #
    ##############################################################################
    N, C, H, W = features.shape
    gram = torch.zeros((N, C, C))
    for i in range(0, N) :
        features_i = features[i, :, :, :]
        features_i_reshaped = features_i.reshape((C, H*W))
        gram_i = features_i_reshaped @ features_i_reshaped.T
        gram[i, :, :] = gram_i
    if normalize :
        gram = gram/(H*W*C)
    return gram
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################


def style_loss(feats, style_layers, style_targets, style_weights):
    """
    Computes the style loss at a set of layers.

    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a PyTorch Variable giving the Gram matrix the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].

    Returns:
    - style_loss: A PyTorch Variable holding a scalar giving the style loss.
    """
    # Hint: you can do this with one for loop over the style layers, and should
    # not be very much code (~5 lines). You will need to use your gram_matrix function.
    ##############################################################################
    #                               YOUR CODE HERE                               #
    ##############################################################################
    style_loss = 0
    for i in range(0, len(style_layers)) :
        feat_idx = style_layers[i]
        #index into feats
        feat_i = feats[feat_idx]
        feat_gram = gram_matrix(feat_i)
        #index into target
        target_gram = style_targets[i]
        # MSE loss
        mse_loss_i = torch.sum(((target_gram - feat_gram) ** 2))
        style_loss += style_weights[i] * mse_loss_i
    return style_loss

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################


def tv_loss(img, tv_weight):
    """
    Compute total variation loss.

    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.

    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    # Your implementation should be vectorized and not require any loops!
    ##############################################################################
    #                               YOUR CODE HERE                               #
    ##############################################################################
    loss = 0
    a, b, H, W = img.shape
    img_centered_y = img[:, :, 0:H-1, :]
    img_shift_y = img[:, :, 1:H, :]
    img_centered_x = img[:, :, :, 0:W-1]
    img_shift_x = img[:, :, :, 1:W]
    loss = torch.sum((img_centered_y - img_shift_y) ** 2) + torch.sum((img_centered_x - img_shift_x) ** 2)
    return tv_weight * loss
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

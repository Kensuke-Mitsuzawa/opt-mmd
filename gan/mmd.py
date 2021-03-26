'''
MMD functions implemented in tensorflow.
'''
from __future__ import division

import tensorflow as tf
import typing
from tf_ops import dot, sq_sum


_eps=1e-8

################################################################################
### Quadratic-time MMD with Gaussian RBF kernel


def _mix_rbf_kernel(X: tf.Tensor,
                    Y: tf.Tensor,
                    sigmas: typing.Sequence[float],
                    wts=None) -> typing.Tuple[typing.Optional[tf.Tensor], typing.Optional[tf.Tensor],
                                              typing.Optional[tf.Tensor], tf.Tensor]:
    """compute terms that is needed to calculate V_m(P, Q) of MMD. See Eq(5) in Sutherland, 2017.

     :return: (K_XX, K_XY, K_YY, tf.reduce_sum(wts))
    """
    if wts is None:
        wts = [1] * len(sigmas)

    # tf.matmul; dot-product between matrices
    XX = tf.matmul(X, X, transpose_b=True)
    XY = tf.matmul(X, Y, transpose_b=True)
    YY = tf.matmul(Y, Y, transpose_b=True)
    # tf.diag_part; extraction of diag-elements of matrices.
    X_sqnorms = tf.diag_part(XX)
    Y_sqnorms = tf.diag_part(YY)

    r = lambda x: tf.expand_dims(x, 0)
    c = lambda x: tf.expand_dims(x, 1)

    # K_XX, K_XY, K_YY = 0, 0, 0
    K_XX: tf.Tensor = tf.zeros(XX.shape)
    K_XY: tf.Tensor = tf.zeros(XY.shape)
    K_YY: tf.Tensor = tf.zeros(YY.shape)

    for sigma, wt in zip(sigmas, wts):
        gamma = 1 / (2 * sigma**2)
        K_XX += wt * tf.exp(-gamma * (-2 * XX + c(X_sqnorms) + r(X_sqnorms)))
        K_XY += wt * tf.exp(-gamma * (-2 * XY + c(X_sqnorms) + r(Y_sqnorms)))
        K_YY += wt * tf.exp(-gamma * (-2 * YY + c(Y_sqnorms) + r(Y_sqnorms)))

    return K_XX, K_XY, K_YY, tf.reduce_sum(wts)


def rbf_mmd2(X, Y, sigma=1, biased=True):
    return mix_rbf_mmd2(X, Y, sigmas=[sigma], biased=biased)


def mix_rbf_mmd2(X, Y, sigmas=(1,), wts=None, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigmas, wts)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)


def rbf_mmd2_and_ratio(X, Y, sigma=1, biased=True):
    return mix_rbf_mmd2_and_ratio(X, Y, sigmas=[sigma], biased=biased)


def mix_rbf_mmd2_and_ratio(X: tf.Tensor,
                           Y: tf.Tensor,
                           sigmas: typing.Sequence[int] =(1,),
                           wts=None,
                           biased=True):
    """

    :param X: A matrix that contains MNIST image.
    :param Y: A matrix that Tensorflow replaced as a place-holder.
    :param sigmas: Sigma parameters of Kernel.
    :param wts: unknown
    :param biased: unknown
    :return:
    """
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigmas, wts)
    return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)


################################################################################
### Helper functions to compute variances based on kernel matrices


def _mmd2(K_XX: tf.Tensor,
          K_XY: tf.Tensor,
          K_YY: tf.Tensor,
          const_diagonal=False,
          biased=False):
    m = tf.cast(K_XX.get_shape()[0], tf.float32)
    n = tf.cast(K_YY.get_shape()[0], tf.float32)

    if biased:
        mmd2 = (tf.reduce_sum(K_XX) / (m * m)
              + tf.reduce_sum(K_YY) / (n * n)
              - 2 * tf.reduce_sum(K_XY) / (m * n))
    else:
        if const_diagonal is not False:
            trace_X = m * const_diagonal
            trace_Y = n * const_diagonal
        else:
            trace_X = tf.trace(K_XX)
            trace_Y = tf.trace(K_YY)

        mmd2 = ((tf.reduce_sum(K_XX) - trace_X) / (m * (m - 1))
              + (tf.reduce_sum(K_YY) - trace_Y) / (n * (n - 1))
              - 2 * tf.reduce_sum(K_XY) / (m * n))

    return mmd2


def _mmd2_and_ratio(K_XX: tf.Tensor,
                    K_XY: tf.Tensor,
                    K_YY: tf.Tensor,
                    const_diagonal=False,
                    biased=False,
                    min_var_est=_eps) -> typing.Tuple[tf.Tensor, tf.Tensor]:
    """computes t_k(X, Y) := MMD2 / sqrt(V_m). The value is equivalent with test-power.

    :param K_XX:
    :param K_XY:
    :param K_YY:
    :param const_diagonal:
    :param biased:
    :param min_var_est:
    :return: (mmd2, t_k(X, Y)); mmd2 is a 1st-Tensor, t_k is a 1st_Tensor.
    """
    mmd2, var_est = _mmd2_and_variance(
        K_XX, K_XY, K_YY, const_diagonal=const_diagonal, biased=biased)
    ratio = mmd2 / tf.sqrt(tf.maximum(var_est, min_var_est))
    return mmd2, ratio


def _mmd2_and_variance(K_XX: tf.Tensor,
                       K_XY: tf.Tensor,
                       K_YY: tf.Tensor,
                       const_diagonal=False,
                       biased=False) -> typing.Tuple[tf.Tensor, tf.Tensor]:
    """computes mmd2 and V_m. For the mmd2, see Eq(2) in the Sutherland. For the V_m, see Eq(5).

    :param K_XX:
    :param K_XY:
    :param K_YY:
    :param const_diagonal:
    :param biased:
    :return: a tuple of (mmd2, V_m); mmd2 is a 1st-Tensor-object, V_m is a 1st_Tensor-object.
    """
    m = tf.cast(K_XX.get_shape()[0], tf.float32)  # Assumes X, Y are same shape

    ### Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        const_diagonal = tf.cast(const_diagonal, tf.float32)
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
        sum_diag2_X = sum_diag2_Y = m * const_diagonal**2
    else:
        diag_X = tf.diag_part(K_XX)
        diag_Y = tf.diag_part(K_YY)

        sum_diag_X = tf.reduce_sum(diag_X)
        sum_diag_Y = tf.reduce_sum(diag_Y)

        sum_diag2_X = sq_sum(diag_X)
        sum_diag2_Y = sq_sum(diag_Y)

    Kt_XX_sums = tf.reduce_sum(K_XX, 1) - diag_X
    Kt_YY_sums = tf.reduce_sum(K_YY, 1) - diag_Y
    K_XY_sums_0 = tf.reduce_sum(K_XY, 0)
    K_XY_sums_1 = tf.reduce_sum(K_XY, 1)

    Kt_XX_sum = tf.reduce_sum(Kt_XX_sums)
    Kt_YY_sum = tf.reduce_sum(Kt_YY_sums)
    K_XY_sum = tf.reduce_sum(K_XY_sums_0)

    Kt_XX_2_sum = sq_sum(K_XX) - sum_diag2_X
    Kt_YY_2_sum = sq_sum(K_YY) - sum_diag2_Y
    K_XY_2_sum  = sq_sum(K_XY)

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
              + (Kt_YY_sum + sum_diag_Y) / (m * m)
              - 2 * K_XY_sum / (m * m))
    else:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * (m-1))
              + (Kt_YY_sum + sum_diag_Y) / (m * (m-1))
              - 2 * K_XY_sum / (m * m))

    var_est = (
          2 / (m**2 * (m-1)**2) * (
              2 * sq_sum(Kt_XX_sums) - Kt_XX_2_sum
            + 2 * sq_sum(Kt_YY_sums) - Kt_YY_2_sum)
        - (4*m-6) / (m**3 * (m-1)**3) * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 4*(m-2) / (m**3 * (m-1)**2) * (
              sq_sum(K_XY_sums_1) + sq_sum(K_XY_sums_0))
        - 4 * (m-3) / (m**3 * (m-1)**2) * K_XY_2_sum
        - (8*m - 12) / (m**5 * (m-1)) * K_XY_sum**2
        + 8 / (m**3 * (m-1)) * (
              1/m * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
            - dot(Kt_XX_sums, K_XY_sums_1)
            - dot(Kt_YY_sums, K_XY_sums_0))
    )

    return mmd2, var_est

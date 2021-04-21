import numpy as np
import torch
import typing
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from logging import getLogger, StreamHandler, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ----------------------------------------------------------------------
# MMD equation


def _mmd2_and_variance(K_XX: torch.Tensor,
                       K_XY: torch.Tensor,
                       K_YY: torch.Tensor, unit_diagonal=False, biased=False):
    m = K_XX.shape[0]  # Assumes X, Y are same shape

    ### Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if unit_diagonal:
        diag_X = diag_Y = 1
        sum_diag_X = sum_diag_Y = m
        sum_diag2_X = sum_diag2_Y = m
    else:
        diag_X = torch.diagonal(K_XX)
        diag_Y = torch.diagonal(K_YY)

        sum_diag_X = diag_X.sum()
        sum_diag_Y = diag_Y.sum()

        sum_diag2_X = diag_X.dot(diag_X)
        sum_diag2_Y = diag_Y.dot(diag_Y)

    # Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_XX_sums = torch.sum(K_XX, dim=1) - diag_X
    # Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    Kt_YY_sums = torch.sum(K_YY, dim=1) - diag_Y
    # K_XY_sums_0 = K_XY.sum(axis=0)
    K_XY_sums_0 = torch.sum(K_XY, dim=0)
    # K_XY_sums_1 = K_XY.sum(axis=1)
    K_XY_sums_1 = torch.sum(K_XY, dim=1)

    # todo maybe, this must be replaced.
    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    # todo maybe, this must be replaced.
    # should figure out if that's faster or not on GPU / with theano...
    Kt_XX_2_sum = (K_XX ** 2).sum() - sum_diag2_X
    Kt_YY_2_sum = (K_YY ** 2).sum() - sum_diag2_Y
    K_XY_2_sum  = (K_XY ** 2).sum()

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
              + (Kt_YY_sum + sum_diag_Y) / (m * m)
              - 2 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m-1))
              + Kt_YY_sum / (m * (m-1))
              - 2 * K_XY_sum / (m * m))

    var_est = (
          2 / (m**2 * (m-1)**2) * (
              2 * Kt_XX_sums.dot(Kt_XX_sums) - Kt_XX_2_sum
            + 2 * Kt_YY_sums.dot(Kt_YY_sums) - Kt_YY_2_sum)
        - (4*m-6) / (m**3 * (m-1)**3) * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 4*(m-2) / (m**3 * (m-1)**2) * (
              K_XY_sums_1.dot(K_XY_sums_1)
            + K_XY_sums_0.dot(K_XY_sums_0))
        - 4 * (m-3) / (m**3 * (m-1)**2) * K_XY_2_sum
        - (8*m - 12) / (m**5 * (m-1)) * K_XY_sum**2
        + 8 / (m**3 * (m-1)) * (
              1/m * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
            - Kt_XX_sums.dot(K_XY_sums_1)
            - Kt_YY_sums.dot(K_XY_sums_0))
    )

    # todo return type
    return mmd2, var_est


def _mmd2_and_ratio(k_xx: torch.Tensor,
                    k_xy: torch.Tensor,
                    k_yy: torch.Tensor,
                    unit_diagonal=False,
                    biased=False,
                    min_var_est: float=1e-8):
    # todo what is return variable?
    mmd2, var_est = _mmd2_and_variance(k_xx, k_xy, k_yy, unit_diagonal=unit_diagonal, biased=biased)
    # ratio = mmd2 / torch.sqrt(T.largest(var_est, min_var_est))
    ratio = mmd2 / torch.sqrt(torch.max(var_est, min_var_est))
    return mmd2, ratio


def rbf_mmd2_and_ratio(x: torch.Tensor,
                       y: torch.Tensor,
                       sigma=0,
                       biased=True):
    # todo return type
    gamma = 1 / (2 * sigma**2)

    # XX = T.dot(X, X.T)
    # torch.t() is transpose function
    xx = torch.dot(x, torch.t(x))
    # XY = T.dot(X, Y.T)
    xy = torch.dot(x, torch.t(y))
    # YY = T.dot(Y, Y.T)
    yy = torch.dot(y, torch.t(y))

    # X_sqnorms = T.diagonal(XX)
    x_sqnorms = torch.diagonal(xx, offset=0)
    # Y_sqnorms = T.diagonal(YY)
    y_sqnorms = torch.diagonal(yy, offset=0)

    #K_XY = T.exp(-gamma * (-2 * XY + X_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))
    k_xy = torch.exp(-1 * gamma * (-2 * xy + x_sqnorms[:, np.newaxis] + y_sqnorms[np.newaxis, :]))
    # K_XX = T.exp(-gamma * (-2 * XX + X_sqnorms[:, np.newaxis] + X_sqnorms[np.newaxis, :]))
    k_xx = torch.exp(-1 * gamma * (-2 * xx + x_sqnorms[:, np.newaxis] + x_sqnorms[np.newaxis, :]))
    # K_YY = T.exp(-gamma * (-2 * YY + Y_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))
    k_yy = torch.exp(-1 * gamma * (-2 * yy + y_sqnorms[:, np.newaxis] + y_sqnorms[np.newaxis, :]))

    return _mmd2_and_ratio(k_xx, k_xy, k_yy, unit_diagonal=True, biased=biased)


# ----------------------------------------------------------------------


def run_train_epoch(X_train, Y_train, batchsize, train_fn) -> typing.Tuple[float, float]:
    total_mmd2 = 0
    total_obj = 0
    n_batches = 0
    batches = zip( # shuffle the two independently
        iterate_minibatches(X_train, batchsize=batchsize, shuffle=True),
        iterate_minibatches(Y_train, batchsize=batchsize, shuffle=True),
    )
    for ((Xbatch,), (Ybatch,)) in batches:
        mmd2, obj = train_fn(Xbatch, Ybatch)
        assert np.isfinite(mmd2)
        assert np.isfinite(obj)
        total_mmd2 += mmd2
        total_obj += obj
        n_batches += 1
    return total_mmd2 / n_batches, total_obj / n_batches


def run_val(X_val, Y_val, batchsize, val_fn) -> typing.Tuple[float, float]:
    total_mmd2 = 0
    total_obj = 0
    n_batches = 0
    for (Xbatch, Ybatch) in iterate_minibatches(
                X_val, Y_val, batchsize=batchsize):
        mmd2, obj = val_fn(Xbatch, Ybatch)
        assert np.isfinite(mmd2)
        assert np.isfinite(obj)
        total_mmd2 += mmd2
        total_obj += obj
        n_batches += 1
    # end for
    return total_mmd2 / n_batches, total_obj / n_batches

# ----------------------------------------------------------------------


def train(x: torch.Tensor, y: torch.Tensor):
    # todo optimization with SGD Nesterov momentum
    # todo torch.optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=False)
    # with Nesterov momentum
    assert len(x.size()) == len(y.size()) == 2
    logger.debug(f'input data N(sample-size)={x.size()[0]}, N(dimension)={x.size()[1]}')
    # global sigma value of RBF kernel
    log_sigma: torch.Tensor = torch.randn(size=[1, ], requires_grad=True)
    # a scale matrix which scales the input matrix X.
    scales: torch.Tensor = torch.randn(size=x.size()[1], requires_grad=True)
    # todo must convert x, y into rep_p, rep_q
    # todo net_p, net_q have coefficient filter. The initial value of net_q is the same as net_p's initial value.

    # todo mmd2_pq
    mmd2_pq, stat = rbf_mmd2_and_ratio(X=rep_p, Y=rep_q, **kwargs)
    # todo obj T is Theano here
    obj = -(T.log(T.largest(stat, 1e-6)) if opt_log else stat) + reg

    # todo must implement the corresponding equation.
    # train_fn = theano.function(inputs=[input_p, input_q], outputs=[mmd2_pq, obj], updates=updates)


def main():
    x_data = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    y_data = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    train(x_data, y_data)


if __name__ == '__main__':
    main()

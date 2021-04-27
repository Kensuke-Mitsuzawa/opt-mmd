import numpy as np
import torch
import typing
import nptyping
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

TypeInputData = typing.Union[torch.Tensor, nptyping.NDArray[(typing.Any, typing.Any), typing.Any]]
TypeScaleVector = nptyping.NDArray[(typing.Any, typing.Any), typing.Any]


class TwoSampleDataSet(torch.utils.data.Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y
        self.length = len(x)
        assert len(x) == len(y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.length


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
                    unit_diagonal: bool=False,
                    biased: bool=False,
                    min_var_est: torch.Tensor=torch.Tensor([1e-8])
                    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    # todo what is return variable?
    mmd2, var_est = _mmd2_and_variance(k_xx, k_xy, k_yy, unit_diagonal=unit_diagonal, biased=biased)
    # ratio = mmd2 / torch.sqrt(T.largest(var_est, min_var_est))
    ratio = mmd2 / torch.sqrt(torch.max(var_est, min_var_est))
    return mmd2, ratio


def rbf_mmd2_and_ratio(x: torch.Tensor, y: torch.Tensor, sigma=0, biased=True):
    # todo return type
    gamma = 1 / (2 * sigma**2)

    # XX = T.dot(X, X.T)
    # torch.t() is transpose function. torch.dot() is only for vectors. For 2nd tensors, "mm".
    xx = torch.mm(x, torch.t(x))
    # XY = T.dot(X, Y.T)
    xy = torch.mm(x, torch.t(y))
    # YY = T.dot(Y, Y.T)
    yy = torch.mm(y, torch.t(y))

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
# a procedure in an epoch

def iterate_minibatches(*arrays, batchsize: int, is_shuffle: bool=False):
    shuffle = is_shuffle

    assert len(arrays) > 0
    n = len(arrays[0])
    assert all(len(a) == n for a in arrays[1:])

    if shuffle:
        indices = np.arange(n)
        np.random.shuffle(indices)

    for start_idx in range(0, max(0, n - batchsize) + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield tuple(a[excerpt] for a in arrays)


def run_train_epoch(dataset: TwoSampleDataSet,
                    batchsize: int,
                    sigma: torch.Tensor,
                    scaler: torch.Tensor,
                    reg: int,
                    opt_log: bool = True) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    total_mmd2 = 0
    total_obj = 0
    n_batches = 0
    # batches = zip( # shuffle the two independently
    #     iterate_minibatches(x_train, batchsize=batchsize, is_shuffle=True),
    #     iterate_minibatches(y_train, batchsize=batchsize, is_shuffle=True),
    # )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=2)
    for xbatch, ybatch in data_loader:
        mmd2_pq, stat, obj = function_forward(xbatch, ybatch, sigma=sigma, scaler=scaler, reg=reg, opt_log=opt_log)
        #assert np.isfinite(mmd2)
        #assert np.isfinite(obj)
        total_mmd2 += mmd2_pq
        total_obj += obj
        n_batches += 1
        # do differentiation now.
        obj.backward()

    return total_mmd2 / n_batches, total_obj / n_batches


def run_validation_epoch(dataset: TwoSampleDataSet,
                         batchsize: int,
                         sigma: torch.Tensor,
                         scaler: torch.Tensor,
                         reg: int,
                         opt_log: bool = True) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    total_mmd2 = 0
    total_obj = 0
    n_batches = 0
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=2)
    #for (x_batch, y_batch) in iterate_minibatches(x_val, y_val, batchsize=batchsize):
    for x_batch, y_batch in data_loader:
        mmd2_pq, stat, obj = function_forward(x_batch, y_batch, sigma=sigma, scaler=scaler, reg=reg, opt_log=opt_log)
        #assert np.isfinite(mmd2)
        #assert np.isfinite(obj)
        total_mmd2 += mmd2_pq
        total_obj += obj
        n_batches += 1
    # end for
    avg_mmd = total_mmd2 / n_batches
    avg_obj = total_obj / n_batches
    return avg_mmd, avg_obj



# ----------------------------------------------------------------------


def operation_scale_product(scaler: torch.Tensor,
                            input_p: torch.Tensor,
                            input_q: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    rep_p = torch.mul(scaler, input_p)
    rep_q = torch.mul(scaler, input_q)

    return rep_p, rep_q


def function_forward(input_p: torch.Tensor,
                   input_q: torch.Tensor,
                   sigma: torch.Tensor,
                   scaler: torch.Tensor,
                   reg: float = 0,
                   opt_log: bool = True):
    """

    :param input_p: input-data
    :param input_q: input-data
    :return:
    """
    # 1. elementwise-product(data, scale-vector)
    rep_p, rep_q = operation_scale_product(scaler, input_p, input_q)

    # 2. exp(sigma)
    __sigma = torch.exp(sigma)

    # 3. compute MMD and ratio
    mmd2_pq, stat = rbf_mmd2_and_ratio(x=rep_p, y=rep_q, sigma=__sigma, biased=True)

    # 4. define the objective-value
    obj = -(torch.log(max(stat, 1e-6)) if opt_log else stat) + reg

    # # todo working to split the commands
    # mmd2_pq, obj, rep_p, net_p, net_q, log_sigma = self.make_network(
    #     input_p, input_q, dim,
    #     criterion=criterion, biased=biased, streaming_est=streaming_est,
    #     opt_log=opt_log, linear_kernel=linear_kernel, log_sigma=init_log_sigma,
    #     hotelling_reg=hotelling_reg, net_version=net_version)
    # Returns a list of Theano shared variables or expressions that parameterize the layer.
    # params: typing.List[theano.tensor.sharedvar.TensorSharedVariable] = \
    #     lasagne.layers.get_all_params([net_p, net_q], trainable=True)
    # if opt_sigma:
    #     params.append(log_sigma)
    # # end if

    # definition of gradient-search.
    # generate a function-object which can take arguments.
    # fn = getattr(lasagne.updates, strat)
    # # updates(return of lasagne.updates) is a dictionary-obj. The dict-obj is with keys:
    # updates: typing.Dict[theano.tensor.sharedvar.TensorSharedVariable, theano.tensor.var.TensorVariable] = \
    #     fn(obj, params, learning_rate=learning_rate, **opt_args)
    #
    # print("Compiling...", file=sys.stderr, end='')
    # # a function for training. updates,
    # # updates is key-value objects. The key a name of variable, the value is a way to update the variable.
    # train_fn = theano.function(
    #     inputs=[input_p, input_q], outputs=[mmd2_pq, obj], updates=updates)
    # val_fn = theano.function(inputs=[input_p, input_q], outputs=[mmd2_pq, obj])
    # get_rep = theano.function(inputs=[input_p], outputs=rep_p)
    # print("done", file=sys.stderr)

    return mmd2_pq, stat, obj


def __init_sigma_value(x_train: TypeInputData, y_train: TypeInputData, log_sigma, init_sigma_median):
    """"""
    # initialization of initial-sigma value
    if log_sigma is not None and init_sigma_median:
        print("Getting median initial sigma value...", end='')
        n_samp = min(500, x_train.shape[0], y_train.shape[0])
        samp = np.vstack([
            x_train[np.random.choice(x_train.shape[0], n_samp, replace=False)],
            y_train[np.random.choice(y_train.shape[0], n_samp, replace=False)],
        ])
        reps = np.vstack([
            get_rep(batch) for batch, in
            self.iterate_minibatches(samp, batchsize=val_batchsize)])
        D2 = euclidean_distances(reps, squared=True)
        med_sqdist = np.median(D2[np.triu_indices_from(D2, k=1)])
        log_sigma.set_value(make_floatX(np.log(med_sqdist / np.sqrt(2)) / 2))
        rep_dim = reps.shape[1]
        del samp, reps, D2, med_sqdist
        print("{:.3g}".format(np.exp(log_sigma.get_value())))
    else:
        rep_dim = get_rep(x_train[:1]).shape[1]
    # end if

    return rep_dim


def split_data(x: TypeInputData,
               y: TypeInputData,
               ratio_train: float = 0.8
               ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # data conversion
    if isinstance(x, np.ndarray):
        x__ = torch.Tensor(x)
    else:
        x__ = x
    # end if
    if isinstance(y, np.ndarray):
        y__ = torch.Tensor(y)
    else:
        y__ = y
    # end if
    if ratio_train < 1.0:
        __split_index = int(len(x) * ratio_train)
        x_train, x_val = torch.utils.data.random_split(x__, lengths=[__split_index, len(x__)-__split_index])
        y_train, y_val = torch.utils.data.random_split(y__, lengths=[__split_index, len(x__)-__split_index])
    else:
        x_train = x__
        y_train = y__
        x_val = None
        y_val = None
    # end if
    return x_train, y_train, x_val, y_val


def train(x_data: TypeInputData,
          y_data: TypeInputData,
          init_log_sigma: float = 0,
          init_sigma_median: bool = False,
          reg: int = 0,
          num_epochs: int = 1000,
          lr: float = 0.5,
          batchsize: int = 200,
          opt_log: bool = True,
          ratio_train: float = 0.8):
    # todo optimization with SGD Nesterov momentum
    # todo torch.optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=False)
    # with Nesterov momentum
    assert len(x_data.shape) == len(y_data.shape) == 2
    logger.debug(f'input data N(sample-size)={x_data.shape[0]}, N(dimension)={x_data.shape[1]}')
    x_train, y_train, x_val, y_val = split_data(x_data, y_data, ratio_train)

    # global sigma value of RBF kernel
    log_sigma: torch.Tensor = torch.rand(size=(1,), requires_grad=True)
    # a scale matrix which scales the input matrix X.
    # must be the same size as the input data.
    scales: torch.Tensor = torch.rand(size=(x_data.shape[1], ), requires_grad=True)
    # __init_sigma_value(x_train=x, y_train=y, log_sigma=log_sigma, init_sigma_median=init_sigma_median)

    #t_mmd2, t_obj = self.run_val(X_train, Y_train, batchsize, val_fn)
    #v_mmd2, v_obj = self.run_val(X_val, Y_val, val_batchsize, val_fn)
    #log(0, t_mmd2, t_obj, v_mmd2, v_obj, 0)
    #start_time = time.time()
    # todo nesterov -> True
    optimizer = torch.optim.SGD([scales, log_sigma], lr=lr, momentum=0, dampening=0, weight_decay=0, nesterov=False)
    # todo what is standard name of epoch??
    dataset_train = TwoSampleDataSet(x_train, y_train)
    dataset_validation = TwoSampleDataSet(x_val, y_val)
    for epoch in range(1, num_epochs + 1):
        # mmd2_pq, stat, obj = function_forward(input_p=x_data, input_q=y_data, sigma=log_sigma, scaler=scales, reg=0, opt_log=True)
        #obj.backward()
        #optimizer.step()
        avg_mmd2, avg_obj = run_train_epoch(dataset_train, batchsize=batchsize, sigma=log_sigma, scaler=scales, reg=reg, opt_log=opt_log)
        # update the variables
        optimizer.step()
        avg_mmd2_val, avg_obj_val = run_validation_epoch(dataset_validation, batchsize=batchsize, sigma=log_sigma, scaler=scales, reg=reg, opt_log=opt_log)
        logger.info(f'Validation obj-value={avg_obj_val}/MMD^2={avg_mmd2_val} with the current parameter sigma={log_sigma} and scaler')
        # for epoch in range(1, num_epochs + 1):
        #     try:
        #         t_mmd2, t_obj = self.run_train_epoch(
        #             X_train, Y_train, batchsize, train_fn)
        #         v_mmd2, v_obj = self.run_val(X_val, Y_val, val_batchsize, val_fn)
        #         log(epoch, t_mmd2, t_obj, v_mmd2, v_obj, time.time() - start_time)
        #     except KeyboardInterrupt:
        #         break
        #     # end try
        # # end for
        # sigma = np.exp(log_sigma.get_value()) if log_sigma is not None else None





# ---------------------------------------------------------------------------

def sample_SG(n: int, dim: int, rs=None) -> typing.Tuple[TypeInputData, TypeInputData]:
    from sklearn.utils import check_random_state
    rs = check_random_state(rs)
    mu = np.zeros(dim)
    sigma = np.eye(dim)
    X = rs.multivariate_normal(mu, sigma, size=n)
    Y = rs.multivariate_normal(mu, sigma, size=n)
    return X, Y


def generate_data(n_train: int, n_test: int):
    np.random.seed(np.random.randint(2 ** 31))
    # X, Y = generate.generate_data(args, n_train + n_test, dtype=floatX)
    # as an example X, Y are from the same distribution.
    X, Y = sample_SG(n_train + n_test, dim=2)
    is_train = np.zeros(n_train + n_test, dtype=bool)
    is_train[np.random.choice(n_train + n_test, n_train, replace=False)] = True
    X_train = X[is_train]
    Y_train = Y[is_train]
    X_test = X[~is_train]
    Y_test = Y[~is_train]

    return X_train, Y_train, X_test, Y_test


def main():
    n_train = 1500
    n_test = 500
    num_epochs = 100
    path_trained_model = './trained_mmd.pickle'

    np.random.seed(np.random.randint(2 ** 31))
    x_train, y_train, x_test, y_test = generate_data(n_train=n_train, n_test=n_test)
    train(x_train, y_train)


if __name__ == '__main__':
    main()

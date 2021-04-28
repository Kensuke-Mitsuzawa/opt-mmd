import numpy as np
import torch
import typing
import nptyping
import dataclasses
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics.pairwise import euclidean_distances
import pickle
import os
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

OBJ_VALUE_MIN_THRESHOLD = torch.tensor([1e-6], dtype=torch.float64)


@dataclasses.dataclass
class TrainingLog(object):
    epoch: int
    avg_mmd_training: float
    avg_obj_train: float
    mmd_validation: float
    obj_validation: float
    sigma: float
    scales: nptyping.NDArray[(typing.Any,), typing.Any]


@dataclasses.dataclass
class TrainedArdKernel(object):
    scales: nptyping.NDArray[(typing.Any, typing.Any), typing.Any]
    training_log: typing.List[TrainingLog]
    sigma: typing.Optional[np.float] = None
    x_train: typing.Optional[nptyping.NDArray[(typing.Any, typing.Any), typing.Any]] = None
    y_train: typing.Optional[nptyping.NDArray[(typing.Any, typing.Any), typing.Any]] = None

    def to_npz(self, path_npz: str):
        assert os.path.exists(os.path.dirname(path_npz))
        dict_obj = dataclasses.asdict(self)
        del dict_obj['mapping_network']
        np.savez(path_npz, **dict_obj)
        logger.info(f'saved as {path_npz}')

    def to_pickle(self, path_pickle: str):
        assert os.path.exists(os.path.dirname(path_pickle))
        f = open(path_pickle, 'wb')
        pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()


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


class ScaleLayer(torch.nn.Module):
    def __init__(self, init_value: TypeInputData, requires_grad: bool = True):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(init_value), requires_grad=requires_grad)

    def forward(self, input):
        return input * self.scale


# ----------------------------------------------------------------------
# MMD equation


def _mmd2_and_variance(K_XX: torch.Tensor,
                       K_XY: torch.Tensor,
                       K_YY: torch.Tensor, unit_diagonal=False, biased=False):
    m = K_XX.shape[0]  # Assumes X, Y are same shape

    # Get the various sums of kernels that we'll use
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
                    unit_diagonal: bool = False,
                    biased: bool = False,
                    min_var_est: torch.Tensor=torch.tensor([1e-8], dtype=torch.float64)
                    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    # todo what is return variable?
    mmd2, var_est = _mmd2_and_variance(k_xx, k_xy, k_yy, unit_diagonal=unit_diagonal, biased=biased)
    ratio = torch.div(mmd2, torch.sqrt(torch.max(var_est, min_var_est)))
    return mmd2, ratio


def rbf_mmd2_and_ratio(x: torch.Tensor,
                       y: torch.Tensor,
                       sigma: torch.Tensor,
                       biased=True):
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


def run_train_epoch(optimizer: torch.optim.SGD,
                    dataset: TwoSampleDataSet,
                    batchsize: int,
                    sigma: torch.Tensor,
                    scales: torch.Tensor,
                    reg: int,
                    opt_log: bool = True) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    total_mmd2 = 0
    total_obj = 0
    n_batches = 0

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=2)
    for xbatch, ybatch in data_loader:
        mmd2_pq, stat, obj = function_forward(xbatch, ybatch, sigma=sigma, scaler=scales, reg=reg, opt_log=opt_log)
        assert np.isfinite(mmd2_pq.detach().numpy())
        assert np.isfinite(obj.detach().numpy())
        total_mmd2 += mmd2_pq
        total_obj += obj
        n_batches += 1
        # do differentiation now.
        obj.backward()
        #
        optimizer.step()
        optimizer.zero_grad()

    # logger.debug(f'[after one epoch] sum(MMD)={total_mmd2}, sum(obj)={total_obj} with N(batch)={n_batches}')
    avg_mmd2 = torch.div(total_mmd2, n_batches)
    avg_obj = torch.div(total_obj, n_batches)
    return avg_mmd2, avg_obj


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
    for x_batch, y_batch in data_loader:
        mmd2_pq, stat, obj = function_forward(x_batch, y_batch, sigma=sigma, scaler=scaler, reg=reg, opt_log=opt_log)
        assert np.isfinite(mmd2_pq.detach().numpy())
        assert np.isfinite(obj.detach().numpy())
        total_mmd2 += mmd2_pq
        total_obj += obj
        n_batches += 1
    # end for
    avg_mmd = torch.div(total_mmd2, n_batches)
    avg_obj = torch.div(total_obj, n_batches)
    return avg_mmd, avg_obj



# ----------------------------------------------------------------------



def operation_scale_product(scaler: torch.Tensor,
                            input_p: torch.Tensor,
                            input_q: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    rep_p = torch.  mul(scaler, input_p)
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
    obj = -(torch.log(torch.max(stat, OBJ_VALUE_MIN_THRESHOLD)) if opt_log else stat) + reg

    return mmd2_pq, stat, obj


def __init_sigma_value(x_train: TypeInputData,
                       y_train: TypeInputData,
                       scales: torch.Tensor,
                       batchsize: int = 1000) -> torch.Tensor:
    """"""
    # initialization of initial-sigma value
    logger.info("Getting median initial sigma value...")
    n_samp = min(500, x_train.shape[0], y_train.shape[0])

    samp = torch.cat([
        x_train[np.random.choice(x_train.shape[0], n_samp, replace=False)],
        y_train[np.random.choice(y_train.shape[0], n_samp, replace=False)],
    ])

    data_loader = torch.utils.data.DataLoader(samp, batch_size=batchsize, shuffle=False)
    reps = torch.cat([torch.mul(batch, scales) for batch in data_loader])
    np_reps = reps.detach().numpy()
    d2 = euclidean_distances(np_reps, squared=True)
    med_sqdist = np.median(d2[np.triu_indices_from(d2, k=1)])
    __init_log_simga = np.log(med_sqdist / np.sqrt(2)) / 2
    del samp, reps, d2, med_sqdist
    logger.info("initial sigma by median-heuristics {:.3g}".format(np.exp(__init_log_simga)))

    return torch.tensor(np.array([__init_log_simga]), requires_grad=True)


def split_data(x: TypeInputData,
               y: TypeInputData,
               x_val: typing.Optional[TypeInputData],
               y_val: typing.Optional[TypeInputData],
               ratio_train: float = 0.8
               ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # data conversion
    if isinstance(x, np.ndarray):
        x__ = torch.tensor(x)
    else:
        x__ = x
    # end if
    if isinstance(y, np.ndarray):
        y__ = torch.tensor(y)
    else:
        y__ = y
    # end if
    if ratio_train < 1.0:
        __split_index = int(len(x) * ratio_train)
        x_train__ = x__[:__split_index]
        x_val__ = x__[__split_index:]
        y_train__ = y__[:__split_index]
        y_val__ = y__[__split_index]
    else:
        x_train__ = x__
        y_train__ = y__
        x_val__ = torch.tensor(x_val) if isinstance(x_val, torch.Tensor) is False else x_val
        y_val__ = torch.tensor(y_val) if isinstance(y_val, torch.Tensor) is False else y_val
    # end if
    return x_train__, y_train__, x_val__, y_val__


def __exp_sigma(sigma: torch.Tensor) -> torch.Tensor:
    __sigma = torch.exp(sigma)
    return __sigma


def train(x_train: TypeInputData,
          y_train: TypeInputData,
          init_sigma_median: bool = False,
          num_epochs: int = 1000,
          lr: float = 0.01,
          batchsize: int = 200,
          opt_log: bool = True,
          opt_sigma: bool = True,
          ratio_train: float = 0.8,
          init_log_sigma: float = 0,
          reg: int = 0,
          init_scale: np.ndarray = None,
          x_val: TypeInputData = None,
          y_val: TypeInputData = None) -> TrainedArdKernel:
    assert len(x_train.shape) == len(y_train.shape) == 2
    logger.debug(f'input data N(sample-size)={x_train.shape[0]}, N(dimension)={x_train.shape[1]}')

    if x_val is None or y_val is None:
        x_train__, y_train__, x_val__, y_val__ = split_data(x_train, y_train, None, None, ratio_train)
    else:
        x_train__, y_train__, x_val__, y_val__ = split_data(x_train, y_train, x_val, y_val, 1.0)
    # end if

    # a scale matrix which scales the input matrix X.
    # must be the same size as the input data.
    if init_scale is None and opt_sigma is False:
        scales: torch.Tensor = torch.rand(size=(x_train.shape[1], ), requires_grad=True)
    else:
        logger.info('Set the initial scales value')
        assert x_train.shape[1] == y_train.shape[1] == init_scale.shape[0]
        scales = torch.tensor(init_scale, requires_grad=True)
    # end if
    # global sigma value of RBF kernel
    if init_sigma_median:
        log_sigma = __init_sigma_value(x_train=x_train__, y_train=y_train__, scales=scales)
    elif init_log_sigma is not None:
        log_sigma: torch.Tensor = torch.tensor([init_log_sigma], requires_grad=True)
    else:
        log_sigma: torch.Tensor = torch.rand(size=(1,), requires_grad=True)
    # end if

    dataset_train = TwoSampleDataSet(x_train__, y_train__)
    val_mmd2_pq, val_stat, val_obj = function_forward(x_val__, y_val__, sigma=log_sigma, scaler=scales, reg=reg, opt_log=opt_log)
    logger.debug(
        f'Validation at 0. MMD^2 = {val_mmd2_pq.detach().numpy()}, obj-value = {val_obj.detach().numpy()} at sigma = {__exp_sigma(log_sigma).detach().numpy()}')
    logger.debug(f'[before optimization] sigma value = {__exp_sigma(log_sigma).detach().numpy()}')

    if opt_log:
        optimizer = torch.optim.SGD([scales, log_sigma], lr=lr, momentum=0.9, nesterov=True)
    else:
        optimizer = torch.optim.SGD([scales], lr=lr, momentum=0.9, nesterov=True)
    # for the logging
    fmt = ("{: >6,}: avg train MMD^2 {} obj {},  "
           "avg val MMD^2 {}  obj {}  elapsed: {:,} sigma: {} scales: {}")
    training_log = []
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        avg_mmd2, avg_obj = run_train_epoch(optimizer, dataset_train, batchsize=batchsize, sigma=log_sigma, scales=scales, reg=reg, opt_log=opt_log)
        val_mmd2_pq, val_stat, val_obj = function_forward(x_val__, y_val__, sigma=log_sigma, scaler=scales, reg=reg, opt_log=opt_log)
        training_log.append(TrainingLog(epoch,
                                        avg_mmd2.detach().numpy(),
                                        avg_obj.detach().numpy(),
                                        val_mmd2_pq.detach().numpy(),
                                        val_obj.detach().numpy(),
                                        sigma=log_sigma.detach().numpy(), scales=scales.detach().numpy()))
        if epoch in {0, 5, 25, 50} or epoch % 100 == 0:
            logger.info(
                fmt.format(epoch, avg_mmd2.detach().numpy(), avg_obj.detach().numpy(), val_mmd2_pq.detach().numpy(),
                           val_obj.detach().numpy(), 0.0, __exp_sigma(log_sigma).detach().numpy(),
                           scales.detach().numpy()))
        # end if
    # end for

    return TrainedArdKernel(
        scales=scales.detach().numpy(),
        sigma=log_sigma.detach().numpy()[0],
        training_log=training_log)


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


def devel_test():
    num_epochs = 100
    path_trained_model = './trained_mmd.pickle'

    np.random.seed(np.random.randint(2**31))
    array_obj = np.load('./interfaces/eval_array.npz')
    x_train = array_obj['x']
    y_train = array_obj['y']
    x_test = array_obj['x_test']
    y_test = array_obj['y_test']
    init_scale = np.array([0.05, 0.55])

    trained_obj = train(x_train,
                        y_train,
                        num_epochs=num_epochs,
                        init_log_sigma=0.0,
                        init_scale=init_scale,
                        x_val=x_test, y_val=y_test,
                        opt_log=True,
                        opt_sigma=True,
                        init_sigma_median=False)
    trained_obj.to_pickle(path_trained_model)
    import math
    logger.info(f'exp(sigma)={math.exp(trained_obj.sigma)} scales={trained_obj.scales}')

    assert np.linalg.norm(trained_obj.scales[0] - trained_obj.scales[1]) < 5
    assert 0.0 < math.exp(trained_obj.sigma) < 1.5
    os.remove(path_trained_model)


def main():
    n_train = 1500
    n_test = 500
    num_epochs = 100
    path_trained_model = './trained_mmd.pickle'

    np.random.seed(np.random.randint(2**31))
    #x_train, y_train, x_test, y_test = generate_data(n_train=n_train, n_test=n_test)
    array_obj = np.load('./interfaces/eval_array.npz')
    x_train = array_obj['x']
    y_train = array_obj['y']
    x_test = array_obj['x_test']
    y_test = array_obj['y_test']
    init_scale = np.array([0.05, 0.55])

    train(x_train,
          y_train,
          num_epochs=num_epochs,
          init_log_sigma=0.0,
          init_scale=init_scale,
          x_val=x_test, y_val=y_test,
          opt_log=True)


if __name__ == '__main__':
    devel_test()

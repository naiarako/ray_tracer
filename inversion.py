"""
:author:
    Naiara Korta Martiartu (naiara.korta@unibe.ch)

    University of Bern
    2021 - 2024

:license:
    GPL-3.0
"""



import numpy as np
import timeit
from scipy.linalg import norm
import matplotlib.pyplot as plt

from regularization import regularization

try:
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as csr_gpu
except ModuleNotFoundError:
    print('Cupy python package not found. Disable GPU options.')
    pass


def inversion(data, forward_op, nx, nz, c0, reg_parameter=1e-3, order=0, gpu=False):

    """
    Function to solve the regularized linear inverse problem.
    :param data: observed propagation times [numpy array, nx*nz x number of transducer elements]
    :param forward_op: a list of sparse submatrices of the forward operator. The length of the list is equal to the
    number of transducer elements. For each element, the submatrix is composed of the weights of line
                                  integrals (ray paths)  [compressed Sparse Row matrix, nx*nz x nx*nz]
    :param nx: number of grid point along x-direction [integer]
    :param nz: number of grid point along z-direction [integer]
    :param c0: assumed speed of sound [scalar]
    :param reg_parameter: regularization parameter [positive scalar]
    :param order: order of Tikhonov regularization [integer, supported: 0, 1, or 2]
    :param gpu: use gpu? [boolean]
    :return delta_rec: reconstructed slowness deviation [numpy array, nz*nx x 1]
            sos: reconstructed speed of sound [numpy array, nz*nx x 1]
    """

    start = timeit.default_timer()  # start counting run time

    print(50 * '-')
    print('Start inversion...')

    npoints = forward_op[0].shape[0]  # number of grid points (number of model parameters)
    nelem = len(forward_op)

    if gpu:  # in GPU

        dobs_gpu = cp.array(data)

        hessian = csr_gpu((npoints, npoints))  # initialize hessian operator and gradient
        grad = cp.zeros((npoints,))

        for iel in range(nelem):  # loop over all transducer elements

            # submatrix of the forward operator for transmit element iel
            A = csr_gpu((forward_op[iel]))

            # save gradients for later
            grad += A.transpose().dot(dobs_gpu[:, iel])

            # Hessian operator: compute hessian for submatrices and add
            hessian += A.transpose() * A

        # compute posterior covariance operator:

        post = regularization(order=order, reg_parameter=reg_parameter, nx=nx,
                              nz=nz, hessian=hessian, gpu=gpu)

        # Reconstruct model parameters:
        delta_rec = cp.dot(post, grad)

        # reconstructed speed of sound:
        sos = c0 / (c0 * delta_rec + 1)

    else:  # in CPU

        hessian = np.zeros((npoints, npoints))  # initialize hessian operator and gradient.
        grad = np.zeros((npoints,))

        for iel in range(nelem):  # loop over all transducer elements

            # submatrix of the forward operator for transmit element iel
            A = forward_op[iel]

            # save gradients for later
            grad += A.transpose().dot(data[:, iel])

            # Hessian operator: compute hessian for submatrices and add
            hessian += A.transpose() * A

        # compute posterior covariance operator:

        post = regularization(order=order, reg_parameter=reg_parameter, nx=nx,
                              nz=nz, hessian=hessian, gpu=gpu)

        # Reconstruct model parameters:
        delta_rec = np.dot(post, grad)

        # reconstructed speed of sound:
        sos = c0 / (c0 * delta_rec + 1)

    end = timeit.default_timer()  # end counting run time

    print('...done!')
    print(f'Run time: {(end - start):.2f} sec')
    print(50 * '-')

    misfit_reduction(delta_rec, data, forward_op)

    return delta_rec, sos


def plot_lcurve(data, forward_op, nx, nz, c0, order=0, gpu=False, reg_params=None,
                reg_anis_ratio=1.0, savefig=True, filename='L-curve_tikhonov_order'):
    """
    Function to plot the L-curve, useful to optimize the regularization parameter.
    :param data: observed propagation times [numpy array, nx*nz x number of transducer elements]
    :param forward_op: a list of sparse submatrices of the forward operator. The length of the list is equal to the
    number of transducer elements. For each element, the submatrix is composed of the weights of line
                                  integrals (ray paths)  [compressed Sparse Row matrix, nx*nz x nx*nz]
    :param order: order of Tikhonov regularization [integer, supported: 0, 1, or 2]
    :param gpu: use gpu? [boolean]
    :param reg_params: list with regularization parameter values. If None, take default ones. [list floats]
    :param reg_anis_ratio:  Ratio between reg. parameter values in x- and z-direction. The value in z is
                                multiplied by this value [float]
    :param savefig: if True, save the plot with L-curve [boolean]
    :param filename: if savefig==True, filename in which the plot will be saved [string]
    """

    start = timeit.default_timer()  # start counting run time

    npoints = forward_op[0].shape[0]  # number of grid points (number of model parameters)
    nelem = len(forward_op)

    # selected regularization parameters
    if reg_params is None:
        reg_params = [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

    # initialize vectors with all misfits and regularization terms
    misfit = np.zeros((len(reg_params),))
    reg = np.zeros_like(misfit)
    reg2 = np.zeros_like(misfit)

    print(50 * '-')
    print(f'Starting {len(misfit)} inversions to compute L-curve...')

    for i in range(len(misfit)):  # loop over different regularization parameters

        # run inversion for the given reg_parameter
        if order == 0:
            reg_parameter = reg_params[i]
        else:
            reg_parameter = [reg_params[i], reg_anis_ratio * reg_params[i]]

        delta_rec, _ = inversion(data, forward_op, nx, nz, c0, reg_parameter=reg_parameter, order=order, gpu=gpu)

        residual = np.zeros((nelem * npoints,))  # initialize vector of residuals

        for iel in range(nelem):  # loop over pair of angles

            # submatrix of the forward operator for transmit element iel
            A = forward_op[iel]

            # compute residuals for the specific angle
            residual[iel * npoints:(iel + 1) * npoints] = data[:, iel] - A.dot(delta_rec)

        misfit[i] = norm(residual)  # store corresponding misfit term

        # compute regularization term

        if order == 0:
            reg[i] = regularization(order=order, reg_parameter=reg_parameter, nx=nx,
                                    nz=nz, slowness=delta_rec)
        elif order < 3:
            reg[i], reg2[i] = regularization(order=order, reg_parameter=reg_parameter,
                                             nx=nx, nz=nz, slowness=delta_rec)

    # plot L-curve (loglog scale) and save it if required
    plt.loglog(misfit, reg + reg2, '*-', color='k')
    plt.xlabel('Misfit')
    plt.ylabel('Regularization')
    plt.title('L-curve')

    if savefig:  # save the image if required

        plt.savefig(filename + f'{order}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(filename + f'{order}.png', dpi=300, bbox_inches='tight', facecolor='white')

    end = timeit.default_timer()  # end counting run time

    print('...done L-curve computation!')
    print(f'Total run time: {(end - start) / 60:.2f} min')
    print(50 * '-')


def misfit_reduction(delta_rec, data, forward_op):
    """
    Function to compute misfit reduction achieved by the solution of the inverse problem.
    :param delta_rec: reconstructed slowness deviation, i.e., solution of the inverse problem [numpy array, nx*nz x 1]
    :param data: observed propagation times [numpy array, nx*nz x number of transducer elements]
    :param forward_op: a list of sparse submatrices of the forward operator. The length of the list is equal to the
    number of transducer elements. For each element, the submatrix is composed of the weights of line
                                  integrals (ray paths)  [compressed Sparse Row matrix, nx*nz x nx*nz]
    """

    npoints = len(delta_rec)  # number of model parameters
    nelem = len(forward_op)

    residual = np.zeros((nelem * npoints,))  # initialize useful vectors

    for iel in range(nelem):  # loop over pair of angles

        # submatrix of the forward operator for transmit element iel
        A = forward_op[iel]

        # residual between observed data and predicted data
        residual[iel * npoints:(iel + 1) * npoints] = data[:, iel] - A.dot(delta_rec)

    initial_misfit = norm(data.reshape(-1))  # initial misfit (equal to norm of observed data)
    final_misfit = norm(residual)  # final misfit

    # Print misfit reduction in %:
    print(f'Misfit reduction: \
        {(initial_misfit - final_misfit) / initial_misfit * 100:.2f} %')

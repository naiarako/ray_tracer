"""
:author:
    Naiara Korta Martiartu (naiara.korta@unibe.ch)

    University of Bern
    2021 - 2024

:license:
    GPL-3.0
"""

import numpy as np
from scipy.sparse import diags
from scipy.linalg import inv, norm

try:
    import cupyx
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as csr_gpu
except ModuleNotFoundError:
    pass


def regularization(order, nx, nz, reg_parameter=0, hessian=None, gpu=False, slowness=None):
    """
    Function to compute the posterior covariance matrix or the regularization term(s)
    :param order: order of Tikhonov regularization. [int]
    :param nx: number of parameters in x-direction [int]
    :param nz: number of parameters in z-direction [int]
    :param reg_parameter: regularization parameter [float or list of 2 floats (x- and z-direction)]
    :param hessian: Hessian operator [GPU: sparse csr_gpu array, CPU: numpy array: nx*nz x nx*nz]
    :param gpu: use gpu? [boolean]
    :param slowness: slowness of the medium, solution of the inverse problem [numpy array nx*nz x 1]
    :return: post: posterior covariance matrix [GPU: cupy array, CPU: numpy array: nx*nz x nx*nz]
             reg: value of the regularization term [for order > 0, reg in x]
             reg2: if order > 0, this is the regularization term in z
    """

    npoints = nx * nz  # number of model parameters

    if order == 0:  # zero-order Tikhonov regularization

        if slowness is None:
            assert (isinstance(reg_parameter, float)), "Regularization parameter must be a scalar."

        if slowness is not None:  # compute regularization term

            reg = norm(slowness)  # value regularization term

        else:  # compute posterior

            if gpu:  # in GPU

                R = cupyx.scipy.sparse.eye(hessian.shape[0])  # regularization matrix
                H = (hessian + reg_parameter * R).toarray()  # regularized hessian operator

                post = cp.linalg.inv(H)  # posterior covariance matrix

            else:  # in CPU

                H = hessian + reg_parameter * np.eye(hessian.shape[0])  # regularized Hessian operator

                post = inv(H)  # posterior covariance matrix

    elif order == 1:  # first-order Tikhonov regularization (Dirichlet boundary conditions)

        # verify that regularization parameter is a list
        if slowness is None:
            assert (isinstance(reg_parameter, list)), "Regularization parameter must be " \
                                                      "a list of length 2, " \
                                                      "the regularization in x- and z-direction."

        if gpu:  # in GPU

            diag = np.ones((npoints,))  # diagonal elements of finite-difference operator

            diag[np.arange(1, nz + 1) * nx - 1] = 0  # 0 when the pixel is at the boundary of domain

            fd_op_x = cupyx.scipy.sparse.diags([diag, -diag], [0, 1],
                                                    shape=(npoints, npoints))  # np.ones_like(diag)

            diag = np.ones((npoints,))  # diagonal elements of finite-difference operator

            diag[-nx:] = 0  # 0 when the pixel is at the boundary of domain

            # finite-difference operator in z-direction (sparse matrix)
            fd_op_z = cupyx.scipy.sparse.diags([diag, -diag], [0, nx], shape=(npoints, npoints))

            if slowness is not None:  # convert reg operators to numpy array for later
                fd_op_x = cp.asnumpy(fd_op_x)
                fd_op_z = cp.asnumpy(fd_op_z)

            else:  # compute posterior

                H = (hessian + reg_parameter[0] * (fd_op_x.transpose() * fd_op_x) +
                     reg_parameter[1] * (fd_op_z.transpose() * fd_op_z)).toarray()  # regularized hessian operator
                
                post = cp.linalg.inv(H)  # posterior covariance matrix

        else:  # in CPU

            diag = np.ones((npoints,))  # diagonal elements of finite-difference operator

            diag[np.arange(1, nz + 1) * nx - 1] = 0  # 0 when the pixel is at the boundary of domain

            # finite-difference operator in x-direction (sparse matrix)
            fd_op_x = diags([diag, -diag], [0, 1])

            diag = np.ones((npoints,))  # diagonal elements of finite-difference operator

            diag[-nx:] = 0  # 0 when the pixel is at the boundary of domain

            # finite-difference operator in z-direction (sparse matrix)
            fd_op_z = diags([diag, -diag], [0, nx])

            if slowness is None:  # compute posterior

                post = inv(hessian + reg_parameter[0] * (fd_op_x.transpose() * fd_op_x)
                           + reg_parameter[1] * (fd_op_z.transpose() * fd_op_z))  # posterior covariance matrix

        if slowness is not None:  # compute regularization term

            reg = norm(fd_op_x.toarray() * slowness)   # value of regularization term in x
            reg2 = norm(fd_op_z.toarray() * slowness)  # value of regularization term in z

    elif order == 2:  # econd-order Tikhonov regularization (Dirichlet boundary conditions)

        # verify that regularization parameter is a list
        if slowness is None:
            assert (isinstance(reg_parameter, list)), "Regularization parameter must be " \
                                                      "an list of length 2, " \
                                                      "the regularization in x- and z-direction."

        if gpu:  # in GPU

            diag = np.ones((npoints,))  # diagonal elements of finite-difference operator
            diag1 = np.ones((npoints,))

            diag[np.arange(1, nz + 1) * nx - 1] = 0  # 0 when the pixel is at the boundary of domain
            diag[np.arange(0, nz) * nx] = 0

            diag1[:-1] = diag[1:]  # 0 when the pixel is at the boundary of domain
            diag1[-1] = 0

            # finite-difference operator in x-direction (sparse matrix)
            fd2_op_x = cupyx.scipy.sparse.diags([2 * diag, -diag, -diag1], [0, 1, -1],
                                                     shape=(npoints, npoints))

            diag = np.ones((npoints,))  # diagonal elements of finite-difference operator
            diag1 = np.ones((npoints,))

            diag[:nx] = 0  # 0 when the pixel is at the boundary of domain
            diag[-nx:] = 0
            diag1[-2 * nx:] = 0

            # finite-difference operator in z-direction (sparse matrix)
            fd2_op_z = cupyx.scipy.sparse.diags([2 * diag, -diag, -diag1], [0, nx, -nx],
                                                     shape=(npoints, npoints))

            if slowness is not None:  # convert reg operators to numpy array for later
                fd2_op_x = cp.asnumpy(fd2_op_x)
                fd2_op_z = cp.asnumpy(fd2_op_z)

            else:  # compute posterior

                H = (hessian + reg_parameter[0] * (fd2_op_x.transpose() * fd2_op_x)
                     + reg_parameter[1] * (fd2_op_z.transpose() * fd2_op_z)).toarray()  # regularized hessian operator

                post = cp.linalg.inv(H)  # posterior covariance matrix

        else:  # in CPU

            diag = np.ones((npoints,))  # diagonal elements of finite-difference operator
            diag1 = np.ones((npoints,))

            diag[np.arange(1, nz + 1) * nx - 1] = 0  # 0 when the pixel is at the boundary of domain
            diag[np.arange(0, nz) * nx] = 0

            diag1[:-1] = diag[1:]  # 0 when the pixel is at the boundary of domain
            diag1[-1] = 0

            # finite-difference operator in x-direction (sparse matrix)
            fd2_op_x = diags([2 * diag, -diag, -diag1], [0, 1, -1])

            diag = np.ones((npoints,))  # diagonal elements of finite-difference operator
            diag1 = np.ones((npoints,))

            diag[:nx] = 0  # 0 when the pixel is at the boundary of domain
            diag[-nx:] = 0
            diag1[-2 * nx:] = 0

            # finite-difference operator in z-direction (sparse matrix)
            fd2_op_z = diags([2 * diag, -diag, -diag1], [0, nx, -nx])

            # compute posterior
            if slowness is None:
                post = inv(hessian + reg_parameter[0] * (fd2_op_x.transpose() * fd2_op_x)
                           + reg_parameter[1] * (fd2_op_z.transpose() * fd2_op_z))  # posterior covariance matrix

        if slowness is not None:  # compute regularization term

            reg = norm(fd2_op_x.toarray() * slowness)  # value of regularization term in x
            reg2 = norm(fd2_op_z.toarray() * slowness)  # value of regularization term in z

    # return posterior if required, else return regularization terms
    if slowness is None:
        return post
    else:
        if np.isscalar(reg_parameter):  # if order == 0
            return reg
        else:
            return reg, reg2




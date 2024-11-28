# Utilities for calculating the SigMMD metric between sets of samples.

# Computation
import numpy as np
import torch
import signatory  # Signature
from scipy.optimize import fsolve  # Tensor normalization
import logging
import sklearn.metrics.pairwise as pairwise


def psi(x):
    """Transformation function for tensor normalization, ref. Ex. 4 in Chevyrev and Oberhauser (2022)"""
    x = x**2
    C = 4.0  # C>1 required
    a = 1.0  # a>0 required
    if x <= C:
        return x
    else:
        return C + C / a - np.power(C, 1 + a) * np.power(x, -a)


def test_psi():
    assert np.isclose(psi(2), 4)
    assert np.isclose(psi(1), 1)
    print("Test psi passed")
    return


def tensor_normalization(t):
    """Naive implementation of tensor normalization, w/o vectorization, see Def. 12 and Prop. 14 in Chevyrev and Oberhauser (2022)"""
    # Compute lambda(t) first for Lambda(t) = delta_lambda(t) t
    target = psi(np.linalg.norm(t, ord=2))

    def f_getter(t_inner):
        def f(x):
            res = 0
            for i, t_i in enumerate(t_inner):
                res += np.power(x, i) * t_i
            return res - target

        return f

    f = f_getter(t)
    # Solve for lambda(t)
    lambda_t = fsolve(f, 1.0)[0]
    # Compute tensor normalization of t, that is the dilation map of t
    res_arr = []
    for i, t_i in enumerate(t):
        res_arr.append(lambda_t**i * t_i)
    return np.array(res_arr)


def sig_mmd(sig1, sig2, kernel, normalization=True, verbose=False):
    """
    Kernelized Sig-MMD naive or with tensor normalization for robustness.
    See 1.1 "A MMD for laws of stoch. processes" in Chevyrev and Oberhauser (2022) for SigMMD.
    See Prop. 18 in Chevyrev and Oberhauser (2022) for Robust Signature.
    See Theorem 21 and Cor. 23 further.
    """
    # require x,y shape (n_x, c) and (n_y, c), outputs (n_x, n_y)
    if kernel == "linear":  # k(x,y) = <x,y>
        k = pairwise.linear_kernel
    elif kernel == "rbf":
        k = pairwise.rbf_kernel  # exp(-gamma * ||x-y||^2)
    elif kernel == "laplacian":
        k = pairwise.laplacian_kernel  # exp(-gamma * ||x-y||)
    else:
        raise ValueError("Kernel not implemented")

    # Apply tensor normalization on the signatures if required
    assert sig1.shape[1] == sig2.shape[1]
    if normalization:
        old_shape1 = sig1.shape
        old_shape2 = sig2.shape
        sig1, sig2 = np.array(sig1), np.array(sig2)
        sig1 = np.stack(
            [np.array(tensor_normalization(sig1[i])) for i in range(sig1.shape[0])]
        )
        sig2 = np.stack(
            [np.array(tensor_normalization(sig2[i])) for i in range(sig2.shape[0])]
        )
        assert sig1.shape == old_shape1 and sig2.shape == old_shape2

    # Monte Carlo approximation of expectation d_k^2(mu, nu) = E[k(X,X')] + E[k(Y,Y')] - 2E[k(X,Y)] where X, X' ~ mu and Y, Y' ~ nu
    A_XX = k(sig1, sig1)
    A_XY = k(sig1, sig2)
    A_YY = k(sig2, sig2)
    distance_sq = np.mean(A_XX) + np.mean(A_YY) - 2 * np.mean(A_XY)
    distance = np.sqrt(distance_sq)
    print(f"Distance = {distance}") if verbose else None
    return distance


def compute_SigMMD_for_paths(
    path_untransformed_A,
    path_untransformed_B,
    depth,
    kernel,
    normalization,
    verbose=False,
):
    """
    Utility function for (robust/naive) Sig-MMD for two paths datasets.
    Boilerplate code for reshaping and calling sig_mmd.

    Parameters
    ----------
    path_untransformed_A : torch.Tensor
        Path dataset A, shape (L, N, C) = (n_t, n_x, d)
    path_untransformed_B : torch.Tensor
        Path dataset B, shape (L, N, C) = (n_t, n_y, d)
    depth : int
        Depth of signature transform
    kernel : str
        Kernel to use, either 'linear' or 'rbf'.
    normalization : bool
        Whether to use tensor normalization or not.
    verbose : bool
        Whether to print debug information or not.

    Returns
    -------
    sigmmd : np.float

    """
    # Reshape permutation for signature and compute signature
    path_A = torch.permute(path_untransformed_A, (1, 0, 2))  # shape N, L, C
    path_B = torch.permute(path_untransformed_B, (1, 0, 2))
    logging.info(
        f"Path shape (should be N(umber), L(ength), C(hannels)) = {path_A.shape})"
    ) if verbose else None
    signature_A = signatory.signature(path_A, depth)
    signature_B = signatory.signature(path_B, depth)
    logging.info(
        f"Signature shape (should be N_A, (C+C^2+...+C^depth)) = {signature_A.shape}"
    )
    logging.info(
        f"Signature shape (should be N_B, (C+C^2+...+C^depth)) = {signature_B.shape}"
    )

    # Compute SigMMD
    return sig_mmd(
        signature_A,
        signature_B,
        kernel=kernel,
        normalization=normalization,
        verbose=verbose,
    )


def compute_RobustSigMMD(x, y, kernel):
    """
    Compute Robust SigMMD for observation level data x and y.
    Boilerplate code that calls function.

    Parameters
    ----------
    x : np.ndarray
        Observation level data, shape (n_t, n_x, d)
    y : np.ndarray
        Observation level data, shape (n_t, n_y, d)
    kernel : str
        Kernel to use, either 'linear' or 'rbf'.

    Returns
    -------
    sigmmd : float
        Robust SigMMD of x and y.
    """
    x = torch.tensor(x)
    y = torch.tensor(y)
    return compute_SigMMD_for_paths(
        x, y, depth=5, kernel=kernel, normalization=True, verbose=False
    )


def compute_conventional_SigMMD(x, y):
    """
    Compute conventional depth-5 SigMMD for observation level data x and y.

    Parameters
    ----------
    x : np.ndarray
        Observation level data, shape (n_t, n_x, d)
    y : np.ndarray
        Observation level data, shape (n_t, n_y, d)

    Returns
    -------
    sigmmd : float
        SigMMD of x and y.
    """
    x = torch.tensor(x)
    y = torch.tensor(y)

    # Reshape permutation for signature and compute signature
    path_A = torch.permute(x, (1, 0, 2))  # shape N, L, C
    path_B = torch.permute(y, (1, 0, 2))
    logging.info(
        f"Path shape (should be N(umber), L(ength), C(hannels)) = {path_A.shape})"
    )
    signature_A = signatory.signature(path_A, 5)
    signature_B = signatory.signature(path_B, 5)
    logging.info(
        f"Signature shape (should be N_A, (C+C^2+...+C^5)) = {signature_A.shape}"
    )
    logging.info(
        f"Signature shape (should be N_B, (C+C^2+...+C^5)) = {signature_B.shape}"
    )

    # Compute SigMMD
    mean_A = torch.mean(signature_A, dim=0)
    mean_B = torch.mean(signature_B, dim=0)
    return torch.norm(mean_A - mean_B, p=2).item()


if __name__ == "__main__":
    test_psi()

    # TODO : Run compute_RobustSigMMD on results from model training.

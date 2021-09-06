import torch
import numpy as np
from collections import Counter


def Dn_fun(omega_vals, U, nk):
    U = U.reshape((-1, 1))
    n = torch.sum(nk)
    # max_U = torch.max(U)

    log_vals = torch.log(torch.mm(omega_vals, torch.exp(U)))
    main_term = torch.mean(log_vals)
    # log_vals = torch.log(torch.mm(omega_vals, torch.exp(U-max_U)))
    # main_term = max_U + torch.mean(log_vals)
    other_term = torch.dot(nk.float(), U.flatten()) / n
    Dn = main_term - other_term
    return Dn


def U_from_Omegas(Omegas, nk):
    Omegas = Omegas.flatten()
    nk = nk.flatten()
    n = torch.sum(nk)
    U = torch.log(nk / (n * Omegas)).float()
    return U.reshape((-1, 1))


def Omegas_from_U(U, nk):
    U = U.flatten()
    nk = nk.flatten()
    n = torch.sum(nk)
    W = (nk / (torch.exp(U) * n)).float()
    return W.reshape((-1, 1))


def solve_Ws(
    omegas,
    source_datasets,
    n_iter=1000,
    batch_size=100,
    lr=0.01,
    momentum=0.9,
    verbose=True,
):
    """
    * omegas is a n x K numpy matrix with the values of the omega functions,
    i.e. (omegas)_{i,j} = \omega_j(Z_{k, l}) where (k,l) \in \Lambda is
    the ith observation.
    """
    K = omegas.shape[1]
    U = torch.normal(0, 1, (K, 1), requires_grad=True)
    nks = dict()
    for k in source_datasets:
        nks[k] = nks.get(k, 0) + 1
    nk_vec = torch.tensor([nks[k] for k in range(0, K)])

    # Solve the minimization of D_n by gradient descent:
    omegas = torch.tensor(omegas).float()
    data_loader = torch.utils.data.DataLoader(
        omegas, shuffle=True, batch_size=batch_size
    )
    optimizer = torch.optim.SGD([U], lr=lr, momentum=momentum)

    i = 0
    while i < n_iter:
        for batch in data_loader:
            optimizer.zero_grad()
            Dn = Dn_fun(omegas, U, nk_vec)
            Dn.backward()
            optimizer.step()
            if verbose and i % 100 == 0:
                print(
                    ("i = {}/{}, Dn = {:.5f} - " "- grad_norm = {:.2f}").format(
                        i, n_iter, Dn.item(), torch.norm(U.grad)
                    ),
                    flush=True,
                )
                # print("{:.5f} - {:.5f}".format(U[0].item(), U[1].item()))
            i += 1
            if i >= n_iter:
                break

    # Recover the W_s
    W = Omegas_from_U(U, nk_vec).detach().numpy().ravel()
    return W


def get_weights(omegas, source_datasets, W):
    K = omegas.shape[1]
    nks = dict()
    for k in source_datasets:
        nks[k] = nks.get(k, 0) + 1
    nk_vec = torch.tensor([nks[k] for k in range(0, K)])
    n = torch.sum(nk_vec)
    tmp = torch.mm(
        torch.tensor(omegas).float(), (nk_vec / (n * W)).reshape((-1, 1)).float()
    )
    weights = 1 / tmp
    tot_weights = torch.sum(weights)
    res = (weights / tot_weights).detach().numpy()
    return res


def main():
    print("Hello world")
    # How to use
    # W = solve_Ws(omegas, source_datasets, n_iter=500, batch_size=100, lr=0.001)

    # sample_weights = get_weights(omegas, source_datasets, W)
    # ideal_weights = get_weights(omegas, source_datasets, Omegas)


if __name__ == "__main__":
    main()

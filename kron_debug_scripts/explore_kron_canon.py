import numpy as np

import cvxpy as cp

"""
kron(M, N) = [M[0,0] * N   , ..., M[0, end] * N  ]
             [M[1,0] * N   , ..., M[1, end] * N  ]
             ...
             [M[end, 0] * N, ..., M[end, end] * N]
"""


def random_problem(z_dims, c_dims, param, seed=0):
    np.random.seed(seed)
    _C_value = np.random.rand(*c_dims).round(decimals=2)
    if param:
        _C = cp.Parameter(shape=c_dims)
        _C.value = _C_value
    else:
        _C = cp.Constant(_C_value)
    _Z = cp.Variable(shape=z_dims)
    _L = np.random.rand(*z_dims).round(decimals=2)
    _constraints = [cp.kron(_C, _Z) >= cp.kron(_C, _L), _Z >= 0]
    # ^ Only the first constraint matters, since _C and _L are nonnegative.
    # We use two constraints because that makes it easier to set
    # conditional breakpoints in canonInterface.py.
    #
    #   Specifically, canonInterface.py:get_problem_matrix is called once for
    #   the objective function and once for the constraint matrix. When it's
    #   called for the objective function the linOps argument is a list of length
    #   one, and when it's called for the constraint matrix the linOps argument
    #   is a list of length equal to the number of constraints.
    #
    _obj_expr = cp.sum(_Z)

    _prob = cp.Problem(cp.Minimize(_obj_expr), _constraints)
    # The optimal solution is Z.value == L.
    return _Z, _C, _L, _prob


def run_example(solve=True):
    seed = 0
    z_dims = (2, 2)
    c_dims = (1, 2)

    param = True  # tests pass when param=False. But maybe worth looking at ...
    Z, C, L, prob = random_problem(z_dims, c_dims, param, seed=seed)
    if solve:
        prob.solve(solver='ECOS')
        print('\nVariable "Z" is the right operand in kron.\nSolving with ECOS ...')
        violations = prob.constraints[0].violation()
        print(f'\tReported problem status: {prob.status}')
        print(f'\tConstraint violation: {np.max(violations)}')
    print('\nDimensions')
    print(f'\tz_dims = {z_dims}')
    print(f'\tc_dims = {c_dims}')
    data = prob.get_problem_data(solver='ECOS', enforce_dpp=False)[0]
    G = data['G'].A
    h = data['h']
    print(f'\nECOS data for param={param}.')
    print(f'\tnnz(G) = {np.count_nonzero(G)}')
    print(f'\tnnz(h) = {np.count_nonzero(h)}\n')
    print(G)
    print(h)
    print()


if __name__ == '__main__':
    solve = True  # same bad results when solve=False.
    run_example(solve)  # compiles incorrectly
    run_example(solve)  # compiles correctly

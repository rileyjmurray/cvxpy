import numpy as np

import cvxpy as cp


def print_array_indented(mat, indent_level=1):
    tab_str = indent_level * '\t'
    mat_str = str(mat)
    mat_str = tab_str + mat_str
    mat_str = mat_str.replace('\n ', f'\n{tab_str} ')
    print(mat_str)


def random_problem(z_dims, c_dims, param, seed=0):
    """
      Construct random nonnegative matrices (C, L) of shapes
      (c_dims, z_dims) respectively. Define an optimization
      problem with a matrix variable of shape z_dims:

        min   sum(Z)
        s.t.  kron(C, Z) >= kron(C, L)
              Z >= 0

      The optimal solution to that problem is Z = L.
      The constraint Z >= 0 is redundant; it's mostly there in case
      the first constraint gets compiled incorrectly.

      If param is True, then C is defined as a CVXPY Parameter.
      If param is False, then C is a CVXPY Constant.
    """
    # Use underscores in all variable names to make sure we
    # don't accidentally pick something up from the outer scope.
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
    _obj_expr = cp.sum(_Z)

    _prob = cp.Problem(cp.Minimize(_obj_expr), _constraints)
    return _Z, _C, _L, _prob


def run_example(solve=True, param=True):
    seed = 0
    z_dims = (2, 2)
    c_dims = (2, 3)

    print('\nContext for this run ...')
    print(f'\tshape of Variable  Z: {z_dims}')
    c_desc = 'Parameter' if param else 'Constant '
    print(f'\tshape of {c_desc} C: {c_dims}')
    Z, C, L, prob = random_problem(z_dims, c_dims, param, seed=seed)

    if solve:
        prob.solve(solver='ECOS')
        print('\nSolving with ECOS ...')
        violations = prob.constraints[0].violation()
        print(f'\tProblem status: {prob.status}')
        print(f'\tZ.value = ...\n')
        print_array_indented(Z.value, indent_level=2)
        print(f'\n\tConstraint violation: {np.max(violations)}')

    print('\nExamining ECOS problem data ...')
    data = prob.get_problem_data(solver='ECOS', enforce_dpp=True)[0]
    # ^ Changing to enforce_dpp=False doesn't make a difference
    G = data['G'].A
    h = data['h']
    print('\n\tThe matrix G in "G x <= h" is \n')
    print_array_indented(G, indent_level=2)
    print('\n\tThe vector h in "G x <= h" is \n')
    print_array_indented(h, indent_level=2)
    print()


if __name__ == '__main__':
    print(80 * '-')
    print(35 * ' ' + 'FIRST RUN ' + 35 * ' ')
    print(80 * '-')
    run_example(solve=True, param=True)  # compiles incorrectly
    print(80 * '-')
    print(35 * ' ' + 'SECOND RUN ' + 35 * ' ')
    print(80 * '-')
    run_example(solve=True, param=True)  # compiles correctly
    # ^ The (G, h) there are very weird, but produce the correct solution.
    #   They do indeed appear to be what's passed to ECOS!
    print(80 * '-')
    print(35 * ' ' + 'THIRD RUN ' + 35 * ' ')
    print(80 * '-')
    run_example(solve=True, param=False)  # compiles correctly
    # ^ The (G, h) look like how you would expect them to.

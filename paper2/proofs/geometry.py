"""Response geometry of the B-channel credit operator, exactly.

Rational columns with the declared B-channel support: recurrent-matrix columns
occupy only the budget row of their owner channel; local controls occupy all
three rows of their channel; four shared controls span every channel. Checks:
Gram = block-diagonal owned factors + shared factor; response second moments
equal K^T K; dense ridge inverse equals the block-plus-rank-four Woodbury form;
the direct action solves the ridge normal equations. Algebra only, no native
trajectory or learning claim.
"""
from fractions import Fraction as Q

from _exact import add, eye, multiply, solve, transpose


def instance(seed):
    # I=1, S=2, B=1: 22 owned columns/channel plus four shared columns.
    # Matrix columns occupy only budget; local controls occupy all three rows.
    # Local columns include an exact zero and a duplicate.
    owned = []
    for channel in range(2):
        block = [[Q(0) for _ in range(22)] for _ in range(3)]
        for column in range(6):
            block[0][column] = Q((column+seed+channel)%5-2,7)
        for column in range(6,22):
            for row in range(3):
                block[row][column] = Q((3*column+row+seed+channel)%7-3,11)
        for row in range(3):
            block[row][6] = Q(0)
            block[row][8] = block[row][7]
        owned.append(block)
    k = [[Q(0) for _ in range(48)] for _ in range(6)]
    for channel,block in enumerate(owned):
        for row in range(3):
            k[3*channel+row][22*channel:22*(channel+1)] = block[row]
    g = [[Q((2*row+column+seed)%5-2,13) for column in range(4)]
         for row in range(6)]
    for row in range(6):
        k[row][44:] = g[row]
    return k,owned,g


def check():
    rows = []
    for seed in range(5):
        k,owned,g = instance(seed)
        kt,gt = transpose(k),transpose(g)
        gram = multiply(k,kt)
        block_gram = [[Q(0) for _ in range(6)] for _ in range(6)]
        for channel,block in enumerate(owned):
            local = multiply(block,transpose(block))
            for row in range(3):
                block_gram[3*channel+row][3*channel:3*channel+3] = local[row]
        assert gram == add(block_gram,multiply(g,gt))
        for channel in range(2):
            for column in range(22*channel,22*(channel+1)):
                assert all(k[row][column]==0 for row in range(6)
                           if row//3!=channel)
        # Summed second moments over basis signals equal K^T K exactly.
        responses = multiply(kt,eye(6))
        covariance = multiply(responses,transpose(responses))
        assert covariance == multiply(kt,k)
        assert all(covariance[i][i] == sum((v*v for v in kt[i]),Q(0))
                   for i in range(48))
        tau = Q(1,3)
        regularized = add(gram,eye(6,tau))
        dense_inverse = solve(regularized,eye(6))
        m_inverse = [[Q(0) for _ in range(6)] for _ in range(6)]
        for channel,block in enumerate(owned):
            local_inverse = solve(add(multiply(block,transpose(block)),eye(3,tau)),eye(3))
            for row in range(3):
                m_inverse[3*channel+row][3*channel:3*channel+3] = local_inverse[row]
        mg = multiply(m_inverse,g)
        correction_inverse = solve(add(eye(4),multiply(gt,mg)),eye(4))
        structured_inverse = add(m_inverse,
            multiply(multiply(mg,correction_inverse),transpose(mg)),sign=-1)
        assert dense_inverse == structured_inverse
        actions = multiply(kt,dense_inverse)
        # The direct action solves the declared ridge normal equations for
        # every basis signal simultaneously; the minus signs cancel here.
        normal = add(multiply(kt,k),eye(48,tau))
        assert multiply(normal,actions) == kt
        rows.append({'seed':seed,'operator_shape':[6,48],
            'zero_local_columns_per_channel':1,'redundant_local_columns_per_channel':1,
            'gram_structure':'exact','response_second_moment':'exact',
            'dense_vs_woodbury':'exact','action_normal_equations':'exact',
            'nonlocal_owned_support':'exact zero'})
    return {'scope':'rational algebra with declared support; no native trajectory or learning claim',
            'arithmetic':'fractions.Fraction','cases':rows}

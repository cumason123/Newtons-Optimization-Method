from sympy import *
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import copy
from numpy import matrix


def find_inverse_simple_jacobian(jacobian, x1, x1_guess, x2, x2_guess):#assumes jacobian matrix is balanced
    new_jacobian = copy.copy(jacobian)
    for x in range(len(new_jacobian)):
        for y in range(len(new_jacobian[0])):
            new_jacobian[x][y] = float(new_jacobian[x][y].subs([(x1, x1_guess), (x2,x2_guess)]).evalf())
    print('jacobian: {0}'.format(new_jacobian.tolist()))
    return np.array(matrix(new_jacobian.tolist()).I)


def make_guess(guess = None):
    if guess is None:
        print('Guess is None')
        x1 = Symbol('x1')
        x2 = Symbol('x2')
        f1 = sin(2*x1 + 3*x2)
        f2 = 2*(x1**2) - x2 - 15

        x1_df1 = diff(f1, x1)
        x2_df1 = diff(f1, x2)

        x1_df2 = diff(f2, x1)
        x2_df2 = diff(f2, x2)

        x1_guess = 1
        x2_guess = 1
        print('x1_guess, x2_guess: {0}, {1}'.format(x1_guess, x2_guess))
    else:
        x1_df1 = guess['x1_df1']
        x2_df1 = guess['x2_df1']
        x1_df2 = guess['x1_df2']
        x2_df2 = guess['x2_df2']

        x1 = guess['x1']
        x2 = guess['x2']
        f1 = guess['f1']
        f2 = guess['f2']

        x1_guess = guess['x1_guess']
        x2_guess = guess['x2_guess']
        print('x1_guess, x2_guess: {0}, {1}'.format(x1_guess, x2_guess))
    start_val = np.array([x1_guess, x2_guess])
    jacobian = np.array([[x1_df1, x2_df1],[x1_df2, x2_df2]])

    f1_eval = f1.subs([(x1,x1_guess), (x2, x2_guess)]).evalf()
    f2_eval = f2.subs([(x1,x1_guess), (x2, x2_guess)]).evalf()
    evaluation = np.array([f1_eval, f2_eval])

    inverse_jacobian = find_inverse_simple_jacobian(jacobian, x1, x1_guess, x2, x2_guess)
    print(inverse_jacobian)
    result = start_val - np.matmul(inverse_jacobian, [float(x) for x in evaluation])
    return {'x1_df1':x1_df1,'x2_df1':x2_df1,
            'x1_df2':x1_df2,'x2_df2':x2_df2,
            'x1':x1,'x2':x2,'f1':f1,'f2':f2,
            'x1_guess':result[0],'x2_guess':result[1]}


def plot(cycles=50):
    initial_guess = make_guess()
    f1, f2, x1, x2 = initial_guess['f1'], initial_guess['f2'], initial_guess['x1'], initial_guess['x2']
    f1_y = [f1.subs([(x1,initial_guess['x1_guess']),(x2,initial_guess['x2_guess'])])]
    f2_y = [f2.subs([(x1,initial_guess['x1_guess']),(x2,initial_guess['x2_guess'])])]
    f1_x = np.arange(cycles)
    f2_x = np.arange(cycles)

    for _ in range(cycles-1):

        try:
            print('\n'+str(_)+':')
            if _ == 0:
                items = make_guess(guess=initial_guess)
            else:
                items = make_guess(guess=items)
            f1_y.append(f1.subs([(x1, items['x1_guess']), (x2, items['x2_guess'])]))
            f2_y.append(f2.subs([(x1, items['x1_guess']), (x2, items['x2_guess'])]))
        except AttributeError:
            break
    print(len(f2_x))
    print(len(f1_x))
    fig = plt.figure()
    plt.plot(f1_x, f1_y, 'r--', f2_x, f2_y, 'k')
    plt.show()
    plt.waitforbuttonpress(0)

    plt.close()
    plt.clf()

if __name__=='__main__':
    plot()
from sympy import *
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import copy
from numpy import matrix


class NonlinearSystems(object):
    def __init__(self, equation, data, initial_guess=None):
        """
        Optimizes an equation such that the equation fits onto a set of data

        :param data: 2D list or numpy array with paired data of shape (None, 2)

                Example 1: data = [(x1, y1), (x2, y2), (x3, y3), ...]
                Example 2: data = [[x1, y1], [x2, y2], [x3, y3], ...]
                Example 3: data = np.array([(x1, y1), [x2, y2], (x3, y3), ...])

        :param equation: sympy symbolic equation to be optimized based
            on given datapoints

                Example: equation = Symbol('X')**2 + Symbol('A')
                    where Symbol('X') is the independent variable and must be present

        :param initial_guess: a dict where keys are symbols,
            values are corresponding symbolic values

            Example: initial_guess = {'a':6, 'b':3, ...}
        """
        self.data = np.array(data)
        self.equation = equation
        self.nonlinear_functions = [y_point - equation.subs('X', x_point)
                                    for x_point, y_point in data]
        self.parametric_symbols = [str(symbol) for symbol in list(equation.free_symbols) if
                                   str(symbol) != 'X']
        if initial_guess is None:
            self.initial_guess = {symbol: 1 for symbol in self.parametric_symbols}
        else:
            self.initial_guess = initial_guess
        self.jacobian = [
            [diff(func, symbol) for symbol in self.parametric_symbols]
            for func in self.nonlinear_functions
        ]

    def find_inverse_simple_jacobian(self, jacobian, parameters):
        """
        Converts a given symbolic jacobian array into
        it's appropriate inverse jacobian matrix

        :param jacobian: a 2D list representing a sympy
            generated symbolic jacobian matrix

        :param parameters: dictionary where key's are function parameter
            symbols, values are their substitution value

        :return: np.array of the evaluated inverse_jacobian matrix
        """

        # Creates a new matrix
        new_jacobian = copy.copy(jacobian)
        new_jacobian = np.array(new_jacobian)
        # Converts symbolic math into floating point values
        for x in range(len(new_jacobian)):
            for y in range(len(new_jacobian[0])):
                new_jacobian[x][y] = float(new_jacobian[x][y].subs(
                    [(symbol_key, parameters[symbol_key]) for symbol_key in self.parametric_symbols]
                ).evalf())
        return np.array(matrix(new_jacobian.tolist()).I)

    def make_guess(self, guess=None):
        """
        Utilizes one cycle of Newton's Method for optimization based on a given guess

        :param guess: a dict where keys are parametric symbols, and values are
            parametric symbolic values to be substituted into the function during
            optimization

        :return: a dict of the newly optimized guess parameters
            where keys are parametric symbols and values are optimized values
        """

        if guess is None:
            guess = self.initial_guess

        start_val = [self.initial_guess[symbol] for symbol in self.parametric_symbols]
        jacobian = self.jacobian

        evaluation = self.evaluation = [float(func.subs([(symbol, self.initial_guess[symbol])
                                            for symbol in self.parametric_symbols]).evalf()) for
                           func in self.nonlinear_functions]

        inverse_jacobian = self.find_inverse_simple_jacobian(jacobian, guess)
        result = start_val - np.matmul(inverse_jacobian, [x for x in evaluation])
        for index, key in enumerate(self.parametric_symbols):
            self.initial_guess[key] = result[index]

    def train(self, cycles=100):
        for _ in range(cycles):
            print('Cycle: {0}'.format(_))
            self.make_guess(self.initial_guess)

    def fit(self, independent):
        fit = self.equation.subs([(symbol, self.initial_guess[symbol]) for symbol in
                                       self.initial_guess])
        newfit = lambda x: float(fit.subs('X', x).evalf())
        return [newfit(x) for x in independent]

if __name__ == '__main__':
    x_points = np.linspace(0, 10, 50)

    x = Symbol('X')
    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')
    d = Symbol('d')

    func = a*x*sin(x)+b
    newfunc = func.subs([('a', 3), ('b', 5), ('c', 2.3), ('d', 1.7)])
    f = lambda independent: float(newfunc.subs('X', independent).evalf())
    y_points = [f(_) for _ in x_points]
    data = [(x_points[_], y_points[_]) for _ in range(len(x_points))]
    obj = NonlinearSystems(func, data)
    obj.train()

    print(obj.initial_guess)
    new_y = obj.fit(x_points)
    x_points = np.linspace(0, 100, 300)
    new_y = obj.fit(x_points)
    y_points = [f(_) for _ in x_points]
    plt.plot(x_points, y_points, 'k*', x_points, new_y, 'k')
    plt.show()
    #plt.plot(x_points, np.array(y_points)-np.array(new_y), 'k')
    #plt.show()

from __NonLinearSystems import NonLinearSystems
from sympy import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def sinusodial_func(x, a):
    return a*x*np.sin(x)


def sin_test():
    # Independent Values
    x_points = np.arange(20)

    # Symbols
    a, b, c, d, x = Symbol('a'), Symbol('b'), Symbol('c'), Symbol('d'), Symbol('x')

    # Function type to approximate onto data
    func = a*sin(b*x+c)+d

    # Parameters to determine
    newfunc = func.subs([('a', 3), ('b', 30), ('c', 18.3), ('d', 1.7)])

    # Test Function
    f = lambda independent: float(newfunc.subs('x', independent).evalf())

    # Adds noise to y values
    y_points = np.array([f(x) for x in x_points]) + np.random.random()
    # Formatted data to be later fed into the class for training
    data = [(x_points[_], y_points[_]) for _ in range(len(x_points))]

    # 10 cycles of Newton's Method
    obj = NonLinearSystems(func, data)
    obj.train(cycles=50)

    # Returns yhat vals
    new_y = obj.fit(x_points, best_fit=True)
    scipy = curve_fit(sinusodial_func, x_points, y_points)
    print(obj)
    print(scipy[0])
    # Plot
    plt.plot(x_points, y_points, 'k*', x_points, new_y, 'b--')

    plt.show()


def linear_test():
    # Independent Variables
    x_points = np.arange(1)

    # Sympy parameters used during optimization, and x the independent variable symbol
    a, b, x = Symbol('a'), Symbol('b'), Symbol('x')

    # Our defined function
    linear_func = (x*a) + b

    # Parameters we wish to test for
    y_func = linear_func.subs([('a', 3), ('b', 20)])

    # Our y values
    y_vals = [float(y_func.subs('x', independent).evalf()) for independent in x_points]

    # Prepare our data
    data = np.array([x_points,y_vals]).T

    # 10 cycles of Newton's Method
    obj = NonLinearSystems(linear_func, data)
    obj.train(cycles=50)

    print(obj)

    # Returns yhat vals
    new_y = obj.fit(x_points, best_fit=True)

    # Plot
    plt.plot(x_points, y_vals, 'k*', x_points, new_y, 'b--')

    plt.show()


def main():
    sin_test()
    # linear_test()


if __name__ == '__main__':
    main()

from NonLinearSystems import NonLinearSystems
from sympy import *
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Independent Values
    x_points = np.arange(50)

    # Symbols
    x = Symbol('X')
    a, b, c, d = Symbol('a'), Symbol('b'), Symbol('c'), Symbol('d')

    # Function type to approximate onto data
    func = a*log(b*x+c)+d

    # Parameters to determine
    newfunc = func.subs([('a', 3), ('b', 30), ('c', 18.3), ('d', 1.7)])

    # Test Function
    f = lambda independent: float(newfunc.subs('X', independent).evalf())

    # Adds noise to y values
    y_points = np.array([f(x) for x in x_points]) + np.random.random()
    # Formatted data to be later fed into the class for training
    data = [(x_points[_], y_points[_]) for _ in range(len(x_points))]

    # 10 cycles of Newton's Method
    obj = NonLinearSystems(func, data)
    obj.train(cycles=10)

    # Returns yhat vals
    new_y = obj.fit(x_points, best_fit=True)

    # Plot
    plt.plot(x_points, y_points, 'k*', x_points, new_y, 'b--')
    plt.show()



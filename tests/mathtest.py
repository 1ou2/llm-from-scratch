import math
import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return 3*x**2 - 4*x +3

def hyperbole():
    xs = np.arange(-5,5,0.25)
    print(xs)
    ys = f(xs)
    print(ys)

    plt.plot(xs,ys)
    plt.show()

def derivative(x):
    # h doit tendre vers 0
    h = 0.0001
    return (f(x+h)-f(x))/h
hyperbole()
print(f(3.0))
print(derivative(3.0))
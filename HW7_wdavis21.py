import numpy as np
import math

def midpoint(f,a,b,n):
    h = (b-a)/n
    result = 0
    for i in range(n):
        xi=a+i*h
        result+=h*f(xi+(h/2))
    return result

def trapezoid(f,a,b,n):
    h = (b-a)/n
    result = (h/2)*(f(a)+f(b))
    for i in range(1,n):
        result+=f(a+i*h)
    result*=h
    return result

def gauss_quad(f,a,b,n):
    if n == 2:
        # 2-point Gaussian quadrature roots and coefficients
        roots = [-0.5773502692, 0.5773502692]
        coefficients = [1, 1]
    elif n == 3:
        # 3-point Gaussian quadrature roots and coefficients
        roots = [-0.7745966692, 0, 0.7745966692]
        coefficients = [0.5555555556, 0.8888888889, 0.5555555556]
    elif n == 4:
        # 4-point Gaussian quadrature roots and coefficients
        roots = [-0.8611363116, -0.3399810436, 0.3399810436, 0.8611363116]
        coefficients = [0.3478548451, 0.6521451549, 0.6521451549, 0.3478548451]
    elif n == 5:
        # 5-point Gaussian quadrature roots and coefficients
        roots = [-0.9061798459, -0.5384693101, 0, 0.5384693101, 0.9061798459]
        coefficients = [0.2369268850, 0.4786286705, 0.5688888889, 0.4786286705, 0.2369268850]
    else:
        raise ValueError("Nodes other than 2, 3, 4, 5 are not supported.")

    def transform(t):
        return 0.5 * (b - a) * t + 0.5 * (a + b)

    result = 0
    for i in range(n):
        result += coefficients[i] * f(transform(roots[i]))

    result *= 0.5 * (b - a)  # Scale by (b - a) / 2
    return result


#Testing Midpoint, Trapezoid, Gauss for integral from [0,1] x^2 dx
f1 = lambda x: x**2
a = 0
b = 1

print("Testing Midpoint and Trapezoid Methods for x^2 dx:")
for n in [1, 4, 10, 50]:
    mid_result = midpoint(f1, a, b, n)
    trap_result = trapezoid(f1, a, b, n)
    print(f"n = {n}: Midpoint = {mid_result}, Trapezoid = {trap_result}")
    
print("\nTesting Gaussian Quadrature Method x^2 dx:")
for n in [2, 3, 4, 5]:
    gauss_result = gauss_quad(f1, a, b, n)
    print(f"n = {n}: Gaussian Quadrature = {gauss_result}")

print("---------------------------------------------------------")

#Testing Midpoint, Trapezoid, Gauss for integral from [1,2] xln(x) dx
def f2(x):
    return x*math.log(x)

a1=1
b1=2

print("Testing Midpoint and Trapezoid Methods xln(x) dx:")
for n in [1, 4, 10, 50]:
    mid_result = midpoint(f2, a1, b1, n)
    trap_result = trapezoid(f2, a1, b1, n)
    print(f"n = {n}: Midpoint = {mid_result}, Trapezoid = {trap_result}")
    
print("\nTesting Gaussian Quadrature Method xln(x) dx:")
for n in [2, 3, 4, 5]:
    gauss_result = gauss_quad(f2, a1, b1, n)
    print(f"n = {n}: Gaussian Quadrature = {gauss_result}")

print("---------------------------------------------------------")

def f3(x):
    return np.sqrt(1+np.cos(x)**2)

a2=0
b2=48

print("Testing Midpoint and Trapezoid Methods sqrt 1+cos^2(x) dx:")
for n in [1, 4, 10, 50]:
    mid_result = midpoint(f3, a2, b2, n)
    trap_result = trapezoid(f3, a2, b2, n)
    print(f"n = {n}: Midpoint = {mid_result}, Trapezoid = {trap_result}")
    
print("\nTesting Gaussian Quadrature Method sqrt 1+cos^2(x) dx:")
for n in [2, 3, 4, 5]:
    gauss_result = gauss_quad(f3, a2, b2, n)
    print(f"n = {n}: Gaussian Quadrature = {gauss_result}")



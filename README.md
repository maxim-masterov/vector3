# Intro
This project represents a small set of headers with classes for representation of coordinates in 3d space. 
All classes have a unified set of functions and overload operators to simlify math operations on coordinates, e.g. dot product, 
cross product, scaling.

The `vector3_reg` class is a standard version which operates on `float` and `double` datatypes.

The `vector3f_simd` is a SSE optimized version of `vector3_reg` class which uses low level intrinsics to perfrom operations on
a coordinates represented by `float`.

The `vector3d_simd` is a SSE optimized version of `vector3_reg` class which uses low level intrinsics to perfrom operations on
a coordinates represented by `double`.

# Example
An example of usage:
```
typedef vector3_reg vector3;
vector3 a, b;

a << 1.1, 2.2, 3.3;
b << -1.1, -2.2, 8.3;

a = 2. * b - a;

std::cout << a << "\n";
```

# HowTo
To start using the project simply include `Vectors.h` header.

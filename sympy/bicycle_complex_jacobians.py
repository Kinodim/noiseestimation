from sympy.abc import J, m, v, beta, delta
from sympy import symbols, Matrix, pprint


if __name__ == "__main__":
    dt, a, c_v, c_h, l_v, l_h, psi_d = symbols("dt a c_v c_h l_v l_h psi_d")

    fxu = Matrix([[(1 - dt * (c_v + c_h) / m / v) * beta
                   - dt * (1 + (c_v * l_v - c_h * l_h) / m / v**2) * psi_d
                   + dt * c_v / m / v * delta],
                  [-dt * (c_v * l_v - c_h * l_h) / J * beta
                   + (1 - dt * (c_v * l_v**2 + c_h * l_h**2) / J / v) * psi_d
                   + dt * c_v * l_v / J * delta]])
    print("Update function f(x,u):")
    pprint(fxu)

    F = fxu.jacobian(Matrix([beta, psi_d]))
    print("\nJacobian of F with respect to state:")
    pprint(F)
    print("\nRaw:")
    print(F)

    V = fxu.jacobian(Matrix([delta, v]))
    print("\nJacobian of F with respect to control:")
    pprint(V)
    print("\nRaw:")
    print(V)

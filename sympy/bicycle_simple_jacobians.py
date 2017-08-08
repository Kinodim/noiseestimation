import sympy
from sympy.abc import alpha, x, y, v, w, theta
from sympy import symbols, Matrix, pprint


if __name__ == "__main__":
    time = symbols('t')
    d = v * time
    beta = (d/w)*sympy.tan(alpha)
    r = w/sympy.tan(alpha)

    fxu = Matrix([[x - r * sympy.sin(theta) + r * sympy.sin(theta + beta)],
                  [y + r * sympy.cos(theta) - r * sympy.cos(theta + beta)],
                  [theta + beta]])
    F = fxu.jacobian(Matrix([x, y, theta]))

    B, R = symbols('beta, R')
    F = F.subs((d / w) * sympy.tan(alpha), B)
    F = F.subs(w / sympy.tan(alpha), R)
    print("\nJacobian of F with respect to state:")
    pprint(F)
    print("\nRaw:")
    print(F)
    print("\nWith alpha = 0:")
    F = F.subs(B, 0)
    F = F.subs(R, sympy.oo)
    pprint(F)

    V = fxu.jacobian(Matrix([v, alpha]))
    V = V.subs(sympy.tan(alpha) / w, 1/R)
    V = V.subs(w / sympy.tan(alpha), R)
    V = V.subs(time * v / R, B)
    V = V.subs(time * v, 'd')
    print("\nJacobian of F with respect to control:")
    pprint(V)
    print("\nRaw:")
    print(V)
    print("\nWith alpha = 0:")
    V = V.subs(B, 0)
    V = V.subs(R, sympy.oo)
    V = V.subs(alpha, 0)
    pprint(V)

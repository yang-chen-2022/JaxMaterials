from sympy import *
from sympy.tensor.array.expressions import ArrayTensorProduct, ArraySymbol
from sympy.vector import Vector


xi = Array(
    symbols(" ".join([f"\\mathring{{\\widetilde{{\\xi}}}}_{j}" for j in range(3)]))
)
K = zeros(3, 3)

C = ArraySymbol("C", (21,))
epsilon = ArraySymbol("varepsilon", (6,))

sigma = zeros(6)


voigt_indices = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
C_indices = (
    (0, 0),
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 4),
    (5, 5),
    (0, 1),
    (0, 2),
    (1, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (1, 3),
    (1, 4),
    (1, 5),
    (2, 3),
    (2, 4),
    (2, 5),
    (3, 4),
    (3, 5),
    (4, 5),
)

tuple((i, j) for i in range(6) for j in range(i, 6))
voigt_map = {k: j for j, k in enumerate(voigt_indices)}
C_map = {k: j for j, k in enumerate(C_indices)}

#### Matrix - vector product sigma = C epsilon

for a, (i, j) in enumerate(voigt_indices):
    for k in range(3):
        for ell in range(3):
            b = voigt_map[tuple(sorted([k, ell]))]
            _a, _b = sorted((a, b))
            sigma[a] += C[C_map[(_a, _b)]] * epsilon[b]
    print(f"\\sigma_{{{a}}} &= ", latex(simplify(sigma[a])), r"\\")

# TODO: generate Jax code for this

# K - matrix

for k in range(3):
    for i in range(3):
        for j in range(3):
            for ell in range(3):
                a, b = sorted(
                    (
                        voigt_map[tuple(sorted((k, j)))],
                        voigt_map[tuple(sorted((i, ell)))],
                    )
                )

                K[k, i] += C[C_map[(a, b)]] * xi[j] * xi[ell]

print("\n%%%%%%%%%%%%%%%%%%%%\n")
for i in range(3):
    for j in range(i + 1):
        print(f"K^{0}_{{{i}{j}}} &= ", latex(K[i, j]).replace("{C}", "C^{0}"), "\\\\")

# TODO: generate Jax code for this

mu0, lmbda0 = symbols("mu^0 lambda^0")

print("\n%%%%%%%%%%%%%%%%%%%%\n")
for i in range(3):
    for j in range(i + 1):
        v = K[i, j]
        for k in range(3):
            v = v.subs(C[k], 2 * mu0 + lmbda0)
        for k in range(3, 6):
            v = v.subs(C[k], mu0)
        for k in range(6, 9):
            v = v.subs(C[k], lmbda0)
        for k in range(9, 21):
            v = v.subs(C[k], 0)
        print(f"K^{0}_{{{i}{j}}} &= ", latex(simplify(v)), "\\\\")

assert False

N00, N01, N02, N11, N12, N22 = symbols(
    "N^{0}_{00} N^{0}_{01} N^{0}_{02} N^{0}_{11} N^{0}_{12} N^{0}_{22}"
)

N = Array([[N00, N01, N02], [N01, N11, N12], [N02, N12, N22]])
N_xi_xi = ArrayTensorProduct(N, xi, xi).as_explicit()
Gamma = Rational(1, 4) * (
    permutedims(N_xi_xi, (3, 0, 1, 2))
    + permutedims(N_xi_xi, (0, 3, 1, 2))
    + permutedims(N_xi_xi, (3, 0, 2, 1))
    + permutedims(N_xi_xi, (0, 3, 2, 1))
)
print("==================================")

for a in range(6):
    for b in range(a + 1):
        k, ell = voigt_indices[a]
        i, j = voigt_indices[b]
        print(
            f"\\widehat{{\\Gamma}}^{(0)}_{{{a},{b}}} &= ",
            latex(Gamma[k, ell, i, j]),
            "\\\\",
        )

# TODO: generate Jax code for this
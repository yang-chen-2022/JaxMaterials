from sympy import *
from sympy.tensor.array.expressions import ArrayTensorProduct, ArraySymbol

# Fourier modes
xi = Array(
    symbols(" ".join([f"\\mathring{{\\widetilde{{\\xi}}}}_{j}" for j in range(3)]))
)
# Acoustic 3x3 matrix
K = zeros(3, 3)

# 21 independent components of the 3x3x3x3 elasticity tensor
C = ArraySymbol("C", (21,))

# Voigt-indices
#
#  0 3 4
#    1 5
#      2
voigt_indices = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))

# Indices used for numbering the 21 indendendent components of the elasticity tensor
# when written down in Voigt notation

#  0  6  7  9 10 11
#     1  8 12 13 14
#        2 15 16 17
#           3 18 19
#              4 20
#                 5

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

voigt_map = {k: j for j, k in enumerate(voigt_indices)}
C_map = {k: j for j, k in enumerate(C_indices)}

# Independent components of strain tensor in Voigt notation
epsilon = ArraySymbol("varepsilon", (6,))
# Independent components of stress tensor in Voigt notation
sigma = zeros(6)

#### Matrix - vector product sigma = C epsilon

print(r"% ---- sigma_{ij} = C_{ijkl} epsilon_{kl} ----")
for a, (i, j) in enumerate(voigt_indices):
    for k in range(3):
        for ell in range(3):
            b = voigt_map[tuple(sorted([k, ell]))]
            _a, _b = sorted((a, b))
            sigma[a] += C[C_map[(_a, _b)]] * epsilon[b]
    print(f"\\sigma_{{{a}}} &= ", latex(simplify(sigma[a])), r"\\")
print("")
# TODO: generate Jax code for this

# Components of the acoustic-matrix

print(r"% ---- acoustic matrix K^0 ----")
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

for i in range(3):
    for j in range(i + 1):
        print(f"K^{0}_{{{i}{j}}} &= ", latex(K[i, j]).replace("{C}", "C^{0}"), "\\\\")

print("")

# TODO: generate Jax code for this

# Sanity-check: components of the acoustic-matrix in isotropic material
mu0, lmbda0 = symbols("mu^0 lambda^0")
print(r"% ---- acoustic matrix K^0 (isotropic material) ----")
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

print("")

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
print(r"% ---- ----")

for a in range(6):
    for b in range(a, 6):
        k, ell = voigt_indices[a]
        i, j = voigt_indices[b]
        print(
            f"\\widehat{{\\Gamma}}^{(0)}_{{{a},{b}}} &= ",
            latex(Gamma[k, ell, i, j]),
            "\\\\",
        )

# TODO: generate Jax code for this

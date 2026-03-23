from sympy import *
from sympy.tensor.array.expressions import ArrayTensorProduct, ArraySymbol


xi = Array(
    symbols(" ".join([f"\\mathring{{\\widetilde{{\\xi}}}}_{j}" for j in range(3)]))
)
K = zeros(3, 3)

C = ArraySymbol("C", (6, 6))


voigt_indices = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
voigt_map = {k: j for j, k in enumerate(voigt_indices)}
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

                K[k, i] += C[a, b] * xi[j] * xi[ell]

for i in range(3):
    for j in range(i + 1):
        print(f"K^{0}_{{{i}{j}}} &= ", latex(K[i, j]).replace("{C}", "C^{0}"), "\\\\")
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

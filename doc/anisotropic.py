from sympy import *
from sympy.tensor.array.expressions import ArrayTensorProduct, ArraySymbol


def index_map(indices):
    return {k: j for j, k in enumerate(indices)}


def elasticity(voigt_indices, C_indices):
    s = r"\begin{equation}\begin{aligned}" + "\n"
    for idx, (a, b) in enumerate(C_indices):
        i, j = voigt_indices[a]
        k, ell = voigt_indices[b]
        s += f"C_{{{idx}}} &= C_{{{i}{j},{k}{ell}}}, "
        if idx % 3 == 2:
            s += r"\\" + "\n"
        else:
            s += " & "
    s += r"\end{aligned}\end{equation}" + "\n"
    return s


def stress_strain(voigt_indices, C_indices):
    """Matrix - vector product sigma = C epsilon"""
    voigt_map = index_map(voigt_indices)
    C_map = index_map(C_indices)

    # 21 independent components of the 3x3x3x3 elasticity tensor
    C = ArraySymbol("C", (21,))

    # Independent components of strain tensor in Voigt notation
    epsilon = ArraySymbol("varepsilon", (6,))
    # Independent components of stress tensor in Voigt notation
    sigma = zeros(6)
    s = r"\begin{equation}\begin{aligned}" + "\n"
    for a in range(6):
        for k in range(3):
            for ell in range(3):
                b = voigt_map[tuple(sorted([k, ell]))]
                _a, _b = sorted((a, b))
                sigma[a] += C[C_map[(_a, _b)]] * epsilon[b]
        s += f"\\sigma_{{{a}}} &= " + latex(simplify(sigma[a])) + r"\\" + "\n"
    s += r"\end{aligned}\end{equation}" + "\n"
    return s
    # TODO: generate Jax code for this


def acoustic_matrix(voigt_indices, C_indices, isotropic=False):
    """Components of acoustic matrix"""
    voigt_map = index_map(voigt_indices)
    C_map = index_map(C_indices)
    # Acoustic 3x3 matrix
    K = zeros(3, 3)
    # 21 independent components of the 3x3x3x3 elasticity tensor
    C0 = ArraySymbol("C^{0}", (21,))
    # Lame coefficients for isotropic material
    mu0, lmbda0 = symbols("mu^0 lambda^0")
    # Fourier modes
    xi = Array(
        symbols(" ".join([f"\\mathring{{\\widetilde{{\\xi}}}}_{j}" for j in range(3)]))
    )
    s = r"\begin{equation}\begin{aligned}" + "\n"
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
                    K[k, i] += C0[C_map[(a, b)]] * xi[j] * xi[ell]

    for i in range(3):
        for j in range(i + 1):
            v = K[i, j]
            if isotropic:
                for k in range(3):
                    v = v.subs(C0[k], 2 * mu0 + lmbda0)
                for k in range(3, 6):
                    v = v.subs(C0[k], mu0)
                for k in range(6, 9):
                    v = v.subs(C0[k], lmbda0)
                for k in range(9, 21):
                    v = v.subs(C0[k], 0)
            s += (
                f"K^{0}_{{{i}{j}}} &= "
                + latex(simplify(v)).replace("{C^{0}}", "C^{0}")
                + "\\\\"
                + "\n"
            )
    s += r"\end{aligned}\end{equation}" + "\n"
    return s
    # TODO: generate Jax code for this


def fourier_solve(voigt_indices):
    # Fourier modes
    xi = Array(
        symbols(" ".join([f"\\mathring{{\\widetilde{{\\xi}}}}_{j}" for j in range(3)]))
    )
    N00, N01, N02, N11, N12, N22 = symbols(
        "N^{0}_{00} N^{0}_{01} N^{0}_{02} N^{0}_{11} N^{0}_{12} N^{0}_{22}"
    )

    N = Array([[N00, N01, N02], [N01, N11, N12], [N02, N12, N22]])
    N_xi_xi = tensorproduct(N, xi, xi)
    Gamma = Rational(1, 4) * (
        permutedims(N_xi_xi, (3, 0, 1, 2))
        + permutedims(N_xi_xi, (0, 3, 1, 2))
        + permutedims(N_xi_xi, (3, 0, 2, 1))
        + permutedims(N_xi_xi, (0, 3, 2, 1))
    )
    s = r"\begin{equation}\begin{aligned}" + "\n"
    count = 0
    for a in range(6):
        for b in range(a, 6):
            k, ell = voigt_indices[a]
            i, j = voigt_indices[b]
            v = simplify(Gamma[k, ell, i, j])
            s += f"\\widehat{{\\Gamma}}^{{0}}_{{{a}{b}}} &= " + latex(v)
            if count % 2 == 0:
                s += r", & " + "\n"
            else:
                s += r",\\[1ex]" + "\n"
            count += 1
    s += r"\end{aligned}\end{equation}" + "\n"
    return s
    # TODO: generate Jax code for this


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
with open("anisotropic_elasticity.tex", "w", encoding="utf8") as f:
    print(r"% ---- elasticity tensor ----", file=f)
    print(elasticity(voigt_indices, C_indices), file=f)
with open("anisotropic_stress_strain.tex", "w", encoding="utf8") as f:
    print(r"% ---- sigma_{ij} = C_{ijkl} epsilon_{kl} ----", file=f)
    print(stress_strain(voigt_indices, C_indices), file=f)
with open("anisotropic_acoustic_matrix.tex", "w", encoding="utf8") as f:
    print(r"% ---- acoustic matrix K^0 ----", file=f)
    print(acoustic_matrix(voigt_indices, C_indices, isotropic=False), file=f)
with open("isotropic_acoustic_matrix.tex", "w", encoding="utf8") as f:
    print(r"% ---- acoustic matrix K^0 [isotropic material] ----", file=f)
    print(acoustic_matrix(voigt_indices, C_indices, isotropic=True), file=f)
with open("anisotropic_fourier_matrix.tex", "w", encoding="utf8") as f:
    print(r"% ---- Fourier solve ----", file=f)
    print(fourier_solve(voigt_indices), file=f)

import numpy as np
from TP2.otimo import (
    PenalidadeInterior,
    PenalidadeExterior,
    LagrangeanoAumentado,
    GradienteConjugado,
    SecaoAurea,
    Restrita,
)

P = 20.0
SQRT2 = np.sqrt(2.0)
X0 = np.array([1.0, 3.0])


def vetor(x):
    """Garante que x seja sempre um vetor 1D."""
    return np.asarray(x, dtype=float).flatten()


def objetivo(x):
    x = vetor(x)
    x1, x2 = x
    return 2.0 * SQRT2 * x1 + x2


def denom(x):
    x = vetor(x)
    x1, x2 = x
    return SQRT2 * x1**2 + 2.0 * x1 * x2


# Todas as restrições no formato g(x) <= 0

def g1(x):
    x = vetor(x)
    x1, x2 = x
    return P * (x2 + SQRT2 * x1) / denom(x) - 20.0


def g2(x):
    x = vetor(x)
    x1, x2 = x
    return P / (x1 + SQRT2 * x2) - 20.0


def g3(x):
    x = vetor(x)
    x1, x2 = x
    return -P * x2 / denom(x) + 5.0


def g_x1_min(x):
    x = vetor(x)
    return 0.1 - x[0]


def g_x1_max(x):
    x = vetor(x)
    return x[0] - 5.0


def g_x2_min(x):
    x = vetor(x)
    return 0.1 - x[1]


def g_x2_max(x):
    x = vetor(x)
    return x[1] - 5.0


RESTRICOES = [
    g1,
    g2,
    g3,
    g_x1_min,
    g_x1_max,
    g_x2_min,
    g_x2_max,
]

TIPOS = np.array(["<", "<", "<", "<", "<", "<", "<"])


def valores_restricoes(x):
    return np.array([g(x) for g in RESTRICOES])


def relata(nome, solucao):
    x = vetor(solucao.x)
    gs = valores_restricoes(x)
    violacoes = Restrita().calcula_violacoes(x, RESTRICOES, TIPOS)

    print(f"\n===== {nome} =====")
    print(f"x* = {x}")
    print(f"f(x*) = {objetivo(x):.8f}")
    print(f"iterações = {solucao.iter}")
    print(f"avaliações = {solucao.aval}")
    print(f"violação máxima = {max(violacoes):.3e}")
    print(f"g(x*) = {gs}")
    print(f"critério de parada = {solucao.criterio_parada}")

    return {
        "Método": nome,
        "x1": x[0],
        "x2": x[1],
        "f(x*)": objetivo(x),
        "Iterações": solucao.iter,
        "Avaliações": solucao.aval,
        "Violação máx.": max(violacoes),
    }


def resolver_questao1(disp=True):
    busca_1d = SecaoAurea(precisao=1e-6, maxaval=1000)

    irrestrito = GradienteConjugado(
        busca_1d,
        precisao=1e-6,
        maxit=10000,
        maxaval=50000,
    )

    resultados = []

    # 1) Penalidade Interior
    interior = PenalidadeInterior(precisao=1e-6, maxaval=50000)

    sol_interior = interior.resolva(
        objetivo,
        X0,
        RESTRICOES,
        TIPOS,
        irrestrito,
        penalidade=100.0,
        desaceleracao=0.5,
        disp=disp,
    )

    resultados.append(relata("Penalidade Interior", sol_interior))

    # 2) Penalidade Exterior
    exterior = PenalidadeExterior(precisao=1e-6)

    sol_exterior = exterior.resolva(
        objetivo,
        X0,
        RESTRICOES,
        TIPOS,
        irrestrito,
        penalidade=1.0,
        aceleracao=2.0,
        disp=disp,
    )

    resultados.append(relata("Penalidade Exterior", sol_exterior))

    # 3) Lagrangeano Aumentado
    lagrangeano = LagrangeanoAumentado(precisao=1e-6)

    sol_lagrangeano = lagrangeano.resolva(
        objetivo,
        X0,
        RESTRICOES,
        TIPOS,
        irrestrito,
        penalidade=1.0,
        aceleracao=1.2,
        disp=disp,
    )

    resultados.append(relata("Lagrangeano Aumentado", sol_lagrangeano))

    return sol_interior, sol_exterior, sol_lagrangeano, resultados


def imprimir_comparacao(resultados):
    print("\n===== COMPARAÇÃO FINAL =====")
    print(
        f"{'Método':<25} "
        f"{'x1':>12} "
        f"{'x2':>12} "
        f"{'f(x*)':>12} "
        f"{'Iter':>8} "
        f"{'Aval':>8} "
        f"{'Violação':>12}"
    )

    for r in resultados:
        print(
            f"{r['Método']:<25} "
            f"{r['x1']:>12.6f} "
            f"{r['x2']:>12.6f} "
            f"{r['f(x*)']:>12.6f} "
            f"{r['Iterações']:>8} "
            f"{r['Avaliações']:>8} "
            f"{r['Violação máx.']:>12.3e}"
        )


def main():
    sol_interior, sol_exterior, sol_lagrangeano, resultados = resolver_questao1(disp=True)
    imprimir_comparacao(resultados)


if __name__ == "__main__":
    main()
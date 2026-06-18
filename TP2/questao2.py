"""
Questão 2 - Despacho econômico de 3 unidades geradoras (otimização restrita)
============================================================================

min  f(P1, P2, P3) = C1(P1) + C2(P2) + C3(P3)
        C1 = 0.15*P1^2 + 38*P1 + 756
        C2 = 0.10*P2^2 + 46*P2 + 451
        C3 = 0.25*P3^2 + 40*P3 + 1049
s.a. h:  P1 + P2 + P3 = 850 + PL          (balanço de potência, igualdade)
        PL = sum_i sum_j Pi * Bij * Pj    (perda na transmissão)
     150 <= P1 <= 600
     100 <= P2 <= 400
      50 <= P3 <= 200

Métodos exigidos: Penalidade Exterior e Lagrangeano Aumentado.
(Penalidade Interior não é usada: a biblioteca não suporta restrições de
 igualdade nesse método.)

Convenção da biblioteca (otimo.py):
    '<' -> g(x) <= 0 ; '>' -> g(x) >= 0 ; '=' -> g(x) = 0
"""

import numpy as np

from otimo import (
    PenalidadeExterior,
    LagrangeanoAumentado,
    GradienteConjugado,
    SecaoAurea,
    Restrita,
)

# ---------------------------------------------------------------------------
# Parâmetros do problema
# ---------------------------------------------------------------------------
DEMANDA = 850.0

# Coeficientes das curvas de custo: [a, b, c] -> a*P^2 + b*P + c
CUSTO = np.array([
    [0.15, 38.0, 756.0],
    [0.10, 46.0, 451.0],
    [0.25, 40.0, 1049.0],
])

# Matriz de coeficientes de perda B (simétrica)
B = np.array([
    [0.000049, 0.000014, 0.000015],
    [0.000014, 0.000045, 0.000016],
    [0.000015, 0.000016, 0.000039],
])

# Limites das variáveis (P1, P2, P3)
LIM_INF = np.array([150.0, 100.0, 50.0])
LIM_SUP = np.array([600.0, 400.0, 200.0])

# TODO: escolher um ponto inicial viável/razoável (ex.: soma ~ demanda)
X0 = np.array([400.0, 300.0, 150.0])


# ---------------------------------------------------------------------------
# Função objetivo
# ---------------------------------------------------------------------------
def custo_unidade(i, Pi):
    a, b, c = CUSTO[i]
    return a * Pi ** 2 + b * Pi + c


def objetivo(P):
    return sum(custo_unidade(i, P[i]) for i in range(3))


def perda_transmissao(P):
    """PL = P^T B P."""
    P = np.asarray(P, dtype=float).flatten()
    return float(P @ B @ P)


# ---------------------------------------------------------------------------
# Restrições
# ---------------------------------------------------------------------------
def h_balanco(P):
    # P1 + P2 + P3 - (850 + PL) = 0
    return P[0] + P[1] + P[2] - (DEMANDA + perda_transmissao(P))


def _limite_min(i):
    return lambda P: LIM_INF[i] - P[i]   # Pi >= LIM_INF[i]


def _limite_max(i):
    return lambda P: P[i] - LIM_SUP[i]   # Pi <= LIM_SUP[i]


RESTRICOES = [h_balanco]
TIPOS = ['=']
for _i in range(3):
    RESTRICOES += [_limite_min(_i), _limite_max(_i)]
    TIPOS += ['<', '<']
TIPOS = np.array(TIPOS)


# ---------------------------------------------------------------------------
# Utilitário de relatório
# ---------------------------------------------------------------------------
def relata(nome, solucao):
    violacoes = Restrita().calcula_violacoes(solucao.x, RESTRICOES, TIPOS)
    P = solucao.x
    print(f"\n===== {nome} =====")
    print(solucao)
    print(f"P = {P}")
    print(f"Soma gerada = {P.sum():.4f} MW | PL = {perda_transmissao(P):.4f} MW")
    print(f"Violação máxima das restrições: {max(violacoes):.3e}")


# ---------------------------------------------------------------------------
# Execução dos métodos
# ---------------------------------------------------------------------------
def main():
    # ---- Otimizador irrestrito interno ------------------------------------
    # TODO: testar outros métodos/precisões internas
    busca_1d = SecaoAurea(precisao=1e-6)
    irrestrito = GradienteConjugado(busca_1d, precisao=1e-6)

    # ---- 1) Penalidade Exterior -------------------------------------------
    # TODO: ajustar penalidade inicial e aceleracao
    exterior = PenalidadeExterior(precisao=1e-6)
    sol_exterior = exterior.resolva(
        objetivo, X0, RESTRICOES, TIPOS, irrestrito,
        penalidade=1.0, aceleracao=2.0, disp=True,
    )
    relata("Penalidade Exterior", sol_exterior)

    # ---- 2) Lagrangeano Aumentado -----------------------------------------
    # TODO: ajustar penalidade inicial e aceleracao
    lagrangeano = LagrangeanoAumentado(precisao=1e-6)
    sol_lagrangeano = lagrangeano.resolva(
        objetivo, X0, RESTRICOES, TIPOS, irrestrito,
        penalidade=1.0, aceleracao=1.2, disp=True,
    )
    relata("Lagrangeano Aumentado", sol_lagrangeano)

    # TODO: comparar os dois métodos (P*, custo, perdas, violação do balanço,
    #       nº de iterações/avaliações) em uma tabela para o relatório.


if __name__ == "__main__":
    main()

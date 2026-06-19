"""
Questão 2 — Dados e formulação do despacho econômico
====================================================

Módulo de formulação (T1 do plano): centraliza os dados do problema, a função
objetivo, as perdas na transmissão, as restrições (1 igualdade de balanço + 6
de caixa) e os helpers analíticos usados na validação e na discussão.

Problema:

    min  f(P1, P2, P3) = C1(P1) + C2(P2) + C3(P3)
            C1 = 0.15*P1^2 + 38*P1 + 756
            C2 = 0.10*P2^2 + 46*P2 + 451
            C3 = 0.25*P3^2 + 40*P3 + 1049
    s.a. h:  P1 + P2 + P3 = 850 + PL          (balanço de potência, igualdade)
            PL = P^T B P                      (perda na transmissão)
         150 <= P1 <= 600
         100 <= P2 <= 400
          50 <= P3 <= 200

Convenção das restrições (otimo.py):
    '<' -> g(x) <= 0 ; '>' -> g(x) >= 0 ; '=' -> g(x) = 0
Mantém-se '<' para ambos os limites de caixa (consistente com o esqueleto
original e já validado).
"""

import numpy as np

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

# Ponto inicial. Soma 850 MW (demanda), levemente infactível na igualdade com
# perdas (ΣP = 850 + PL ≈ 880) — aceitável para Penalidade Exterior.
X0 = np.array([400.0, 300.0, 150.0])


# ---------------------------------------------------------------------------
# Função objetivo e perdas
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
# Helpers analíticos (validação e discussão)
# ---------------------------------------------------------------------------
def custo_incremental(P):
    """Custo incremental dCi/dPi = 2*a_i*P_i + b_i (vetor de 3)."""
    P = np.asarray(P, dtype=float).flatten()
    a = CUSTO[:, 0]
    b = CUSTO[:, 1]
    return 2.0 * a * P + b


def grad_perdas(P):
    """Gradiente das perdas ∂PL/∂P = 2*B*P (vetor de 3); B é simétrica."""
    P = np.asarray(P, dtype=float).flatten()
    return 2.0 * (B @ P)


def penalty_factors(P):
    """Penalty factors L_i = 1 / (1 - ∂PL/∂P_i) (vetor de 3)."""
    return 1.0 / (1.0 - grad_perdas(P))


# ---------------------------------------------------------------------------
# Sanidade
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("===== Sanidade da formulação (X0) =====")
    print(f"X0          = {X0}")
    print(f"ΣX0         = {X0.sum():.4f} MW")
    print(f"f(X0)       = {objetivo(X0):.4f} R$/h")
    print(f"PL(X0)      = {perda_transmissao(X0):.4f} MW")
    print(f"h(X0)       = {h_balanco(X0):.4f}  (= ΣP - 850 - PL)")
    print(f"dCi/dPi(X0) = {custo_incremental(X0)}")
    print(f"∂PL/∂P(X0)  = {grad_perdas(X0)}")
    print(f"L_i(X0)     = {penalty_factors(X0)}")

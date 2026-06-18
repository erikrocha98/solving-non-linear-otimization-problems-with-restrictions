"""
Questão 1 - Treliça de três barras (otimização restrita)
=========================================================

min  f(x1, x2) = 2*sqrt(2)*x1 + x2
s.a. g1: P*(x2 + sqrt(2)*x1) / (sqrt(2)*x1^2 + 2*x1*x2) <= 20   (estresse barra 1)
     g2: P / (x1 + sqrt(2)*x2)                          <= 20   (estresse barra 2)
     g3: -P*x2 / (sqrt(2)*x1^2 + 2*x1*x2)               <= -5   (estresse min barra 3)
     0.1 <= x1, x2 <= 5

P = 20, ponto inicial sugerido (x1, x2) = (1, 3).

Métodos exigidos: Penalidade Interior, Penalidade Exterior e Lagrangeano Aumentado.

Convenção da biblioteca (otimo.py):
    Cada restrição é uma função g(x) e seu tipo indica a forma:
        '<'  ->  g(x) <= 0
        '>'  ->  g(x) >= 0
        '='  ->  g(x)  = 0
    Portanto toda restrição abaixo é escrita na forma  g(x) <= 0  (tipo '<').
"""

import numpy as np

from otimo import (
    PenalidadeInterior,
    PenalidadeExterior,
    LagrangeanoAumentado,
    GradienteConjugado,
    SecaoAurea,
    Restrita,
)

# ---------------------------------------------------------------------------
# Parâmetros do problema
# ---------------------------------------------------------------------------
P = 20.0
SQRT2 = np.sqrt(2.0)

X0 = np.array([1.0, 3.0])   # ponto inicial sugerido no enunciado


# ---------------------------------------------------------------------------
# Função objetivo
# ---------------------------------------------------------------------------
def objetivo(x):
    """Peso da treliça: f(x1, x2) = 2*sqrt(2)*x1 + x2."""
    x1, x2 = x[0], x[1]
    return 2.0 * SQRT2 * x1 + x2


# ---------------------------------------------------------------------------
# Restrições (todas na forma g(x) <= 0)
# ---------------------------------------------------------------------------
def _denom(x):
    """Denominador comum das tensões: sqrt(2)*x1^2 + 2*x1*x2."""
    x1, x2 = x[0], x[1]
    return SQRT2 * x1 ** 2 + 2.0 * x1 * x2


def g_estresse_barra1(x):
    # P*(x2 + sqrt(2)*x1) / denom <= 20
    x1, x2 = x[0], x[1]
    return P * (x2 + SQRT2 * x1) / _denom(x) - 20.0


def g_estresse_barra2(x):
    # P / (x1 + sqrt(2)*x2) <= 20
    x1, x2 = x[0], x[1]
    return P / (x1 + SQRT2 * x2) - 20.0


def g_estresse_barra3(x):
    # -P*x2 / denom <= -5   <=>   -P*x2/denom + 5 <= 0
    x1, x2 = x[0], x[1]
    return -P * x2 / _denom(x) + 5.0


def g_x1_min(x):
    return 0.1 - x[0]          # x1 >= 0.1


def g_x1_max(x):
    return x[0] - 5.0          # x1 <= 5


def g_x2_min(x):
    return 0.1 - x[1]          # x2 >= 0.1


def g_x2_max(x):
    return x[1] - 5.0          # x2 <= 5


RESTRICOES = [
    g_estresse_barra1,
    g_estresse_barra2,
    g_estresse_barra3,
    g_x1_min,
    g_x1_max,
    g_x2_min,
    g_x2_max,
]
TIPOS = np.array(['<', '<', '<', '<', '<', '<', '<'])


# ---------------------------------------------------------------------------
# Utilitário de relatório
# ---------------------------------------------------------------------------
def relata(nome, solucao):
    """Imprime um resumo padronizado da solução e das violações."""
    violacoes = Restrita().calcula_violacoes(solucao.x, RESTRICOES, TIPOS)
    print(f"\n===== {nome} =====")
    print(solucao)
    print(f"Violação máxima das restrições: {max(violacoes):.3e}")
    # TODO: salvar/plotar trajetória com solucao.resultados(...) se desejado


# ---------------------------------------------------------------------------
# Execução dos três métodos
# ---------------------------------------------------------------------------
def main():
    # ---- Otimizador irrestrito interno (usado por todos os métodos) --------
    # TODO: experimentar outros métodos/precisões (BFGS, GradienteConjugado...)
    busca_1d = SecaoAurea(precisao=1e-6)
    irrestrito = GradienteConjugado(busca_1d, precisao=1e-6)

    # ---- 1) Penalidade Interior -------------------------------------------
    # TODO: ajustar penalidade inicial e desaceleracao
    interior = PenalidadeInterior(precisao=1e-6)
    sol_interior = interior.resolva(
        objetivo, X0, RESTRICOES, TIPOS, irrestrito,
        penalidade=1.0, desaceleracao=0.5, disp=True,
    )
    relata("Penalidade Interior", sol_interior)

    # ---- 2) Penalidade Exterior -------------------------------------------
    # TODO: ajustar penalidade inicial e aceleracao
    exterior = PenalidadeExterior(precisao=1e-6)
    sol_exterior = exterior.resolva(
        objetivo, X0, RESTRICOES, TIPOS, irrestrito,
        penalidade=1.0, aceleracao=2.0, disp=True,
    )
    relata("Penalidade Exterior", sol_exterior)

    # ---- 3) Lagrangeano Aumentado -----------------------------------------
    # TODO: ajustar penalidade inicial e aceleracao
    lagrangeano = LagrangeanoAumentado(precisao=1e-6)
    sol_lagrangeano = lagrangeano.resolva(
        objetivo, X0, RESTRICOES, TIPOS, irrestrito,
        penalidade=1.0, aceleracao=1.2, disp=True,
    )
    relata("Lagrangeano Aumentado", sol_lagrangeano)

    # TODO: comparar os três métodos (x*, f(x*), nº iterações, nº avaliações,
    #       violações) em uma tabela para o relatório.


if __name__ == "__main__":
    main()

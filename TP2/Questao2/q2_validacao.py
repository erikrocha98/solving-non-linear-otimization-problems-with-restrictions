"""
Questão 2 — Validação independente por custo incremental (T4)
=============================================================

Referência analítica do despacho econômico, em numpy puro (sem scipy), pela
equação de coordenação: no ótimo, para toda unidade não saturada,

    dCi/dPi = λ · (1 − ∂PL/∂Pi)            (custo incremental corrigido)

com  dCi/dPi = 2·a_i·P_i + b_i  e  ∂PL/∂Pi = 2·Σ_j B_ij·P_j.

Resolve por **lambda-iteration**:
  1. Para λ fixo, ponto fixo em P:
        P_i = [λ·(1 − 2·Σ_{j≠i} B_ij·P_j) − b_i] / (2·a_i + 2·λ·B_ii),
     iterando até convergir e aplicando clamp aos limites de caixa.
  2. Ajusta λ por bisseção até o balanço ΣP − PL − 850 = 0.

`valida()` compara P* e f* analíticos com os do otimizador e alerta se a
divergência relativa passar de ~1%.
"""

import numpy as np

from q2_problema import (
    CUSTO, B, LIM_INF, LIM_SUP, DEMANDA,
    objetivo, perda_transmissao,
)
from q2_solvers import resolve_lagrangeano


_A = CUSTO[:, 0]
_B = CUSTO[:, 1]


def despacho_para_lambda(lam, tol=1e-12, maxit=10000):
    """Ponto fixo em P para um λ fixo, com clamp aos limites de caixa."""
    P = 0.5 * (LIM_INF + LIM_SUP)
    for _ in range(maxit):
        P_novo = P.copy()
        for i in range(3):
            soma_cruzada = np.dot(B[i], P) - B[i, i] * P[i]  # Σ_{j≠i} B_ij P_j
            Pi = (lam * (1.0 - 2.0 * soma_cruzada) - _B[i]) / (2.0 * _A[i] + 2.0 * lam * B[i, i])
            P_novo[i] = np.clip(Pi, LIM_INF[i], LIM_SUP[i])
        if np.linalg.norm(P_novo - P) < tol:
            P = P_novo
            break
        P = P_novo
    return P


def _residuo_balanco(lam):
    """g(λ) = ΣP(λ) − PL(P(λ)) − DEMANDA. Crescente em λ."""
    P = despacho_para_lambda(lam)
    return P.sum() - perda_transmissao(P) - DEMANDA


def resolve_lambda(lam_lo=1.0, lam_hi=1000.0, tol=1e-8, maxit=200):
    """Bisseção em λ até zerar o resíduo do balanço."""
    g_lo = _residuo_balanco(lam_lo)
    g_hi = _residuo_balanco(lam_hi)
    if g_lo * g_hi > 0:
        raise ValueError(
            f"λ não está bracketado em [{lam_lo}, {lam_hi}]: "
            f"g_lo={g_lo:.3e}, g_hi={g_hi:.3e}"
        )
    for _ in range(maxit):
        lam = 0.5 * (lam_lo + lam_hi)
        g = _residuo_balanco(lam)
        if abs(g) < tol:
            break
        if g_lo * g < 0:
            lam_hi = lam
        else:
            lam_lo, g_lo = lam, g
    P = despacho_para_lambda(lam)
    return lam, P


def referencia_analitica():
    """Devolve dict com λ, P*, f* e PL da solução analítica."""
    lam, P = resolve_lambda()
    return {
        "lambda": lam,
        "P": P,
        "f": objetivo(P),
        "PL": perda_transmissao(P),
        "balanco": P.sum() - perda_transmissao(P) - DEMANDA,
    }


def valida(solucao=None, tol_relativa=0.01):
    """Compara o otimizador com a referência analítica.

    Se `solucao` não for dada, resolve via Lagrangeano Aumentado (config final).
    Alerta se a divergência relativa em P* ou f* passar de `tol_relativa`.
    """
    ref = referencia_analitica()

    if solucao is None:
        solucao = resolve_lagrangeano(penalidade=100.0, aceleracao=1.2)
    P_opt = np.asarray(solucao.x, dtype=float).flatten()
    f_opt = float(solucao.fx)

    dif_P = np.abs(P_opt - ref["P"]) / np.maximum(np.abs(ref["P"]), 1e-12)
    dif_f = abs(f_opt - ref["f"]) / max(abs(ref["f"]), 1e-12)
    ok = bool(np.all(dif_P <= tol_relativa) and dif_f <= tol_relativa)

    print("===== Validação por custo incremental (lambda-iteration) =====")
    print(f"λ*               = {ref['lambda']:.6f}")
    print(f"P* analítico     = {ref['P']}")
    print(f"P* otimizador    = {P_opt}")
    print(f"diferença rel. P = {dif_P}")
    print(f"f* analítico     = {ref['f']:.4f} R$/h")
    print(f"f* otimizador    = {f_opt:.4f} R$/h")
    print(f"diferença rel. f = {dif_f:.3e}")
    print(f"balanço analítico (ΣP-PL-850) = {ref['balanco']:.3e}")
    print(f"custo incremental corrigido dCi/dPi/(1-∂PL/∂Pi) = "
          f"{(2*_A*ref['P'] + _B) / (1 - 2*(B @ ref['P']))}")
    if ok:
        print("\n✔ Otimizador bate com a referência analítica (≤ 1%).")
    else:
        print("\n⚠ DIVERGÊNCIA > 1% — verificar sinal/PL/formulação.")
    return {"ref": ref, "ok": ok, "dif_P": dif_P, "dif_f": dif_f}


if __name__ == "__main__":
    valida()

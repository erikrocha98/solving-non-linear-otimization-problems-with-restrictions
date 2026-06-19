"""
Questão 2 — Estudo de sensibilidade da penalidade (T3, Seção 4)
===============================================================

Varre a penalidade inicial de cada método e registra como a violação da
igualdade, o custo e o esforço computacional respondem. Salva os resultados em
CSV e imprime uma tabela, documentando a configuração final escolhida.

Achado-chave (verificado em runtime): no Penalidade Exterior a aceleração alta
(2.0) faz a penalidade explodir e o solver interno diverge no quártico
`f + rho*h^2` (h é quadrático via PL). Por isso a varredura do Exterior usa
aceleração moderada (1.2) e ainda assim detecta/anota divergências.
"""

import csv
import os

import numpy as np

from q2_solvers import resolve_exterior, resolve_lagrangeano
from q2_problema import perda_transmissao, h_balanco, RESTRICOES, TIPOS

from otimo import Restrita

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "q2_sensibilidade.csv")

# Limite acima do qual consideramos que a otimização divergiu.
_LIMITE_DIVERGENCIA = 1e5


def _avalia(metodo, penalidade, aceleracao, solver):
    """Roda um solver e devolve uma linha de resultado robusta a divergência."""
    sol = solver(penalidade=penalidade, aceleracao=aceleracao, disp=False)
    P = np.asarray(sol.x, dtype=float).flatten()
    divergiu = bool(np.any(~np.isfinite(P)) or np.max(np.abs(P)) > _LIMITE_DIVERGENCIA)

    viol = Restrita().calcula_violacoes(P, RESTRICOES, TIPOS)
    return {
        "metodo": metodo,
        "penalidade": penalidade,
        "aceleracao": aceleracao,
        "P1": P[0], "P2": P[1], "P3": P[2],
        "custo": float(sol.fx),
        "PL": perda_transmissao(P) if not divergiu else float("nan"),
        "viol_balanco": abs(h_balanco(P)),
        "viol_max": float(max(viol)),
        "iter": int(sol.iter),
        "aval": int(sol.aval),
        "divergiu": divergiu,
    }


def varre_exterior(penalidades=(1.0, 10.0, 1e2, 1e3, 1e4), aceleracao=1.2):
    return [
        _avalia("Exterior", p, aceleracao, resolve_exterior)
        for p in penalidades
    ]


def varre_lagrangeano(penalidades=(1.0, 10.0, 1e2), aceleracao=1.2):
    return [
        _avalia("Lagrangeano", p, aceleracao, resolve_lagrangeano)
        for p in penalidades
    ]


# ---------------------------------------------------------------------------
# Saída
# ---------------------------------------------------------------------------
_CAMPOS = ["metodo", "penalidade", "aceleracao", "P1", "P2", "P3", "custo",
           "PL", "viol_balanco", "viol_max", "iter", "aval", "divergiu"]


def salva_csv(linhas, caminho=CSV_PATH):
    with open(caminho, "w", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=_CAMPOS)
        escritor.writeheader()
        for linha in linhas:
            escritor.writerow(linha)
    return caminho


def imprime_tabela(linhas):
    cab = (f"{'metodo':<12}{'pen0':>8}{'acel':>6}{'custo':>14}"
           f"{'|h|':>12}{'viol_max':>12}{'iter':>6}{'aval':>8}  obs")
    print(cab)
    print("-" * len(cab))
    for r in linhas:
        obs = "DIVERGIU" if r["divergiu"] else ""
        print(f"{r['metodo']:<12}{r['penalidade']:>8.0f}{r['aceleracao']:>6.1f}"
              f"{r['custo']:>14.2f}{r['viol_balanco']:>12.2e}"
              f"{r['viol_max']:>12.2e}{r['iter']:>6}{r['aval']:>8}  {obs}")


def _melhor(linhas):
    """Config final: menor violação entre as que convergiram (desempate: aval)."""
    validas = [r for r in linhas if not r["divergiu"]]
    if not validas:
        return None
    return min(validas, key=lambda r: (r["viol_balanco"], r["aval"]))


def varre_tudo(salvar=True):
    linhas = varre_exterior() + varre_lagrangeano()
    if salvar:
        salva_csv(linhas)
    return linhas


if __name__ == "__main__":
    linhas = varre_tudo(salvar=True)

    print("\n========== Sensibilidade da penalidade ==========\n")
    imprime_tabela(linhas)
    print(f"\nCSV salvo em: {CSV_PATH}")

    ext = [r for r in linhas if r["metodo"] == "Exterior"]
    lag = [r for r in linhas if r["metodo"] == "Lagrangeano"]

    print("\n----- Configuração final documentada -----")
    melhor_ext = _melhor(ext)
    melhor_lag = _melhor(lag)
    if melhor_ext:
        print(f"Exterior   : penalidade inicial = {melhor_ext['penalidade']:.0f}, "
              f"aceleracao = {melhor_ext['aceleracao']:.1f}  "
              f"(|h| = {melhor_ext['viol_balanco']:.2e}, aval = {melhor_ext['aval']})")
    if melhor_lag:
        print(f"Lagrangeano: penalidade inicial = {melhor_lag['penalidade']:.0f}, "
              f"aceleracao = {melhor_lag['aceleracao']:.1f}  "
              f"(|h| = {melhor_lag['viol_balanco']:.2e}, aval = {melhor_lag['aval']})")

    print("\nDiscussão: o Lagrangeano Aumentado atinge violação baixa já com "
          "penalidade inicial pequena (multiplicadores corrigem o balanço), "
          "enquanto o Exterior depende de penalidade maior — e aceleração alta "
          "o leva à divergência no termo quártico das perdas.")

"""
Questão 2 — Execução dos métodos e coleta de histórico (T2)
===========================================================

Wrappers dos dois métodos exigidos (Penalidade Exterior e Lagrangeano
Aumentado), construção do otimizador irrestrito interno, extração da
trajetória externa (`xhist`/`fxhist`) e resumo comparável por método.
"""

import os
import sys

import numpy as np

# otimo.py é compartilhado entre as duas questões e vive na raiz de TP2/.
# Como este módulo está em TP2/Questao2/, garantimos que a raiz esteja no path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from otimo import (
    PenalidadeExterior,
    LagrangeanoAumentado,
    GradienteConjugado,
    BFGS,
    SecaoAurea,
    Restrita,
)

from q2_problema import (
    objetivo,
    perda_transmissao,
    h_balanco,
    custo_incremental,
    penalty_factors,
    RESTRICOES,
    TIPOS,
    DEMANDA,
    X0,
)


# ---------------------------------------------------------------------------
# Otimizador irrestrito interno
# ---------------------------------------------------------------------------
def build_irrestrito(metodo="GC"):
    """Constrói o método de otimização irrestrita usado internamente.

    metodo="GC"  -> Gradiente Conjugado (padrão)
    metodo="BFGS" -> BFGS (para comparar robustez)
    Ambos usam Seção Áurea como busca unidimensional.
    """
    busca = SecaoAurea(precisao=1e-6)
    if metodo == "GC":
        return GradienteConjugado(busca, precisao=1e-6)
    if metodo == "BFGS":
        return BFGS(busca, precisao=1e-6)
    raise ValueError(f"Método irrestrito desconhecido: {metodo!r}")


# ---------------------------------------------------------------------------
# Wrappers dos métodos restritos
# ---------------------------------------------------------------------------
def resolve_exterior(penalidade, aceleracao=2.0, x0=X0, irrestrito=None,
                     disp=False):
    """Resolve via Penalidade Exterior."""
    if irrestrito is None:
        irrestrito = build_irrestrito("GC")
    return PenalidadeExterior(precisao=1e-6).resolva(
        objetivo, x0, RESTRICOES, TIPOS, irrestrito,
        penalidade=penalidade, aceleracao=aceleracao, disp=disp,
    )


def resolve_lagrangeano(penalidade, aceleracao=1.2, x0=X0, irrestrito=None,
                        disp=False):
    """Resolve via Lagrangeano Aumentado."""
    if irrestrito is None:
        irrestrito = build_irrestrito("GC")
    return LagrangeanoAumentado(precisao=1e-6).resolva(
        objetivo, x0, RESTRICOES, TIPOS, irrestrito,
        penalidade=penalidade, aceleracao=aceleracao, disp=disp,
    )


# ---------------------------------------------------------------------------
# Coleta de histórico
# ---------------------------------------------------------------------------
def _normaliza_xhist(xhist):
    """Devolve xhist como matriz (n, 3), independente da forma original.

    O método appenda colunas (3,1); np.array vira (n,3,1). Normaliza para
    (n, 3) iterando por linha e achatando cada ponto.
    """
    xhist = np.asarray(xhist, dtype=float)
    if xhist.ndim == 1:
        xhist = np.atleast_2d(xhist)
    return np.array([np.asarray(p, dtype=float).flatten() for p in xhist])


def coleta_historico(solucao):
    """Extrai a trajetória externa em um dict para gráficos/análise.

    Retorna:
        P_hist       (n, 3)  potências por iteração
        f_hist       (n,)    valor objetivo por iteração
        viol_balanco (n,)    |h_balanco(P_k)| por iteração
        viol_max     (n,)    máxima violação de todas as restrições por iteração
    """
    P_hist = _normaliza_xhist(solucao.xhist)
    f_hist = np.asarray(solucao.fxhist, dtype=float).flatten()

    restrita = Restrita()
    viol_balanco = np.array([abs(h_balanco(P)) for P in P_hist])
    viol_max = np.array([
        max(restrita.calcula_violacoes(P, RESTRICOES, TIPOS)) for P in P_hist
    ])

    return {
        "P_hist": P_hist,
        "f_hist": f_hist,
        "viol_balanco": viol_balanco,
        "viol_max": viol_max,
    }


# ---------------------------------------------------------------------------
# Critério de parada (derivado)
# ---------------------------------------------------------------------------
def status_convergencia(solucao, maxit=10000, maxaval=10000, precisao=1e-6):
    """Reconstrói o critério de parada a partir dos dados de `solucao`.

    O `Solution.__init__` de otimo.py ignora o argumento `criterio_parada`
    (faz `self.criterio_parada = None` fixo), então o campo é sempre None.
    Aqui replicamos a lógica de decisão dos métodos restritos usando
    iter/aval/xhist, com os mesmos limiares usados na execução.
    """
    if solucao.iter >= maxit:
        return "máximo de iterações atingido"
    if solucao.aval >= maxaval:
        return "máximo de avaliações atingido"
    # otimo.py compara o ponto atual (xhist[-1]) com xhist[-5] (5 atrás).
    xhist = _normaliza_xhist(solucao.xhist)
    if len(xhist) >= 5 and np.linalg.norm(xhist[-1] - xhist[-5]) < precisao:
        return f"convergência (Δx < {precisao:g})"
    return "critério de convergência atingido"


# ---------------------------------------------------------------------------
# Resumo por método (substitui `relata`)
# ---------------------------------------------------------------------------
def resumo(nome, solucao, maxit=10000, maxaval=10000, precisao=1e-6):
    """Imprime um resumo comparável da solução de um método."""
    P = np.asarray(solucao.x, dtype=float).flatten()
    viol = Restrita().calcula_violacoes(P, RESTRICOES, TIPOS)
    PL = perda_transmissao(P)
    status = status_convergencia(solucao, maxit, maxaval, precisao)

    print(f"\n===== {nome} =====")
    print(f"P*               = {P}")
    print(f"f*               = {solucao.fx:.4f} R$/h")
    print(f"ΣP               = {P.sum():.4f} MW")
    print(f"PL               = {PL:.4f} MW")
    print(f"ΣP - PL          = {P.sum() - PL:.4f} MW (alvo = {DEMANDA:.1f})")
    print(f"iterações        = {solucao.iter}")
    print(f"avaliações       = {solucao.aval}")
    print(f"critério parada  = {status}")
    print(f"|h_balanco|      = {abs(h_balanco(P)):.3e}")
    print(f"violação máxima  = {max(viol):.3e}")
    print(f"custo incremental= {custo_incremental(P)}")
    print(f"penalty factors  = {penalty_factors(P)}")
    return {
        "nome": nome,
        "P": P,
        "f": float(solucao.fx),
        "soma": float(P.sum()),
        "PL": PL,
        "iter": int(solucao.iter),
        "aval": int(solucao.aval),
        "viol_max": float(max(viol)),
        "viol_balanco": float(abs(h_balanco(P))),
        "status": status,
    }


# ---------------------------------------------------------------------------
# Sanidade
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Obs.: aceleracao alta (ex.: 2.0) faz a penalidade explodir e o solver
    # interno diverge no quártico f + rho*h^2 (h é quadrático via PL).
    # A escolha da config final do Exterior é tarefa do estudo de sensibilidade.
    print("### Penalidade Exterior (penalidade=10, aceleracao=1.2) ###")
    sol_e = resolve_exterior(penalidade=10.0, aceleracao=1.2, disp=True)
    resumo("Penalidade Exterior", sol_e)
    hist_e = coleta_historico(sol_e)
    print(f"\nhistórico: {hist_e['P_hist'].shape[0]} iterações externas, "
          f"viol_balanco final = {hist_e['viol_balanco'][-1]:.3e}")

    print("\n### Lagrangeano Aumentado (penalidade=10, aceleracao=1.2) ###")
    sol_l = resolve_lagrangeano(penalidade=10.0, aceleracao=1.2, disp=True)
    resumo("Lagrangeano Aumentado", sol_l)
    hist_l = coleta_historico(sol_l)
    print(f"\nhistórico: {hist_l['P_hist'].shape[0]} iterações externas, "
          f"viol_balanco final = {hist_l['viol_balanco'][-1]:.3e}")

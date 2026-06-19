"""
Questão 2 — Figuras para o relatório (T5)
=========================================

Gera as figuras em `figuras/` a partir dos históricos coletados nos solvers
(T2), da varredura de sensibilidade (T3) e da referência analítica (T4).
Usa o backend Agg (sem display) e salva PNGs.

Figuras:
  1. q2_convergencia_custo.png       — f(P_k) por iteração, os dois métodos
  2. q2_violacao_balanco.png         — |h(P_k)| em semilog (gráfico-chave)
  3. q2_despacho_final.png           — barras P1,P2,P3 por método + ref. analítica
  4. q2_evolucao_potencias.png       — P1,P2,P3 ao longo das iterações
  5. q2_sensibilidade_penalidade.png — viol final / nº avaliações vs penalidade
  6. q2_custos_incrementais.png      — dCi/dPi no ótimo (igualdade corrigida)
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from q2_problema import custo_incremental, grad_perdas

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figuras")

_COR = {"Exterior": "tab:blue", "Lagrangeano": "tab:red", "Analítico": "tab:green"}


def _garante_dir():
    os.makedirs(FIG_DIR, exist_ok=True)


def _salva(fig, nome):
    caminho = os.path.join(FIG_DIR, nome)
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return caminho


# ---------------------------------------------------------------------------
# Figuras individuais
# ---------------------------------------------------------------------------
def convergencia_custo(historicos):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for nome, h in historicos.items():
        ax.plot(h["f_hist"], "-o", ms=3, label=nome, color=_COR.get(nome))
    ax.set_xlabel("iteração externa")
    ax.set_ylabel("f(P) [R$/h]")
    ax.set_title("Convergência do custo")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _salva(fig, "q2_convergencia_custo.png")


def violacao_balanco(historicos):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for nome, h in historicos.items():
        viol = np.maximum(h["viol_balanco"], 1e-16)  # evita log(0)
        ax.semilogy(viol, "-o", ms=3, label=nome, color=_COR.get(nome))
    ax.set_xlabel("iteração externa")
    ax.set_ylabel("|h(P)| = |ΣP − PL − 850|  [MW]")
    ax.set_title("Violação do balanço de potência (semilog)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return _salva(fig, "q2_violacao_balanco.png")


def despacho_final(historicos, ref=None):
    nomes = list(historicos.keys())
    rotulos = ["P1", "P2", "P3"]
    x = np.arange(3)
    series = [(nome, historicos[nome]["P_hist"][-1]) for nome in nomes]
    if ref is not None:
        series.append(("Analítico", np.asarray(ref["P"], dtype=float)))

    n = len(series)
    largura = 0.8 / n
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, (nome, P) in enumerate(series):
        ax.bar(x + (k - (n - 1) / 2) * largura, P, largura, label=nome,
               color=_COR.get(nome))
    ax.set_xticks(x)
    ax.set_xticklabels(rotulos)
    ax.set_ylabel("potência [MW]")
    ax.set_title("Despacho final por método")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    return _salva(fig, "q2_despacho_final.png")


def evolucao_potencias(historicos):
    n = len(historicos)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5), squeeze=False)
    for ax, (nome, h) in zip(axes[0], historicos.items()):
        P = h["P_hist"]
        for j in range(3):
            ax.plot(P[:, j], "-o", ms=3, label=f"P{j+1}")
        ax.set_xlabel("iteração externa")
        ax.set_ylabel("potência [MW]")
        ax.set_title(f"Evolução das potências — {nome}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    return _salva(fig, "q2_evolucao_potencias.png")


def sensibilidade_penalidade(sensibilidade):
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax2 = ax1.twinx()
    metodos = sorted({r["metodo"] for r in sensibilidade})
    for metodo in metodos:
        linhas = [r for r in sensibilidade
                  if r["metodo"] == metodo and not r["divergiu"]]
        linhas.sort(key=lambda r: r["penalidade"])
        if not linhas:
            continue
        pen = [r["penalidade"] for r in linhas]
        viol = [max(r["viol_balanco"], 1e-16) for r in linhas]
        aval = [r["aval"] for r in linhas]
        cor = _COR.get(metodo)
        ax1.loglog(pen, viol, "-o", color=cor, label=f"{metodo} — |h|")
        ax2.semilogx(pen, aval, "--s", color=cor, alpha=0.6,
                     label=f"{metodo} — avaliações")
    ax1.set_xlabel("penalidade inicial")
    ax1.set_ylabel("|h(P*)| final  [MW]")
    ax2.set_ylabel("nº de avaliações")
    ax1.set_title("Sensibilidade à penalidade inicial")
    ax1.grid(True, which="both", alpha=0.3)
    linhas1, rot1 = ax1.get_legend_handles_labels()
    linhas2, rot2 = ax2.get_legend_handles_labels()
    ax1.legend(linhas1 + linhas2, rot1 + rot2, fontsize=8)
    return _salva(fig, "q2_sensibilidade_penalidade.png")


def custos_incrementais(historicos, ref=None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rotulos = ["P1", "P2", "P3"]
    x = np.arange(3)
    series = []
    for nome, h in historicos.items():
        P = h["P_hist"][-1]
        # custo incremental corrigido pelos penalty factors: deve ser ~igual
        ci_corr = custo_incremental(P) / (1.0 - grad_perdas(P))
        series.append((nome, ci_corr))
    if ref is not None:
        P = np.asarray(ref["P"], dtype=float)
        series.append(("Analítico", custo_incremental(P) / (1.0 - grad_perdas(P))))

    n = len(series)
    largura = 0.8 / n
    for k, (nome, ci) in enumerate(series):
        ax.bar(x + (k - (n - 1) / 2) * largura, ci, largura, label=nome,
               color=_COR.get(nome))
    if ref is not None:
        ax.axhline(ref["lambda"], ls=":", color="k",
                   label=f"λ = {ref['lambda']:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(rotulos)
    ax.set_ylabel("dCi/dPi · 1/(1−∂PL/∂Pi)  [R$/MWh]")
    ax.set_title("Custos incrementais corrigidos no ótimo")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    return _salva(fig, "q2_custos_incrementais.png")


# ---------------------------------------------------------------------------
# Orquestrador
# ---------------------------------------------------------------------------
def gera_todos(historicos, sensibilidade=None, ref=None):
    """Gera todas as figuras possíveis com os dados disponíveis."""
    _garante_dir()
    salvos = [
        convergencia_custo(historicos),
        violacao_balanco(historicos),
        despacho_final(historicos, ref),
        evolucao_potencias(historicos),
        custos_incrementais(historicos, ref),
    ]
    if sensibilidade:
        salvos.append(sensibilidade_penalidade(sensibilidade))
    return salvos


if __name__ == "__main__":
    from q2_solvers import resolve_exterior, resolve_lagrangeano, coleta_historico
    from q2_sensibilidade import varre_tudo
    from q2_validacao import referencia_analitica

    print("Rodando solvers (config final)...")
    sol_e = resolve_exterior(penalidade=1000.0, aceleracao=1.2)
    sol_l = resolve_lagrangeano(penalidade=100.0, aceleracao=1.2)
    historicos = {
        "Exterior": coleta_historico(sol_e),
        "Lagrangeano": coleta_historico(sol_l),
    }

    print("Rodando varredura de sensibilidade...")
    sensibilidade = varre_tudo(salvar=False)

    print("Calculando referência analítica...")
    ref = referencia_analitica()

    print("Gerando figuras...")
    for caminho in gera_todos(historicos, sensibilidade, ref):
        print(f"  {caminho}")

"""
Questão 2 — Despacho econômico de 3 unidades geradoras (orquestrador)
=====================================================================

min  f(P1, P2, P3) = C1(P1) + C2(P2) + C3(P3)
        C1 = 0.15*P1^2 + 38*P1 + 756
        C2 = 0.10*P2^2 + 46*P2 + 451
        C3 = 0.25*P3^2 + 40*P3 + 1049
s.a. h:  P1 + P2 + P3 = 850 + PL          (balanço de potência, igualdade)
        PL = P^T B P                      (perda na transmissão)
     150 <= P1 <= 600 ; 100 <= P2 <= 400 ; 50 <= P3 <= 200

Métodos: Penalidade Exterior e Lagrangeano Aumentado.
(Penalidade Interior não é usada: a biblioteca não suporta restrições de
 igualdade nesse método.)

Este arquivo apenas orquestra os módulos `q2_*`:
    q2_problema     — dados e formulação
    q2_solvers      — execução dos métodos e coleta de histórico
    q2_sensibilidade— varredura da penalidade inicial (Seção 4)
    q2_validacao    — referência analítica (lambda-iteration)
    q2_graficos     — figuras para o relatório
"""

from q2_problema import objetivo, perda_transmissao, X0
from q2_solvers import resolve_exterior, resolve_lagrangeano, coleta_historico, resumo
from q2_sensibilidade import varre_tudo, imprime_tabela, salva_csv, _melhor, CSV_PATH
from q2_validacao import valida, referencia_analitica
from q2_graficos import gera_todos

# Configurações finais escolhidas/documentadas no estudo de sensibilidade (T3).
CONFIG_EXTERIOR = {"penalidade": 1000.0, "aceleracao": 1.2}
CONFIG_LAGRANGEANO = {"penalidade": 100.0, "aceleracao": 1.2}


def main():
    # ---- 1) Sanidade do ponto inicial -------------------------------------
    print("========== Sanidade (X0) ==========")
    print(f"X0     = {X0}  (ΣX0 = {X0.sum():.2f} MW)")
    print(f"f(X0)  = {objetivo(X0):.2f} R$/h")
    print(f"PL(X0) = {perda_transmissao(X0):.4f} MW")

    # ---- 2) Configuração final dos dois métodos ---------------------------
    print("\n========== Métodos (configuração final) ==========")
    sol_exterior = resolve_exterior(**CONFIG_EXTERIOR)
    resumo("Penalidade Exterior", sol_exterior)

    sol_lagrangeano = resolve_lagrangeano(**CONFIG_LAGRANGEANO)
    resumo("Lagrangeano Aumentado", sol_lagrangeano)

    historicos = {
        "Exterior": coleta_historico(sol_exterior),
        "Lagrangeano": coleta_historico(sol_lagrangeano),
    }

    # ---- 3) Estudo de sensibilidade da penalidade -------------------------
    print("\n========== Sensibilidade da penalidade ==========\n")
    sensibilidade = varre_tudo(salvar=False)
    imprime_tabela(sensibilidade)
    salva_csv(sensibilidade)
    print(f"\nCSV salvo em: {CSV_PATH}")
    melhor_e = _melhor([r for r in sensibilidade if r["metodo"] == "Exterior"])
    melhor_l = _melhor([r for r in sensibilidade if r["metodo"] == "Lagrangeano"])
    if melhor_e:
        print(f"Config final Exterior   : pen0={melhor_e['penalidade']:.0f}, "
              f"|h|={melhor_e['viol_balanco']:.2e}, aval={melhor_e['aval']}")
    if melhor_l:
        print(f"Config final Lagrangeano: pen0={melhor_l['penalidade']:.0f}, "
              f"|h|={melhor_l['viol_balanco']:.2e}, aval={melhor_l['aval']}")

    # ---- 4) Validação independente ----------------------------------------
    print("\n========== Validação analítica ==========")
    valida(sol_lagrangeano)
    ref = referencia_analitica()

    # ---- 5) Figuras -------------------------------------------------------
    print("\n========== Figuras ==========")
    for caminho in gera_todos(historicos, sensibilidade, ref):
        print(f"  {caminho}")

    print("\nConcluído.")


if __name__ == "__main__":
    main()

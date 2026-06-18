import numpy as np
from otimo import PenalidadeExterior, GradienteConjugado, SecaoAurea

# Problema: min f(x) = x₁² + x₂²
#          s.a. x₁ + x₂ ≥ 1
#               x₁, x₂ ≥ 0

def objetivo(x):
    return x[0]**2 + x[1]**2

def restricao1(x):
    return x[0] + x[1] - 1  # x₁ + x₂ ≥ 1

def restricao2(x):
    return x[0]  # x₁ ≥ 0

def restricao3(x):
    return x[1]  # x₂ ≥ 0

# Configuração
restricoes = [restricao1, restricao2, restricao3]
tipos = np.array(['>', '>', '>'])
x0 = np.array([1.0, 1.0])

busca_1d = SecaoAurea(precisao=1e-6)
irrestrito = GradienteConjugado(busca_1d, precisao=1e-6)
restrito = PenalidadeExterior(precisao=1e-6)

# Resolução
solucao = restrito.resolva(objetivo, x0, restricoes, tipos, 
                          irrestrito, penalidade=1.0, aceleracao=2.0,
                          disp=True)

print(f"Solução: {solucao.x}")
print(f"Valor objetivo: {solucao.fx}")
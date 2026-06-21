import os
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize

os.makedirs("figuras", exist_ok=True)

# ============================================================
# Problema da treliça
# ============================================================

P = 20.0
SQRT2 = np.sqrt(2.0)

x0 = np.array([1.0, 3.0], dtype=float)


def f(x):
    x1, x2 = x
    return 2.0 * SQRT2 * x1 + x2


def denom(x):
    x1, x2 = x
    return SQRT2 * x1**2 + 2.0 * x1 * x2


# Restrições no formato g(x) <= 0

def g1(x):
    x1, x2 = x
    return P * (x2 + SQRT2 * x1) / denom(x) - 20.0


def g2(x):
    x1, x2 = x
    return P / (x1 + SQRT2 * x2) - 20.0


def g3(x):
    x1, x2 = x
    return -P * x2 / denom(x) + 5.0


def g4(x):
    return 0.1 - x[0]


def g5(x):
    return x[0] - 5.0


def g6(x):
    return 0.1 - x[1]


def g7(x):
    return x[1] - 5.0


restricoes = [g1, g2, g3, g4, g5, g6, g7]


def gs(x):
    return np.array([g(x) for g in restricoes])


def max_violacao(x):
    return max(0.0, np.max(gs(x)))


def factivel(x):
    return np.all(gs(x) <= 0)


# ============================================================
# 1) Penalidade Interior
# ============================================================

def penalidade_interior():
    x = x0.copy()
    u = 1.0
    beta = 0.5
    precisao = 1e-4

    historia = [x.copy()]
    f_hist = [f(x)]
    viol_hist = [max_violacao(x)]

    while True:
        x_anterior = x.copy()

        def FI(x):
            gx = gs(x)

            if np.any(gx >= 0):
                return 1e20

            return f(x) - u * np.sum(np.log(-gx))

        sol = optimize.minimize(
            FI,
            x,
            method="Nelder-Mead",
            options={"maxiter": 5000}
        )

        x = sol.x
        historia.append(x.copy())
        f_hist.append(f(x))
        viol_hist.append(max_violacao(x))

        u = beta * u

        if np.linalg.norm(x - x_anterior) / np.linalg.norm(x) < precisao:
            break

        if len(historia) > 50:
            break

    return x, np.array(historia), np.array(f_hist), np.array(viol_hist)


# ============================================================
# 2) Penalidade Exterior
# ============================================================

def penalidade_exterior():
    x = x0.copy()
    u = 1.0
    alpha = 2.0
    precisao = 1e-4

    historia = [x.copy()]
    f_hist = [f(x)]
    viol_hist = [max_violacao(x)]

    while True:
        x_anterior = x.copy()

        def FE(x):
            gx = gs(x)
            penalizacao = np.sum(np.maximum(0.0, gx) ** 2)
            return f(x) + u * penalizacao

        sol = optimize.minimize(
            FE,
            x,
            method="BFGS",
            options={"maxiter": 5000}
        )

        x = sol.x
        historia.append(x.copy())
        f_hist.append(f(x))
        viol_hist.append(max_violacao(x))

        u = alpha * u

        if np.linalg.norm(x - x_anterior) / np.linalg.norm(x) < precisao:
            break

        if len(historia) > 50:
            break

    return x, np.array(historia), np.array(f_hist), np.array(viol_hist)


# ============================================================
# 3) Lagrangeano Aumentado
# ============================================================

def lagrangeano_aumentado():
    x = x0.copy()
    mu = np.zeros(len(restricoes))
    u = 1.0
    alpha = 1.2
    precisao = 1e-4

    historia = [x.copy()]
    f_hist = [f(x)]
    viol_hist = [max_violacao(x)]

    while True:
        x_anterior = x.copy()

        def LA(x):
            gx = gs(x)
            gx_pos = np.maximum(0.0, gx)

            return (
                f(x)
                + np.dot(mu, gx_pos)
                + (u / 2.0) * np.sum(gx_pos ** 2)
            )

        sol = optimize.minimize(
            LA,
            x,
            method="BFGS",
            options={"maxiter": 5000}
        )

        x = sol.x
        gx = gs(x)

        mu = np.maximum(0.0, mu + u * gx)
        u = alpha * u

        historia.append(x.copy())
        f_hist.append(f(x))
        viol_hist.append(max_violacao(x))

        if np.linalg.norm(x - x_anterior) / np.linalg.norm(x) < precisao:
            break

        if len(historia) > 50:
            break

    return x, np.array(historia), np.array(f_hist), np.array(viol_hist)


# ============================================================
# Rodar métodos
# ============================================================

x_int, hist_int, f_int, v_int = penalidade_interior()
x_ext, hist_ext, f_ext, v_ext = penalidade_exterior()
x_lag, hist_lag, f_lag, v_lag = lagrangeano_aumentado()

metodos = {
    "Penalidade Interior": (x_int, hist_int, f_int, v_int),
    "Penalidade Exterior": (x_ext, hist_ext, f_ext, v_ext),
    "Lagrangeano Aumentado": (x_lag, hist_lag, f_lag, v_lag),
}

print("\n===== RESULTADOS =====")
print(f"{'Método':<25} {'x1':>12} {'x2':>12} {'f(x*)':>12} {'Violação':>12}")

for nome, (x, hist, fh, vh) in metodos.items():
    print(
        f"{nome:<25} "
        f"{x[0]:>12.6f} "
        f"{x[1]:>12.6f} "
        f"{f(x):>12.6f} "
        f"{max_violacao(x):>12.3e}"
    )


# ============================================================
# Malha para gráficos
# ============================================================

x1lim = (0.1, 1.5)
x2lim = (0.1, 1.5)

X1, X2 = np.meshgrid(
    np.linspace(*x1lim, 300),
    np.linspace(*x2lim, 300)
)

F = np.zeros_like(X1)
FEAS = np.zeros_like(X1, dtype=bool)

for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        x = np.array([X1[i, j], X2[i, j]])
        F[i, j] = f(x)
        FEAS[i, j] = factivel(x)


# ============================================================
# Figura 1: região factível
# ============================================================

plt.figure(figsize=(7, 5))
plt.contour(X1, X2, F, levels=30)
plt.contourf(X1, X2, FEAS, levels=[0.5, 1.5], alpha=0.25)

plt.plot(x_lag[0], x_lag[1], "*k", markersize=15, label="Ótimo")

plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title("Região factível, curvas de nível e ponto ótimo")
plt.xlim(*x1lim)
plt.ylim(*x2lim)
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("figuras/q1_regiao_factivel.png", dpi=300)
plt.close()


# ============================================================
# Figura 2: trajetórias
# ============================================================

plt.figure(figsize=(7, 5))
plt.contour(X1, X2, F, levels=30)
plt.contourf(X1, X2, FEAS, levels=[0.5, 1.5], alpha=0.25)

for nome, (x, hist, fh, vh) in metodos.items():
    plt.plot(hist[:, 0], hist[:, 1], "*--", label=nome)

plt.plot(x_lag[0], x_lag[1], "*k", markersize=15, label="Ótimo")

plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title("Trajetórias dos métodos")
plt.xlim(*x1lim)
plt.ylim(*x2lim)
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("figuras/q1_trajetorias.png", dpi=300)
plt.close()


# ============================================================
# Figura 3: convergência por iteração
# ============================================================

plt.figure(figsize=(7, 5))

for nome, (x, hist, fh, vh) in metodos.items():
    plt.plot(range(len(fh)), fh, "*--", label=nome)

plt.xlabel("Iteração externa")
plt.ylabel(r"$f(x_k)$")
plt.title("Convergência da função objetivo")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("figuras/q1_convergencia_iter.png", dpi=300)
plt.close()


# ============================================================
# Figura 4: violação máxima
# ============================================================

plt.figure(figsize=(7, 5))

for nome, (x, hist, fh, vh) in metodos.items():
    plt.semilogy(range(len(vh)), np.maximum(vh, 1e-16), "*--", label=nome)

plt.xlabel("Iteração externa")
plt.ylabel("Violação máxima")
plt.title("Violação máxima das restrições")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("figuras/q1_violacao.png", dpi=300)
plt.close()


print("\nFiguras geradas na pasta figuras/")
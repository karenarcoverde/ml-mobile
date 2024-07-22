import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Geração de valores de x para a PDF e CDF
x = np.linspace(-4, 4, 100)
pdf = norm.pdf(x)  # PDF da distribuição normal padrão
cdf = norm.cdf(x)  # CDF da distribuição normal padrão

# Ajustando o tamanho da fonte globalmente
plt.rcParams.update({'font.size': 14})

# Plot da PDF
plt.figure(figsize=(10, 6))
plt.plot(x, pdf, label='PDF da Distribuição Normal Padrão')
plt.title('Função Densidade de Probabilidade (PDF)', fontsize=16)
plt.xlabel('x', fontsize=14)
plt.ylabel('f(x)', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

# Plot da CDF
plt.figure(figsize=(10, 6))
plt.plot(x, cdf, label='CDF da Distribuição Normal Padrão')
plt.title('Função de Distribuição Acumulada (CDF)', fontsize=16)
plt.xlabel('x', fontsize=14)
plt.ylabel('F(x)', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()
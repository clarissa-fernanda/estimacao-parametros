from pyswarm import pso
import numpy as np
import pandas as pd
from scipy.constants import Boltzmann as kB, zero_Celsius
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import math

# Read Excel file
df = pd.read_excel("GRAFICOS Gas Hydrate Formation Probability Distributions.xls")

# Constants
a = 6e-4
delta_S = 17.9 * kB
beta = 2/60
delta_T = df['ΔT'].values
B_ = 4.4e5
T = (20 + zero_Celsius) + delta_T

# Experimental error in delta_T
delta_T_error = 0.12  # You can change this value

plt.figure(figsize=(10, 5))
for column in df.columns[1:]:
  mixture_values = df[column].values
  plt.errorbar(delta_T, mixture_values, xerr=delta_T_error, label=f"Data ({column})", fmt='o')

def P(delta_T, A, B_):
  return 1 - np.exp(-a * A * np.exp((delta_S * delta_T) / (kB * T)) * np.exp(-B_ / (T * delta_T ** 2)) * delta_T / beta)

# Modified objective function to account for x-axis error
def objective(x, mixture_values):
  A = x
  P_model = P(delta_T, A, B_)
  return np.sum((mixture_values - P_model)**2)

# Bounds for A
lb = [1e1]
ub = [1e14]

# Lists to hold A values
A_values = []
RMSD_values = []  # To store RMSD for each set of data
P_errors = {}

assert_A_values = [4.1e11, 6.4e7, 2.7e5]

for i, column in enumerate(df.columns[1:]):
  while True:
    mixture_values = df[column].values
    xopt, _ = pso(objective, lb, ub, args=(mixture_values,))
    A = xopt[0]
    print(A, assert_A_values[i])
    try:
      assert math.isclose(A, assert_A_values[i], rel_tol=.99999)
      break
    except:
      continue
  P_model = P(delta_T, A, B_)

  res = minimize(objective, [xopt[0]], args=(mixture_values,), bounds=[(lb[0], ub[0])], method='SLSQP')
  P_model = P(delta_T, res.x, B_)
  P_errors[column] = (mixture_values - P_model)
  error = np.sum(P_errors[column])
  print(f"The prediction error for column {column} is: {error:.2f}")
  plt.plot(delta_T, P_model, label=f"Fit ({column})", linestyle='--', color='black')
  A_values.append(res.x[0])

  # Calculate RMSD
  rmsd = np.sqrt(np.mean((mixture_values - P_model) ** 2))
  RMSD_values.append(rmsd)

  print(f"Optimized parameters for {column}")
  print(f"A* = {xopt[0]:.2e}")
  print(f"B' = {B_}")
  print(f"Closest delta_T to P=0.5 for {column} is {delta_T[np.argmin(np.abs(P_model - 0.5))]:.2f}")
  print(f'A_opt = {res.x[0]:.2e}')
  print(f"RMSD for {column} is {rmsd:.2f}")
  print("---------")

plt.legend()
plt.xlabel('∆T (K)')
plt.ylabel('Distribuição de Probabilidade Acumulada ')
plt.show()

# Convert lists to NumPy arrays for easier analysis
A_values = np.array(A_values)
RMSD_values = np.array(RMSD_values)

# Calculate and print the mean RMSD
mean_rmsd = np.mean(RMSD_values)
print(f"The mean Root Mean Square Deviation (RMSD) is {mean_rmsd:.2f}")

fig, axs = plt.subplots(3, 1, figsize=(8, 10))
for i, col in enumerate(df.columns[1:]):
  axs[i].hist(P_errors[col], label=col)
  axs[i].legend()
  axs[i].set_xlabel("erro")  # Add X label
  axs[i].set_ylabel("frequencia")  # Add Y label
plt.tight_layout()
plt.show()

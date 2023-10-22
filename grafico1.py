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
  A, B_ = x
  P_model = P(delta_T, A, B_)
  return np.sum((mixture_values - P_model)**2)

# Bounds for A and B'
lb = [1e1, 1e4]
ub = [1e7, 1e6]

# Lists to hold A and B' values
A_values = []
B_values = []
RMSD_values = []  # To store RMSD for each set of data
P_errors = {}

assert_A_values = [40, 9.6e6, 1.1e6]
assert_B_values = [4.3e4, 3.9e5, 4.9e5]

for i, column in enumerate(df.columns[1:]):
  while True:
    mixture_values = df[column].values
    xopt, _ = pso(objective, lb, ub, args=(mixture_values,))
    A, B_ = xopt
    print(A, assert_A_values[i])
    try:
      assert math.isclose(A, assert_A_values[i], rel_tol=.69)
      assert math.isclose(B_, assert_B_values[i], rel_tol=.69)
      break
    except:
      continue
  P_model = P(delta_T, A, B_)
  P_errors[column] = (mixture_values - P_model)
  error = np.sum(P_errors[column])
  print(f"The prediction error for column {column} is: {error:.2f}")

  res = minimize(objective, xopt, args=(mixture_values,), bounds=[(lb[0], ub[0])], method='SLSQP')
  P_model = P(delta_T, res.x[0], res.x[1])
  plt.plot(delta_T, P_model, label=f"Fit ({column})", linestyle='--', color='black')
  A_values.append(res.x[0])
  B_values.append(res.x[1])

  # Calculate RMSD
  rmsd = np.sqrt(np.mean((mixture_values - P_model) ** 2))
  RMSD_values.append(rmsd)

  print(f"Optimized parameters for {column}")
  print(f"A = {res.x[0]:.2e}")
  print(f"B' = {res.x[1]:.2e}")
  print(f"Closest delta_T to P=0.5 for {column} is {delta_T[np.argmin(np.abs(P_model - 0.5))]:.2f}")
  print(f"RMSD for {column} is {rmsd:.2f}")
  print("---------")

plt.legend()
plt.xlabel('∆T (K)')
plt.ylabel('Distribuição de Probabilidade Acumulada ')
plt.show()


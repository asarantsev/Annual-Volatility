import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from scipy import stats

save_path = "2-1"
df = pd.read_excel('VolatilityData.xlsx', sheet_name = 'Yearly')
df['Year'] = pd.to_datetime(df['Year'])
df.set_index('Year', inplace=True)
df.index = pd.date_range(start=df.index[0], periods=len(df), freq='YE')  # Set explicit yearly frequency
df.dropna(inplace=True)

model = AutoReg(df['Log Return Stdev'], lags=1)
ar1_model = model.fit()
slope = ar1_model.params
print(slope)

model1 = AutoReg(df['Ln(Log Return Stdev)'], lags = 1)
ar1_model1 = model1.fit()
slope1 = ar1_model1.params
print(slope1)

#ln(S_t) = a+ b* lnS_(t-1)+Z_t

#autocorrelation of ln stdev
plt.figure(figsize=(10,6))
plot_acf(df['Ln(Log Return Stdev)'])
plt.title("Ln(Volatility) \n ACF")
plt.savefig(save_path + "lnvacf.png", dpi=300, bbox_inches='tight')
#acf of residuals ln stdev
residuals1 = ar1_model1.resid
plt.figure(figsize=(10,6))
plot_acf(residuals1)
plt.title("Ln(Volatility) \n ACF for Original Values of Residuals")
plt.savefig(save_path + "lnvracf.png", dpi=300, bbox_inches='tight')
#absolute value
plt.figure(figsize=(10,6))
plot_acf(np.abs(residuals1))
plt.title("Ln(Volatility) \n ACF for Absolute Value of Residuals")
plt.savefig(save_path + "lnvaracf.png", dpi=300, bbox_inches='tight')
#QQ plot of residuals (ln of stdev)
sm.qqplot(residuals1, line = 's')
plt.title("Ln(Volatility) \n Q-Q Plot vs Normal for Residuals")
plt.savefig(save_path + "lnvrqq.png", dpi=300, bbox_inches='tight')


#autocorrelation of stdev
plt.figure(figsize=(10,6))
plot_acf(df['Log Return Stdev'])
plt.title("Volatility \n ACF")
plt.savefig(save_path + "vacf.png", dpi=300, bbox_inches='tight')

#acf of residuals
residuals = ar1_model.resid
plt.figure(figsize=(10,6))
plot_acf(residuals)
plt.title("Volatility \n ACF for Original Values of Residuals")
plt.savefig(save_path + "vracf.png", dpi=300, bbox_inches='tight')

#acf of abs residuals
plt.figure(figsize=(10,6))
plot_acf(np.abs(residuals))
plt.title("Volatility \n ACF for Absolute Value of Residuals")
plt.savefig(save_path + "varacf.png", dpi=300, bbox_inches='tight')

#QQ plot of residuals
sm.qqplot(residuals, line = 's')
plt.title("Volatility \n Q-Q Plot vs Normal for Residuals")
plt.savefig(save_path + "vrqq.png", dpi=300, bbox_inches='tight')


#Finding expected value and variance of ln(std)
a = ar1_model1.params[0]  # Intercept of ln(std)
b = ar1_model1.params[1]   # Slope of ln(std)
varZ = residuals1.var()  # Variance of residuals Z

# Find stationary distribution mean and variance for ln(std)
stationaryMean = a / (1 - b)  # Mean of stationary distribution
stationaryVar = varZ / (1 - b**2)  # Variance of stationary distribution

print("Stationary distribution of ln(S_t):")
print(f"Mean = {stationaryMean}")
print(f"Variance = {stationaryVar}")

# Expectation E[S_t^u] where S_t = exp(ln(S_t))
def computeE_S_t_u(u):
    return np.exp(u * stationaryMean + 0.5 * u**2 * stationaryVar)#using normal MGF

uValues = [1, 2] #E[S_t^1] and E[S_t^2]
for u in uValues:
    E_S_u = computeE_S_t_u(u)
    print(f"E[S_t^{u}] = {E_S_u}")

# Variance of S_t (log-normal distribution property)
E_S = computeE_S_t_u(1)  # E[S_t^1]
E_S_squared = computeE_S_t_u(2)  # E[S_t^2]
Var_S_t = E_S_squared - E_S**2

print(f"Variance of S_t: {Var_S_t}")

print(df.tail())


import matplotlib.pyplot as plt
import statsmodels.api as sm

dta = sm.datasets.sunspots.load_pandas().data
print(dta)

plt.plot(dta['YEAR'], dta['SUNACTIVITY'])
plt.show()

# Show autocorrelation function.
# General correlation of lags with past lags.
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(dta['SUNACTIVITY'], lags=50)
plt.show()

# Show partial-autocorrelation function.
# Shows correlation of 1st lag with past lags.
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(dta['SUNACTIVITY'], lags=50)
plt.show()

print(dta)

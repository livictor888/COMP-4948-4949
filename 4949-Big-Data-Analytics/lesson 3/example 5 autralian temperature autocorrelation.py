from pandas                     import read_csv
import matplotlib.pyplot        as plt
from statsmodels.tsa.ar_model   import AutoReg
from sklearn.metrics            import mean_squared_error
from   math                     import sqrt
import warnings
warnings.filterwarnings("ignore")

# Load the data.
PATH = "C:\\datasets\\"
series = read_csv(PATH + 'daily-min-temperatures.csv',
                  header=0, index_col=0, parse_dates=True, squeeze=True)

# Plot ACF.
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(series, lags=20)
plt.show()

# Plot PACF.
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(series, lags=20)
plt.show()

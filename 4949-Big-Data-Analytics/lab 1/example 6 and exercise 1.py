import pandas as pd

co2 = [
342.76, 343.96, 344.82, 345.82, 347.24, 348.09, 348.66, 347.90, 346.27, 344.21,
342.88, 342.58, 343.99, 345.31, 345.98, 346.72, 347.63, 349.24, 349.83, 349.10,
347.52, 345.43, 344.48, 343.89, 345.29, 346.54, 347.66, 348.07, 349.12, 350.55,
351.34, 350.80, 349.10, 347.54, 346.20, 346.20, 347.44, 348.67]

# increments weekly
# starting from the first monday of the most recent september
df = pd.DataFrame({'CO2':co2}, index=pd.date_range(
     start='09-01-2022', periods=len(co2), freq='W-MON'))
print(df)

import pandas as pd
PATH = "C:\\datasets\\"
FILE = "aritzia.csv"
df   = pd.read_csv(PATH + FILE,  parse_dates=['Date'], index_col='Date')
print(type(df.index)) # Verify the data type.
print(df)

df['year']    = df.index.year
df['month']   = df.index.month
df['day']     = df.index.day
df['dayName'] = df.index.strftime("%A")
print(df)

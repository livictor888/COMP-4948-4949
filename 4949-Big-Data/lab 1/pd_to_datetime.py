import pandas as pd

df = pd.DataFrame(
    data={
        "Dates":['2014-07-04', '2014-08-04', '2015-07-04', '2015-08-04'],
        "Temperature":[28,27,29,26]
    }
)
print(df)

# change 'Dates' from string to datetime object
df['Dates'] = pd.to_datetime(df['Dates'])

# set indexing to "Dates" from integer
df          = df.set_index('Dates')
print(df)
print(type(df))
print("Index data type: ")
print(type(df.index))

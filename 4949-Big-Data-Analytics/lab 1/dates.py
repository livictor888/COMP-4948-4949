import pandas as pd
from datetime import datetime
dt1 = datetime(year=2015, month=12, day=4)
dt2 = pd.to_datetime('12/8/1952')
dt3 = pd.to_datetime('12/8/1952', dayfirst=True)

print(dt1)
print(dt2)
print(dt3)


# datetime object to date object
d1 = dt1.date()
print(d1)
d2 = dt2.date()
print(d2)
d3 = dt3.date()
print(d3)

import pandas as pd
import numpy as np


""" Series """
x = pd.Series([6, 4, 3, 6])
print(x)

x = pd.Series([6, 4, 3, 6], index=['a', 'b', 'c', 'd'])
print(x)
print('c = ', x['c'])

data = {'abc': 1, 'def': 2, 'xyz': 3}
print(pd.Series(data))

x = pd.Series(3, index=['a', 'b', 'c', 'd'])
print(x)


""" DataFrame """
dates = pd.date_range('20170505', periods=8)
print(dates)

df = pd.DataFrame(np.random.randn(8, 3), index=dates, columns=list('ABC'))
print(df)

df.head()

df.tail()

df.describe()

df.apply(np.cumsum)
print(df)

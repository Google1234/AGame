import pandas
import re
import Config
rats=pandas.read_csv("test.csv",header=None)
print type(rats.iloc[5][3])
print rats.iloc[5][3]
print rats.iloc[5][3].split(":")
if re.match(pattern=":",string=':'):
    print 1111
#print type(rats.iloc[])
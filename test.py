import pandas
import re
import Config
rats=pandas.read_csv("test.csv",header=None)
print type(rats.iloc[5][3])
if re.match(pattern=".*:",string=rats.iloc[5][3]):
    list=rats.iloc[5][3].split(":")
    print int(list[0]),int(list[1])
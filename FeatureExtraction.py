import csv
import Config
def sort(filename,sort_by_line,output_file):
    f=open(filename)
    f_csv = csv.reader(f)
    for row in f_csv:
        print row
    f.close()

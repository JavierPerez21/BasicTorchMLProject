import os
import re
import datetime

path = str(os.getcwd()) + "\\" + "Experiment_logs" + "\\"
log_path = 'test.txt'

line1 = "Today is" + str(datetime.datetime.now())
line2 = 'Second line for test'
items = [line1, line2]
with open(path + log_path, 'w') as fp:
    for item in items:
        fp.write(f"{item}\n")
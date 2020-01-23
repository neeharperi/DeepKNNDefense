import os
import Utilities

import pdb
logFileDirectory = "./Logs/"

for logFile in os.listdir(logFileDirectory):
    data = Utilities.parseLogFile(logFileDirectory + logFile)

    if None in data.values():
        os.remove(logFileDirectory + logFile)
        print("Removed " + logFileDirectory + logFile)
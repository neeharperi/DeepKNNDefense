import os
import Utilities

import pdb
logFileDirectory = "./Logs/"

for logFile in os.listdir(logFileDirectory):
    data = Utilities.parseLogFile(logFileDirectory + logFile)

    if data["Status"] is None:
        os.remove(logFileDirectory + logFile)
        print("Removed " + logFileDirectory + logFile)
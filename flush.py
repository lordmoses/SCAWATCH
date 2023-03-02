#This is a script to make sure you end all procmon processes (if they still remain) in the Windows System

import psutil, sys

def end_analysis():

    to_kill = ['Procmon.exe', 'Procmon64.exe']
    ## Shutdown procmon instances upon Keyboard Interrupt
    for proc in psutil.process_iter():
        if proc.name() in to_kill:
            proc.kill()
    print("Done.")
    sys.exit()

end_analysis()


#param ($logfile, $saveas)
#C:/Users/username/Desktop/ProcessMonitor/Procmon.exe /Quiet /OpenLog $logfile /SaveAs $saveas

param ($procmonlocation, $logfile, $saveas)
&$procmonlocation /Quiet /Minimized /OpenLog $logfile /SaveApplyFilter /SaveAs $saveas

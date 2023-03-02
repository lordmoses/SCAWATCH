#param ($configfile, $backingfile)
#C:\Users\tingwei\Desktop\ProcessMonitor\Procmon.exe /Quiet /LoadConfig $configfile /BackingFile $backingfile

param ($procmonlocation, $configfile, $backingfile)
&$procmonlocation /LoadConfig $configfile /BackingFile $backingfile /Quiet /Minimized /AcceptEula
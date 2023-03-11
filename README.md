# SCAWATCH

## Automated Dynamic Analysis of SCADA Software Execution

This tool, SCAWATCH, monitors and analyzes the dynamic system-level execution of SCADA software processes. The purpose is to detect malicious system-level activity, such as injected malware code or logic aimed at causing physical damage or disruption to physical processes. With SCAWATCH, the collected logs can be stored and analyzed locally or sent to a remote server for storage and/or analysis. 

### Host Requrements

SCAWATCH uses a Windows utility called **Process Monitor (Procmon)** to trace software processes. Procmon can be downloaded here. https://learn.microsoft.com/en-us/sysinternals/downloads/procmon. Procmon captures events from five (5) different category of system activity: Registry, Processes/Threads, Filesystem, Network, and Profilling events. More information on Procmon can be found here: https://adamtheautomator.com/procmon/

SCAWATCH is **python-based** and executed in **Powershell**. Powershell must be run as **administrator** (to allow Procmon to work). Tested on Windows 7.

### Basic Steps:
1. `git clone https://github.com/lordmoses/SCAWATCH.git`
2. `cd SCAWATCH`
3. `pip install -r requirements.txt`
4. Edit config.json (to suit your Windows environment and other tracing needs, see instructions below)
5. `python3 scawatch.py`
6. To end things, press CTRL -C


```
SCAWATCH v1.1
==User Specification==
Procmon Location: C:/Users/username/Desktop/ProcessMonitor/Procmon.exe
SCADA software process investigated: ScadaBR.exe
Log batch size in MB: 50
==End==
Making procmon configuration and filters ...
Starting Tracing ...

Procmon instance  1  Date-time: 10-08-35-03-02-2023
Logging trace in logs/logfile-10-06-24-03-02-2023.pml ...
```




### SCAWATCH performs the following:
1.	Automatically traces the system-level software execution (e.g., library/API calls) of a running software process.
2.	Automatically manages the collection and storage of the traces (or logs) based on user-specification. Traces are collected in batches of user-specified sizes and zipped. Zipped log files can be stored locally or sent to a configurable remote server.
3.	Automatically analyzes the traces for anomalies (e.g., injected attacks). Anomaly detection is based on statistical analysis of the frequency and timing behavior of SCADA “physical-targeted” executions (**Not yet integrated in this code release**).

A Windows VM with a proof-of-concept demonstration is provided (via special request). It runs an open-source SCADA suite, called ScadaBR.

### User Configuration (config.json)
1. **`size_limit`** refers to the file size (in megabytes) the Procmon logs are allowed to reached before it is zipped and stored. 
2. **`check_interval`** is the time in seconds our daemon has to poll to check if the file size limit is reached. 5 seconds is fairly good, but use your judgement based on your environment.
3. **`procmon_location`** refer to the absolute path where your procmon.exe executable resides on your windows machine
4. **`scada_process`** refer to the process name you want to trace. To know this, first run your software and use task manager to know the process name. E.g., process name of chrome is "ScadaBr.exe".
5. **`ENABLE_LOCAL_STORAGE`** allows user to specify if they want to retain the collected traces locally (in a created logs folder) within the same directory where SCAWATCH was launched.
6. **`SEND_LOGS_TO_REMOTE_SERVER`** allows user to specify if they want to send the collected traces to a remote server. This file transfer is done with the SCP utility available in the SSH protocol, using the command `scp -i <identify_file> <sourcefile> <destination IP and folder>`. In order to work, the powershell where SCAWATCH is ran should be able to run this command successfully. Please test this first, otherwise the file transfer may not work.
7. **`remote_server_machine`** specifies the user's SSH login information in the form of `<user>@<SSH server machine IP or domain name>`.
8. **`remote_server_folder`** specifies the folder on the remote (SSH) server machine where the transferred logs will be saved/stored.
9. **`ssh_client_identify_file`** specifies the full path to the client private key to be used in that SCP file transfer.
10. **`DEBUG_MODE`** For verbose printing of what SCAWATCH is doing, mainly for debugging purposes by the developers. Should be 0 by default, unless you want to debug things and send to us. 



### Some Technical Details.
Process Monitor logs traces in a native file format called PML. After trace batch is completed (based on user-specified size_limit), our tool converts the resulting PML file to .CSV, and then Zips it. The CSV files are about 25% of the PML logs (i.e., much reduced). We also delete the converted PML files. We will add to this details as we get more comments.

# SCAWATCH

## Automated Dynamic Analysis of SCADA Phyiscal-Targeted Software Execution

### Host Requrements

This tool, SCAWATCH, uses a Windows utility called **Process Monitor (Procmon)** to trace software processes. Procmon can be downloaded here. https://learn.microsoft.com/en-us/sysinternals/downloads/procmon. 

SCAWATCH is **python-based** and executed in **Powershell**. Powershell must be run as **administrator** (to allow Procmon to work). Tested on Windows 7.

### Basic Steps:
1. `git clone https://github.com/lordmoses/SCAWATCH.git`
2. `cd SCADA-Dynamic-Analysis`
3. `pip install -r requirements.txt`
4. Edit config.json (to suit your Windows environment, see instructions below)
5. `python orchestrate.py`
6. To end things, press CTRL -C


### SCAWATCH performs the following:
1.	Automatically traces the system-level software execution (e.g., library/API calls) of a running software process.
2.	Automatically manages the collection and storage of the traces (or logs) based on user-specification. Traces are collected in batches of user-specified sizes and zipped.
3.	Automatically analyzes the traces for anomalies (e.g., injected attacks). Anomaly detection is based on statistical analysis of the frequency and timing behavior of SCADA “physical-targeted” executions (Not yet integrated in this code release).

A Windows VM with a proof-of-concept demonstration is provided (via special request). It runs an open-source SCADA suite, called ScadaBR.

### User Configuration (config.json)
1. **`size_limit`** refers to the file size (in megabytes) the Procmon logs are allowed to reached before it is zipped and stored. 
2. **`check_interval`** is the time in seconds our daemon has to poll to check if the file size limit is reached. 5 seconds is fairly good, but use your judgement based on your environment.
3. **`procmon_location`** refer to the absolute path where your procmon.exe executable resides on your windows machine
4. **`scada_process`** refer to the process name you want to trace. To know this, first run your software and use task manager to know the process name. E.g., process name of chrome is "chrome.exe"


### Some Technical Details
Process Monitor logs traces in a native file format called PML. After trace batch is completed (based on user-specified size_limit), our tool converts the resulting PML file to .CSV, and then Zips it. The CSV files are about 25% of the PML logs (i.e., much reduced). We also delete the converted PML files. 

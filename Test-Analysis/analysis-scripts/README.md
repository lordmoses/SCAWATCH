# Scripts

Assumes that ScadaBR has been run and captured API calls with Process Monitor.

## Run Experiment

### Parse ScadaBR.exe API calls into traces

Run `find_write_api_sequence.py`. It takes `ScadaBREXE.CSV` as input, which contains all API calls executed by ScadaBR, captured by Process Monitor. It outputs two files.
- `WRITE_API_Sequences.txt`, which is the trace file. Each line starts from TCP SEND (WRITE) and ends with the next TCP SEND (WRITE), including their timestamps.
- `All_TCP_Send_and_Receive_Sequences.txt`, which contains all TCP SEND and TCP RECEIVE API calls, including their timestamps. and TCP data length. The reason we need to have this is to identify READ and WRITE commands. For the WRITE commands, the TCP data length is the same for Send and Receive.

```python
python3 find_write_api_sequence.py
```

### Timing and Frequency Analysis

Run `timing_and_frequency_analysis.py`. It takes two files, `All_TCP_Send_and_Receive_Sequences.txt` and `WRITE_API_Sequences.txt` as input, which are the outputs of `find_write_api_sequence.py`. It prints out the timing and frequency analysis on the console, including WRITE_TO_WRITE, READ_TO_WRITE, READ_TO_READ, and Cycle Frequencies (Use a WRITE TO WRITE as a cycle). It contains their mean value and standard deviation.

```python
python3 timing_and_frequency_analysis.py
```


### Extract ScadaBR Dependencies API calls

Run `get_scadabr_and_dependency_api_calls.py`. It takes `All_API_Calls.CSV` as input, which contains all the API calls captured by Process Monitor. It outputs `ScadaBR_Dependencies.csv`, which contains all the ScadaBR dependent API calls.

```python
python3 get_scadabr_and_dependency_api_calls.py
``` 

## Run Attack Experiment

1. Clone the ScadaBR repository. (https://github.com/ScadaBR/ScadaBR)
2. Install Eclipse (version 2019-09) and JDK 8 to compile the ScadaBR project.
3. Modify the file `https://github.com/ScadaBR/ScadaBR/blob/master/src/com/serotonin/mango/rt/dataSource/modbus/ModbusDataSource.java`, and compile it with Eclipse. It will output a file `ModbusDataSource.class`. The attack source code can be found in `Attack_Scripts/ModbusDataSource.java`.
4. Replace the compiled `ModbusDataSource.class` with the current installed ScadaBR file `C:\Program Files\ScadaBR\tomcat\webapps\ScadaBR\WEB-INF\classes\com\serotonin\mango\rt\dataSource\modbus\ModbusDataSource.class`.
5. Put the attack config `Attack_Scripts/AttackConfig.txt` into `C:\Program Files\ScadaBR\tomcat\conf\AttackConfig.txt`. `ModbusDataSource.class` will read this config file to perform attacks.
6. Restart ScadaBR, the attack log will be created in `C:\Program Files\ScadaBR\tomcat\logs\AttackLog.txt`.
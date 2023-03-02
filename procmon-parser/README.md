# Parsing Procmon files with Python

[![Build Status](https://github.com/eronnen/procmon-parser/actions/workflows/python-package.yml/badge.svg)](https://github.com/eronnen/procmon-parser/actions)
[![Coverage Status](https://coveralls.io/repos/github/eronnen/procmon-parser/badge.svg?branch=master&service=github)](https://coveralls.io/github/eronnen/procmon-parser?branch=master&service=github)
[![PyPI version](https://badge.fury.io/py/procmon-parser.svg)](https://badge.fury.io/py/procmon-parser)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/procmon-parser)

Procmon (https://docs.microsoft.com/en-us/sysinternals/downloads/procmon) is a very powerful monitoring tool for Windows,
capable of capturing file system, registry, process/thread and network activity. 

Procmon uses internal file formats for configuration (**PMC**) and logs (**PML**). 
Prior to ``procmon-parser``, **PMC** files could only be parsed and generated by the Procmon GUI, and **PML** files 
could be read only using the Procmon GUI, or by converting them to CSV or XML using Procmon command line.

The goals of `procmon-parser` are:
* Parsing & Building **PMC** files - making it possible to dynamically add/remove filter rules, which can significantly
reduce the size of the log file over time as Procmon captures millions of events.
* Parsing **PML** files - making it possible to directly load the raw **PML** file into convenient python objects
instead of having to convert the file to CSV/XML formats prior to loading.


## PMC (Process Monitor Configuration) Parser

### Usage

Loading configuration of a pre-exported Procmon configuration:
```python
>>> from procmon_parser import load_configuration, dump_configuration, Rule
>>> with open("ProcmonConfiguration.pmc", "rb") as f:
...     config = load_configuration(f)
>>> config["DestructiveFilter"]
0
>>> config["FilterRules"]
[Rule(Column.PROCESS_NAME, RuleRelation.IS, "System", RuleAction.EXCLUDE), Rule(Column.PROCESS_NAME, RuleRelation.IS, "Procmon64.exe", RuleAction.EXCLUDE), Rule(Column.PROCESS_NAME, RuleRelation.IS, "Procmon.exe", RuleAction.EXCLUDE), Rule(Column.PROCESS_NAME, RuleRelation.IS, "Procexp64.exe", RuleAction.EXCLUDE), Rule(Column.PROCESS_NAME, RuleRelation.IS, "Procexp.exe", RuleAction.EXCLUDE), Rule(Column.PROCESS_NAME, RuleRelation.IS, "Autoruns.exe", RuleAction.EXCLUDE), Rule(Column.OPERATION, RuleRelation.BEGINS_WITH, "IRP_MJ_", RuleAction.EXCLUDE), Rule(Column.OPERATION, RuleRelation.BEGINS_WITH, "FASTIO_", RuleAction.EXCLUDE), Rule(Column.RESULT, RuleRelation.BEGINS_WITH, "FAST IO", RuleAction.EXCLUDE), Rule(Column.PATH, RuleRelation.ENDS_WITH, "pagefile.sys", RuleAction.EXCLUDE), Rule(Column.PATH, RuleRelation.ENDS_WITH, "$Volume", RuleAction.EXCLUDE), Rule(Column.PATH, RuleRelation.ENDS_WITH, "$UpCase", RuleAction.EXCLUDE), Rule(Column.PATH, RuleRelation.ENDS_WITH, "$Secure", RuleAction.EXCLUDE), Rule(Column.PATH, RuleRelation.ENDS_WITH, "$Root", RuleAction.EXCLUDE), Rule(Column.PATH, RuleRelation.ENDS_WITH, "$MftMirr", RuleAction.EXCLUDE), Rule(Column.PATH, RuleRelation.ENDS_WITH, "$Mft", RuleAction.EXCLUDE), Rule(Column.PATH, RuleRelation.ENDS_WITH, "$LogFile", RuleAction.EXCLUDE), Rule(Column.PATH, RuleRelation.CONTAINS, "$Extend", RuleAction.EXCLUDE), Rule(Column.PATH, RuleRelation.ENDS_WITH, "$Boot", RuleAction.EXCLUDE), Rule(Column.PATH, RuleRelation.ENDS_WITH, "$Bitmap", RuleAction.EXCLUDE), Rule(Column.PATH, RuleRelation.ENDS_WITH, "$BadClus", RuleAction.EXCLUDE), Rule(Column.PATH, RuleRelation.ENDS_WITH, "$AttrDef", RuleAction.EXCLUDE), Rule(Column.EVENT_CLASS, RuleRelation.IS, "Profiling", RuleAction.EXCLUDE)]
```

Adding some new rules
```python
>>> new_rules = [Rule('PID', 'is', '1336', 'include'), Rule('Process_Name', 'contains', 'python')]
>>> config["FilterRules"] = new_rules + config["FilterRules"]
```

Dropping filtered events
```python
>>> config["DestructiveFilter"] = 1
```

Dumping the new configuration to a file
```python
>>> with open("ProcmonConfiguration1337.pmc", "wb") as f:
...     dump_configuration(config, f)
```

### File Format

For the raw binary format of PMC files you can refer to the [docs](docs/PMC%20Format.md), or take a look at the source code in [configuration_format.py](procmon_parser/configuration_format.py).

## PML (Process Monitor Log) Parser

### Usage

`procmon-parser` exports a `ProcmonLogsReader` class for reading logs directly from a PML file:
```python
>>> from procmon_parser import ProcmonLogsReader
>>> f = open("LogFile.PML", "rb")
>>> pml_reader = ProcmonLogsReader(f)
>>> len(pml_reader)  # number of logs
53214

>>> first_event = next(pml_reader)  # reading the next event in the log
>>> print(first_event)
Process Name=dwm.exe, Pid=932, Operation=RegQueryValue, Path="HKCU\Software\Microsoft\Windows\DWM\ColorPrevalence", Time=7/12/2020 1:18:10.7752429 AM

>>> print(first_event.process)  #  Accessing the process of the event
"C:\Windows\system32\dwm.exe", 932
>>> for module in first_event.process.modules[:3]:
...     print(module)  # printing information about some modules
"C:\Windows\system32\dwm.exe", address=0x7ff6fa980000, size=0x18000
"C:\Windows\system32\d3d10warp.dll", address=0x7fff96700000, size=0x76c000
"C:\Windows\system32\wuceffects.dll", address=0x7fff9a920000, size=0x3f000

>>> first_event.stacktrace  # get a list of the stack frames addresses from the event
[18446735291098361031, 18446735291098336505, 18446735291095097155, 140736399934388, 140736346856333, 140736346854333, 140698742953668, 140736303659045, 140736303655429, 140736303639145, 140736303628747, 140736303625739, 140736303693867, 140736303347333, 140736303383760, 140736303385017, 140736398440420, 140736399723393]
>>>
```

### File Format

For the raw binary format of PML files you can refer to the [docs](docs/PML%20Format.md), or take a look at the source code in [stream_logs_format.py](procmon_parser/stream_logs_format.py).

Currently the parser is only tested with PML files saved by *Procmon.exe* of versions v3.4.0 or higher.

### TODO

The PML format is very complex so there are some features (unchecked in the list) that are not supported yet:
- [ ] Getting the IRP name of the operation.
- [ ] Category column and Detail column, which contains different information about each operation type, is supported only for some of the operations:
    - [x] Network operations 
        - [x] UDP/TCP Unknown
        - [x] UDP/TCP Other
        - [x] UDP/TCP Send
        - [x] UDP/TCP Receive
        - [x] UDP/TCP Accept
        - [x] UDP/TCP Connect
        - [x] UDP/TCP Disconnect
        - [x] UDP/TCP Reconnect
        - [x] UDP/TCP Retransmit
        - [x] UDP/TCP TCPCopy
    - [ ] Process operations
        - [x] Process Defined
        - [x] Process Create
        - [x] Process Exit
        - [x] Thread Create
        - [x] Thread Exit
        - [x] Load Image
        - [x] Thread Profile
        - [x] Process Start
        - [x] Process Statistics
        - [ ] System Statistics
    - [x] Registry operations
        - [x] RegOpenKey
        - [x] RegCreateKey
        - [x] RegCloseKey
        - [x] RegQueryKey
        - [x] RegSetValue
        - [x] RegQueryValue
        - [x] RegEnumValue
        - [x] RegEnumKey
        - [x] RegSetInfoKey
        - [x] RegDeleteKey
        - [x] RegDeleteValue
        - [x] RegFlushKey
        - [x] RegLoadKey
        - [x] RegUnloadKey
        - [x] RegRenameKey
        - [x] RegQueryMultipleValueKey
        - [x] RegSetKeySecurity
        - [x] RegQueryKeySecurity
    - [ ] Filesystem Operations
        - [ ] VolumeDismount
        - [ ] VolumeMount
        - [x] CreateFileMapping
        - [x] CreateFile
        - [ ] CreatePipe
        - [x] ReadFile
        - [x] WriteFile
        - [ ] QueryInformationFile
        - [ ] SetInformationFile
        - [ ] QueryEAFile
        - [ ] SetEAFile
        - [ ] FlushBuffersFile
        - [ ] QueryVolumeInformation
        - [ ] SetVolumeInformation
        - [x] DirectoryControl
        - [x] FileSystemControl
        - [x] DeviceIoControl
        - [x] InternalDeviceIoControl
        - [ ] Shutdown
        - [ ] LockUnlockFile
        - [x] CloseFile
        - [ ] CreateMailSlot
        - [ ] QuerySecurityFile
        - [ ] SetSecurityFile
        - [ ] Power
        - [ ] SystemControl
        - [ ] DeviceChange
        - [ ] QueryFileQuota
        - [ ] SetFileQuota
        - [ ] PlugAndPlay
    - [ ] Profiling Operations
        - [ ] Thread Profiling
        - [ ] Process Profiling
        - [ ] Debug Output Profiling

These are a lot of operation types so I didn't manage to get to all of them yet :(<br/>
If there is an unsupported operation which you think its details are interesting, please let me know :) 

### Tests

To test that the parsing is done correctly, There are two fairly large Procmon PML files and their respective CSV format
log files, taken from 64 bit and 32 bit machine. The test checks that each event in the PML parsed by ``procmon-parser``
equals to the respective event in the CSV. 

## Contributing

`procmon-parser` is developed on GitHub at [eronnen/procmon-parser](https://github.com/eronnen/procmon-parser).
Feel free to report an issue or send a pull request, use the
[issue tracker](https://github.com/eronnen/procmon-parser/issues).
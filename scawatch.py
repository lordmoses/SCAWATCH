from datetime import datetime
import time
import os, sys
import zipfile
from zipfile import ZipFile
import subprocess
import argparse
import psutil
from pathlib import Path
import json

#Adding an external tool that can make Procmon config on the fly
#https://github.com/eronnen/procmon-parser 
sys.path.insert(0, "./procmon-parser")
from procmon_parser import load_configuration, dump_configuration, Rule

with open('config.json') as config_file:
    data = json.load(config_file)


#USER CONFIGURATIONS : 
SIZE_LIMIT = data["size_limit"]                 #default storage size at which we wish to stop writing the log
CHECK_INTERVAL = data["check_interval"]          #default time interval to sleep between size checks
#default_config_file = data["procmon_config_file"]    #Default PROCMON configuration file
procmon_location = data["procmon_location"]    #default procmon location
scada_process_name = data["scada_process"]    #default procmon location

print()
print("SCAWATCH v1.1")
print("==User Specification==")
print("Procmon Location:", procmon_location)
print("SCADA software process investigated:", scada_process_name)
print("Log batch size in MB:", SIZE_LIMIT)
print("==End==")

print("Making procmon configuration and filters ...")

#Lets make the procmon filter/config called "added_config.pmc"
blank_config_file_name = "fresh_config.pmc" #located in the same folder as the script
new_config_file_name = "new_config.pmc" #new config that we will make

new_rule =  [Rule('Process_Name', 'contains', scada_process_name, 'include')] # New rule
#loading the blank config
with open(blank_config_file_name, "rb") as f:
    blank_config = load_configuration(f)  


blank_config["FilterRules"] = new_rule + blank_config["FilterRules"] #Adding our rule
#blank_config["FilterRules"] = new_rule #Replacing with our one rule
#Lets specify to DROP FILTERED EVENTS. So that Procmon don't store all events in the PML File. Efficeint
blank_config["DestructiveFilter"] = 1
with open(new_config_file_name, "wb") as f:
    dump_configuration(blank_config, f)      #Saving the filter/config to a file

CONFIG_FILE = os.getcwd() + "\\" + new_config_file_name
print("Starting Tracing ...")
print()


"""""
#Setting up command line integration
parser = argparse.ArgumentParser()
parser.add_argument("-s","--SIZE_LIMIT", default = default_size, required = False, help = "Size limit in megabytes", type = float)
parser.add_argument("-i", "--CHECK_INTERVAL", default = default_check_interval, required = False, help = "Check interval in seconds", type = int)
parser.add_argument("-c", "--CONFIG_FILE", default = default_config_file, required = False, help = "Location to  .pmc file", type = str)
args = parser.parse_args()
"""



log_dir = "logs\\"

#Make logs folder, skip if it exists
try:
    os.mkdir(log_dir)
except FileExistsError:
    pass

global user_terminate_action #This is to enable graceful or ungraceful shutdown of the program
user_terminate_action = 0

debug_mode = False

def check_size(file :str , size_limit : int):

    ## Check if current size is greater than required
    if os.path.getsize(file) > size_limit * 1_000_000:
        return True
    else:
        False


def ensure_file_is_longer_being_used(file_paths, max_wait_time):
    ## Function to wait until procmon is done saving the CSV file completely
    #lets grab the processes we are interested in (namely the procmon we ran)
    processes_of_interest = []
    for proc in psutil.process_iter(['name', 'cmdline']):
        if "procmon" in str(proc.info['name']).lower() and "openlog" in str(proc.info['cmdline']).lower():
            processes_of_interest.append(proc)
     
    if len(processes_of_interest) == 0:
        debug_output("No process was found")
        return
    

    start_time = time.time()
    #We observe that procmon may not yet have opened the pml/csv files for processing, so Lets get a minimum time to allow this to happen, i.e., so we don't leave too quickly and assume it is done opening/closing the files 
    minimum_time_to_access_file = 5 # seconds
    file_has_been_determined_to_be_accessed  = False
    while True:
        if (time.time() - start_time) > max_wait_time:
            print("***WARNING. UNRESPONSIVE Procmon process. Found during checking CSV Conversion. Force-Terminating. CSV Logs may be corrupted", file_paths)
            force_kill(processes_of_interest,"2b")
            return False
        else:
            open_files = "" #lets use open files to see if the processes are still running. It should have the PML open
            for proc in processes_of_interest:
                try:
                    #if "running" in str([proc.status() for proc in tracing_procs]).lower()
                    # if pml_log in str([str(proc.open_files()) for proc in tracing_procs])
                    open_files += str(proc.open_files())
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    debug_output("exception hit 2c")            
            if  file_paths[0].split("\\")[-1] in open_files or file_paths[1].split("\\")[-1] in open_files:
                    file_has_been_determined_to_be_accessed = True
                    time.sleep(1)
                    debug_output("waiting for open file to be released")
            else: #they don't show as running, but let me still try to kill them
                if (not file_has_been_determined_to_be_accessed) and (time.time() - start_time) < minimum_time_to_access_file:
                    continue
                else:
                    #We assume procmon successfuly accesed the files and existed nicely
                    force_kill(processes_of_interest,"2d") #this is not intended to address e.g., idled processes
                    break
    return True

def ensure_procmon_is_done_tracing(pml_log, max_wait_time):
    #Get the processes in question
    tracing_procs = [] #Made it a list because since Procmon.exe may be a wrapper on Procmon64.exe
    terminating_proc = False
    for proc in psutil.process_iter(['name','cmdline']):
        if "procmon" in str(proc.name).lower():
            cmdline = str(proc.info['cmdline'])
            if ("loadconfig" in cmdline.lower()):
                tracing_procs.append(proc)
            elif ("terminate" in cmdline.lower()):
                terminating_proc = proc
    
    if len(tracing_procs) == 0: #the procmon that is tracing is completed/done, yay !
        force_kill([terminating_proc],"1a")
        return

    start_time = time.time()
    while True:  
        if (time.time() - start_time) > max_wait_time:
            print("***WARNING. Unresponsive Procmon process. Found during checking Procmon to terminate Tracing. Force-Terminating. PML Logs may be corrupted", pml_log)
            force_kill(tracing_procs + [terminating_proc],"1b")
            break 
        else:
            open_files = "" #lets use open files to see if the processes are still running. It should have the PML open
            for proc in tracing_procs:
                try:
                    #if   "running" in str([proc.status() for proc in tracing_procs]).lower()
                    # if pml_log in str([str(proc.open_files()) for proc in tracing_procs])
                    open_files += str(proc.open_files())
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    debug_output("exception hit 1c")            
            if  pml_log.split("\\")[-1] in open_files:
                    time.sleep(1)
                    debug_output("waiting for the tracing to let go of the pml log file")
            else: #they don't show as running, but let me still try to kill them
                force_kill(tracing_procs + [terminating_proc],"1d") #remove and test, not neccessary
                break

def force_kill(process_list, msg):
    debug_output("attempting to force kill ", str(process_list))
    for proc in process_list:
        try:
            proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            debug_output("exception hit force kill", msg)  



def setup_and_trace():
    time_label =  str(datetime.strftime(datetime.now(),"%H-%M-%S-%m-%d-%Y"))
    CURRENT_PML_LOG = log_dir + 'logfile-' + time_label + ".pml"
    CURRENT_CSV_LOG = CURRENT_PML_LOG.replace('.pml','.csv')
    CURRENT_ZIP_NAME =  CURRENT_CSV_LOG.replace('.csv', '.zip')

    
    ## Starting run_procmon.ps1
    run_procmon = subprocess.Popen(['powershell.exe', '-File','run_procmon.ps1', procmon_location, CONFIG_FILE, CURRENT_PML_LOG])
    
    log_file = Path(CURRENT_PML_LOG)
    
    while not log_file.exists():    #This loop is used to verify that Procmon has started logging to the file
        time.sleep(.5)
    
    return CURRENT_PML_LOG, CURRENT_CSV_LOG, CURRENT_ZIP_NAME

## This function watches the file size over Procmon tracing instance and when it reaches the size limit, it stops it 
def watch_and_stop(size_limit, check_interval, CURRENT_PML_LOG, CURRENT_CSV_LOG, CURRENT_ZIP_NAME):
    while not check_size(CURRENT_PML_LOG, size_limit):  #This loop checks if the log file has reached its limit
        time.sleep(check_interval)
         
    ## Stopping tracing on Procmon
    stop_trace = subprocess.Popen(['powershell.exe', '-File', 'stop_trace.ps1', procmon_location])
    stop_trace.communicate()    #Some weird inter-process communication, between Python and PowerShell

    ensure_procmon_is_done_tracing(CURRENT_PML_LOG, 30) #two instances of procmon will be checked here: the one from run_procmon.ps1 and the stop_trace.ps1
    #may be force-killed if they become unresponsive after a max_wait_time

    COMPLETED_PML_LOG = CURRENT_PML_LOG
    COMPLETED_CSV_LOG = CURRENT_CSV_LOG
    COMPLETED_ZIP_NAME = CURRENT_ZIP_NAME

    return COMPLETED_PML_LOG, COMPLETED_CSV_LOG, COMPLETED_ZIP_NAME


def convert_pml_to_csv_and_zip(PML_LOG, CSV_LOG, ZIP_NAME):
    debug_output("Converting PML to CSV", str(datetime.strftime(datetime.now(),"%H-%M-%S-%m-%d-%Y")))
    convert_pml_csv = subprocess.Popen(['powershell.exe', '-File', 'convert-pml-csv.ps1', procmon_location, PML_LOG, CSV_LOG])      
    convert_pml_csv.communicate()
    
    ## Make sure Procmon is done converting .pml to CSV and has let go of the CSV log
    debug_output("Ensuring CSV is no longer being converted by procmon", str(datetime.strftime(datetime.now(),"%H-%M-%S-%m-%d-%Y")))
    time.sleep(1) #One seconds grace to allow that powershell script to kick off.
    start_time = time.time()
    status = ensure_file_is_longer_being_used([CSV_LOG, PML_LOG], 60)
    #if not status: #something bad happenned e.g., procmon was forced killed so pml/csv is corrupted, or does not exist 
    try:
        debug_output("time taken", (time.time() - start_time), "sec . PML size ", os.stat(PML_LOG).st_size/(1024*1024), " MB. CSV size ", os.stat(CSV_LOG).st_size/(1024*1024), " MB")
        #ensure_file_is_longer_being_used(PML_LOG, 5)
        ## Zipping the logfile then removing the original logfile
        debug_output("Now zipping the file", str(datetime.strftime(datetime.now(),"%H-%M-%S-%m-%d-%Y")))
        zip_file = ZipFile(ZIP_NAME, 'w', zipfile.ZIP_DEFLATED)
        zip_file.write(CSV_LOG, arcname=os.path.basename(CSV_LOG))
        os.remove(PML_LOG)
        os.remove(CSV_LOG)
    except FileNotFoundError as e:
        print("***WARNING. POSSIBLE DATA LOSS", PML_LOG, "not converted to CSV. Corroborated due to previous step?: ", status)

def debug_output(*list_to_print):
    if debug_mode:
        print(str(list_to_print))

def end_analysis():
    ## Shutdown procmon instances upon Keyboard Interrupt
    for proc in psutil.process_iter():
        if "procmon" in str(proc.name()).lower():
            proc.kill()
    sys.exit()

def run(size_limit, check_interval):

    global user_terminate_action
    procmon_instantiation_counter = 1
    print("Procmon instance ", procmon_instantiation_counter, " Date-time:", str(datetime.strftime(datetime.now(),"%H-%M-%S-%m-%d-%Y")))
    #Start the first/current tracing
    current_pml_log, current_csv_log, current_zip_name = setup_and_trace()
    print("Logging trace in ", current_pml_log, "...")
    
    while True:
        try:
            debug_output("Watch and Stop... Date-Time: ", str(datetime.strftime(datetime.now(),"%H-%M-%S-%m-%d-%Y")), "pml_log", current_pml_log)    
            completed_pml_log, completed_csv_log, completed_zip_name = watch_and_stop(size_limit, check_interval, current_pml_log, current_csv_log, current_zip_name)
        
            if user_terminate_action == 0: #user has not asked to stop e.g., via CTRL C
                ##Starting the next tracing. We want to start tracing the next instance immediately after the first/current one finishes, i.e. before we start zipping its logs, to minimize missed logs/events
                print("Procmon instance ", procmon_instantiation_counter, " Date-time: ", str(datetime.strftime(datetime.now(),"%H-%M-%S-%m-%d-%Y")))
                current_pml_log, current_csv_log, current_zip_name = setup_and_trace()
                print("Logging trace in ", current_pml_log, "...")
                procmon_instantiation_counter += 1
                
            ## Make sure Procmon is done using the completed pml log
            #print("Ensuring PML is not longer in use", str(datetime.strftime(datetime.now(),"%H-%M-%S-%m-%d-%Y")))
            #ensure_file_is_longer_being_used(completed_pml_log, 10) #may not be neccessary
  
            ## Converted completed pml to csv and zip it
            convert_pml_to_csv_and_zip(completed_pml_log, completed_csv_log, completed_zip_name)
        
            if user_terminate_action == 1:
                print("Ending Gracefully ...")
                end_analysis()
            if user_terminate_action > 1:
                print("Will End UnGracefully. The last set of Logs may be corrupted.")
                end_analysis()

        except KeyboardInterrupt:
            user_terminate_action += 1
            print("Now gracefully shutting things down. You can wait for current trace batch to be completed, or press Ctrl C again to shut down now!. Current trace may be corrupted.")

            #Ungraceful shutdown since user pressed Ctrl C more than once
            if user_terminate_action > 1:
                print("Will End UnGracefully. The last set of Logs may be corrupted.")
                end_analysis()

run(SIZE_LIMIT, CHECK_INTERVAL)




    







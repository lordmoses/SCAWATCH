#!/usr/bin/env python3

"""
This script is to extract training data and timing frequency analysis data from CSV file extracted from Process Monitor

The input file (INPUT_FILE) is a CSV exported from Process Monitor. It contains all the API calls exectued by ScadaBR.exe
The first output file (OUTPUT_FILE_1) contains API sequences between every 2 WRITE calls.
Format [[Timestamp 1, API 1], [Timestamp 2, API 2], ..., [Timestamp N, API N]]
The second output file (OUTPUT_FILE_2) contains all "TCP Send" and "TCP Receive" sequences. It is used for timing and frequency analysis.
Format [[Timestamp 1, API 1, TCP Length 1], [Timestamp 2, API 2, Length 2], ..., [Timestamp N, API N, Length N]]
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import time

USE_TCP_RECEIVE_HEURISTIC = 1
USE_TIME_DB_READ_HEURISTIC = 0
INPUT_FILE = 'input_data/ScadaBREXE.CSV'
OUTPUT_FILE_1 = 'output_data/WRITE_API_Sequences.txt'
OUTPUT_FILE_2 = 'output_data/All_TCP_Send_and_Receive_Sequences.txt'

pd.set_option('mode.chained_assignment', None)
df = pd.read_csv(INPUT_FILE)



# Export OUTPUT_FILE_1
def export_api_and_timecode_between_writes(write_time, scada_df):
    
    f = open(OUTPUT_FILE_1, 'wb')
    for i in range(len(write_time) - 1):
        # Extract APIs and timecode between 2 WRITEs (include write at the beginning and end)
        mask = ((scada_df['Time of Day'] >= (write_time[i])) &\
            (scada_df['Time of Day'] <= (write_time[i + 1])))
        tmp = scada_df[mask]

        # Drop irrelevant columns
        result = tmp.drop(['Process Name', 'Result', 'Detail', 'Path', 'index'], axis=1)
        #print(f"\n\n[+] Part of API Sequence {i+1}")
        
        # Change Timestamp to microseconds
        result['Time of Day'] = result['Time of Day'].values.astype(np.int64) // 1000
        
        result_list = result.values.tolist()
        #print(f"{result_list[0:5]} \n...\n {result_list[-5:]}")

        # Save to file
        f.write(str(result_list).replace('\'', '"').encode() + b'\n')
    f.close()

# Export OUTPUT_FILE_2
def export_all_tcp_send_and_receive(scada_df):
    # Find all API calls either TCP Send or TCP Receive
    scada_tcp_send_recv_df = scada_df[((scada_df['Operation'] == 'TCP Send') | (scada_df['Operation'] == 'TCP Receive'))  & (scada_df['Path'].str.contains(':502'))]
    scada_tcp_send_recv_df = scada_tcp_send_recv_df.reset_index(drop=True).drop(['index'], axis=1)

    # Change Timestamp to microseconds
    scada_tcp_send_recv_df['Time of Day'] = scada_tcp_send_recv_df['Time of Day'].values.astype(np.int64) // 1000
    
    # Prepare the output 
    tcp_send_recv = []
    for index, row in scada_tcp_send_recv_df.iterrows():
        rec = [row["Time of Day"], row["Operation"], row['Detail'].split(',')[0].split('Length: ')[1]]
        tcp_send_recv.append(rec)
        
    # Save to file
    with open(OUTPUT_FILE_2, 'wb') as f:
        f.write(str(tcp_send_recv).replace('\'', '"').encode())
    
# Show the comparison between 2 heuristics
def show_correctness_rate(write_time, write_time_2):
    # write_time is extracted from USE_TIME_DB_READ_HEURISTIC
    # write_time_2 is extracted from USE_TCP_RECEIVE_HEURISTIC
    setA = set(write_time)
    setB = set(write_time_2)
    
    correctness_rate = float(len(setA & setB)) / len(setB) * 100
    false_positive_rate = float(len(setA - setB)) / len(setB) * 100

    print("USE_TIME_DB_READ_HEURISTIC Statistics")
    print(f"\tCorrectness Rate: {round(correctness_rate, 2)}%")
    print(f"\tFalse Positive Rate: {round(false_positive_rate, 2)}%")


scada_df = df.reset_index()
scada_df['Time of Day'] = pd.to_datetime(scada_df['Time of Day'])

# Identify WRITE Commands with TIME_DB_READ_HEURISTIC
write_time = []
if USE_TIME_DB_READ_HEURISTIC:
    DB_CHECK_LINES = 300
    DB_CHECK_THRESHOLD = 8
    THREAD_EXIT_CHECK_LINES = 10
    THREAD_EXIT_USAGE_TIME = 0.1
    NORMAL_DB_CHECK_TIME = 0.01

    print("[+] Running USE_TIME_DB_READ_HEURISTIC Approach ...", end='', flush=True)
    start = time.time()

    # Find all TCP Send to Modbus
    scada_tcp_send_df = scada_df[(scada_df['Operation'] == 'TCP Send') & (scada_df['Path'].str.contains(':502'))]

    for index, row in scada_tcp_send_df.iterrows():
        
        # Get how many scadabrDB checks in DB_CHECK_LINES lines before TCP Send
        mask = ((scada_df.index < index) &\
                (scada_df.index > (index - DB_CHECK_LINES)) &\
                (scada_df['Path'] == 'C:\Program Files\ScadaBR\\tomcat\webapps\ScadaBR\db\scadabrDB'))
        tmp = scada_df[mask]

        # If there are more than DB_CHECK_THRESHOLD checks in DB_CHECK_LINES
        # If true, we check the first scadabrDB check is TIME_THRESHOLD before TCP Send
        # If true, we classify it as a WRITE call
        if tmp.shape[0] > DB_CHECK_THRESHOLD:
            
            
            # Get the time of last scadabrDB check
            t_delta = (row['Time of Day'] - tmp.iloc[-1:]['Time of Day']).values[0] / np.timedelta64(1, 's')
            
            # Check if "Thread Exit" is in previous THREAD_EXIT_CHECK_LINES
            mask = ((scada_df.index < index) &\
                (scada_df.index > (index - THREAD_EXIT_CHECK_LINES)) &\
                (scada_df['Operation'] == 'Thread Exit'))
            thread_exit = scada_df[mask]

            t = NORMAL_DB_CHECK_TIME
            # Check if there is Thread Exit before TCP Send
            # We add more time (THREAD_EXIT_TIME seconds) to NORMAL_DB_CHECK_TIME if there is "Thread Exit" before "TCP Send"
            if thread_exit.shape[0] != 0:
                t += THREAD_EXIT_USAGE_TIME

            # If scadabrDB check time is close enough to TCP Send, we said this is a WRITE call
            if t > t_delta:
                write_time.append(row['Time of Day'])

    print(" Finished!!")
    end = time.time()
    print(f"It takes {round(end - start, 2)} seconds")
    
# Identify WRITE Commands with TCP_RECEIVE_HEURISTIC
# Check if length of TCP Send is equal to length of TCP Receive
write_time_2 = []
if USE_TCP_RECEIVE_HEURISTIC:
    print("[+] Running USE_TCP_RECEIVE_HEURISTIC Approach ...", end='', flush=True)
    start = time.time()

    # Find all TCP Send to Modbus
    scada_tcp_send_recv_df = scada_df[((scada_df['Operation'] == 'TCP Send') | (scada_df['Operation'] == 'TCP Receive'))  & (scada_df['Path'].str.contains(':502'))]

    # Drop irrelevant columns
    scada_tcp_send_recv_df = scada_tcp_send_recv_df.reset_index(drop=True).drop(['index'], axis=1)
    
    i = 0
    # Look for TCP Send and TCP Receive pairs
    while i < scada_tcp_send_recv_df.shape[0]:
        if scada_tcp_send_recv_df.iloc[i]['Operation'] == 'TCP Send':
            j = i + 1
            while scada_tcp_send_recv_df.iloc[j]['Operation'] != 'TCP Receive':
                j += 1
                # Reach the last API call, exit
                if j == scada_tcp_send_recv_df.shape[0]:
                    break

            # Extract lengths from TCP Send and TCP Receive pairs
            tcp_send_length = scada_tcp_send_recv_df.iloc[i]['Detail'].split(',')[0].split('Length: ')[1]
            tcp_recv_length = scada_tcp_send_recv_df.iloc[j]['Detail'].split(',')[0].split('Length: ')[1]
            
            # If length of TCP Send and TCP Receive match, it is a WRITE call
            if tcp_send_length == tcp_recv_length:
                write_time_2.append(scada_tcp_send_recv_df.iloc[i]['Time of Day'])
            
            # Starts from the next API call after the last TCP Receive
            i = j + 1

    print(" Finished!!")
    end = time.time()
    print(f"It takes {round(end - start, 2)} seconds")

# Show the performance by comparing the two heuristics
if USE_TCP_RECEIVE_HEURISTIC and USE_TIME_DB_READ_HEURISTIC:
    show_correctness_rate(write_time, write_time_2)

# Use the result from USE_TCP_RECEIVE_HEURISTIC if we activates it, instead of the result from USE_TIME_DB_READ_HEURISTIC
if USE_TCP_RECEIVE_HEURISTIC:
    write_time = write_time_2

print(f"\nThere are {len(write_time)} WRITE commands in total\n")

print("")

print("[+] Exporting Files ...", end='', flush=True)
start = time.time()

# Export OUTPUT_FILE_1
export_api_and_timecode_between_writes(write_time, scada_df)

# Export OUTPUT_FILE_2. It is useful for timing and frequency analysis
export_all_tcp_send_and_receive(scada_df)

print(" Finished!!")
end = time.time()
print(f"It takes {round(end - start, 2)} seconds")
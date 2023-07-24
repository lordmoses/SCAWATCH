#!/usr/bin/env python3

"""
This script is used to provide timing and frequency analysis, including count, mean, and standard deviation
of WRITE to WRITE, READ to WRITE, and READ to READ

The first input file (INPUT_FILE_1) is a text file containing all TCP Send, TCP Receive, and their timestamps
The second input file (INPUT_FILE_2) is a text file containing API sequences between every 2 WRITE calls.
Both files are generated from find_write_api_sequence.py
"""

import json

INPUT_FILE_1 = 'output_data/All_TCP_Send_and_Receive_Sequences.txt'
INPUT_FILE_2 = 'output_data/WRITE_API_Sequences.txt'

# Change the format of INPUT_FILE_1 from 
# [[Timestamp 1, API 1, TCP Length 1], [Timestamp 2, API 2, Length 2], ..., [Timestamp N, API N, Length N]]
# to
# [[Timestamp 1, "READ"], [Timestamp 2, "WRITE"]]
def get_tcp_read_write(input_file):
    
    with open(input_file, 'rb') as f:
        tcp_api_sequences = json.load(f)
    
    send_length = ''
    send_time = 0
    look_for_receive = 0
    tcp_read_write = []
    for i in range(len(tcp_api_sequences)):
        # Look for TCP Receive
        if look_for_receive:
            if tcp_api_sequences[i][1] == "TCP Receive":
                # If length of TCP Send is the same as length of TCP Receive, it is a WRITE call
                if tcp_api_sequences[i][2] == send_length:
                    tcp_read_write.append([send_time, "WRITE"])
                # Else, it is a READ call
                else:
                    tcp_read_write.append([send_time, "READ"])
                look_for_receive = 0
            elif tcp_api_sequences[i][1] == "TCP Send":
                send_time = tcp_api_sequences[i][0]
                send_length = tcp_api_sequences[i][2]
        # Look for TCP Send
        else:
            if tcp_api_sequences[i][1] == "TCP Send":
                send_time = tcp_api_sequences[i][0]
                send_length = tcp_api_sequences[i][2]
                look_for_receive = 1
    return tcp_read_write

# Output number of API calls between every 2 WRITE calls
# Format: [10934, 1885, 2191, 518, 3495, ..., 3491]
def get_number_of_all_scada_apis_between_writes():
    result = []
    with open(INPUT_FILE_2, 'rb') as f:
        lines = f.readlines()
        for line in lines:
            result.append(len(json.loads(line)))
            
    return result

# Get indexes of a specific element from a list
def indexes(iterable, obj):
    result = []
    for index, elem in enumerate(iterable):
        if elem == obj:
            yield index


# Get number of WRITE and READ between 2 WRITE calls
# Format: [6, 4, 4, 2, 8, 4, 2, 4, 8, 4, 2, 4, 9, 4, ..., 4, 2]
def get_number_of_write_plus_read_apis(tcp_read_write):
    tmp = []
    # Discard timestamp
    for sequence in tcp_read_write:
        tmp.append(sequence[1])
    
    # Get indexes of WRITE in tcp_read_write
    idxs = list(indexes(tmp, "WRITE"))
    result = [idxs[i+1] - idxs[i] + 1 for i in range(len(idxs) - 1)]
    
    return result

# Calculate Mean
def calculate_mean(arr):
    return sum(arr) / len(arr)

# Calculate Standard Deviation
def calculate_standard_deviation(arr, mean):
    variance = sum([((x - mean) ** 2) for x in arr]) / len(arr)
    return variance ** 0.5

# Preprocess for READ to WRITE and READ to READ
tcp_read_write = get_tcp_read_write(INPUT_FILE_1)


# WRITE to WRITE
write_to_write = []
last_write_time = 0
for sequence in tcp_read_write:
    if sequence[1] == "WRITE":
        if last_write_time != 0:
            write_to_write.append(sequence[0] - last_write_time)
        last_write_time = sequence[0]
    
mean = calculate_mean(write_to_write)
standard_deviation = calculate_standard_deviation(write_to_write, mean)
print("Write_to_Write_Timing:")
print(f"\tCount: {len(write_to_write)} in total")
print(f"\tMean: {round(mean / 1000, 2)} milliseconds")
print(f"\tStandard Deviation: {round(standard_deviation / 1000, 2)} milliseconds")

# READ to WRITE
read_to_write = []
for i in range(len(tcp_read_write) - 1):
    if tcp_read_write[i][1] == "READ" and tcp_read_write[i+1][1] == "WRITE":
        read_to_write.append(tcp_read_write[i+1][0] - tcp_read_write[i][0])
        

mean = calculate_mean(read_to_write)
standard_deviation = calculate_standard_deviation(read_to_write, mean)
print("Read_to_Write_Timing:")
print(f"\tCount: {len(read_to_write)} in total")
print(f"\tMean: {round(mean / 1000, 2)} milliseconds")
print(f"\tStandard Deviation: {round(standard_deviation / 1000, 2)} milliseconds")

# READ to READ
read_to_read = []
last_read_time = 0
for sequence in tcp_read_write:
    if sequence[1] == "READ":
        if last_read_time != 0:
            read_to_read.append(sequence[0] - last_read_time)
        last_read_time = sequence[0]


mean = calculate_mean(read_to_read)
standard_deviation = calculate_standard_deviation(read_to_read, mean)
print("Read_to_Read_Timing:")
print(f"\tCount: {len(read_to_read)} in total")
print(f"\tMean: {round(mean / 1000, 2)} milliseconds")
print(f"\tStandard Deviation: {round(standard_deviation / 1000, 2)} milliseconds")




# Use a cycle of a WRITE TO WRITE to calculate the frequencies
print("")
number_of_write_plus_read_apis = get_number_of_write_plus_read_apis(tcp_read_write)
number_of_all_scada_apis = get_number_of_all_scada_apis_between_writes()

# Check that the count is the same
assert len(number_of_write_plus_read_apis) == len(number_of_all_scada_apis)

# Calculate frequencies
# 1    No of WRITEs/(WRITE + READ)
# 2    No of WRITEs/(ALL SCADA APIs)​
# No of WRITES = 2​
# Frequency Sequences: (A1,A2), (B1,B2)………...,(N1, N2) ​
frequency_sequences = []
frequency_read_plus_write = [round(2/x, 8) for x in number_of_write_plus_read_apis]
frequency_all_scada_apis = [round(2/x, 8) for x in number_of_all_scada_apis]

for i in range(len(frequency_read_plus_write)):
    frequency_sequences.append((frequency_read_plus_write[i], frequency_all_scada_apis[i]))

# Get Mean = (mean1, mean2)
mean1 = calculate_mean(frequency_read_plus_write)
mean2 = calculate_mean(frequency_all_scada_apis)
mean = (round(mean1, 8), round(mean2, 8))

# Get SD = (SD1, SD2)
standard_deviation1 = calculate_standard_deviation(frequency_read_plus_write, mean1)
standard_deviation2 = calculate_standard_deviation(frequency_all_scada_apis, mean2)
standard_deviation = (round(standard_deviation1, 8), round(standard_deviation2, 8))

print("Cycle Frequencies (Use a WRITE TO WRITE as a cycle)")
print(f"\tCycle Count: {len(frequency_sequences)} in total")
#print(f"\tFrequency Sequences: {frequency_sequences}")
print(f"\tMean: {mean}")
print(f"\tStandard Deviation: {standard_deviation}")

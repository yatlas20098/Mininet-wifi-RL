#!/bin/bash

pcap_directory="/mydata/mydata/RL_agent/output"
output_directory="${pcap_directory}/extracted_data"

# Ensure the output directory exists
mkdir -p "$output_directory"

# Process pcap files from 0 to 10
for i in {0..10}; do
    pcap_file="${pcap_directory}/tcpdump_sender_sensor${i}.pcap"
    output_file="${output_directory}/tcpdump_output_sensor${i}.txt"
    if [[ -f "$pcap_file" ]]; then
        echo "Processing $pcap_file"
        sudo tcpdump -r "$pcap_file" > "$output_file"
    else
        echo "File $pcap_file does not exist."
    fi
done

# Process the additional pcap file
additional_pcap="${pcap_directory}/capture.pcap"
additional_output="${output_directory}/tcpdump_output_capture.txt"

if [[ -f "$additional_pcap" ]]; then
    echo "Processing $additional_pcap"
    sudo tcpdump -r "$additional_pcap" > "$additional_output"
else
    echo "File $additional_pcap does not exist."
fi

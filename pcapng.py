import os
import csv
import pyshark
import pandas as pd
import numpy as np
from datetime import datetime
import glob
from pathlib import Path
from collections import defaultdict
import ipaddress
import math

def is_private_ip(ip_str):
    """Check if an IP address is private."""
    try:
        ip = ipaddress.ip_address(ip_str)
        return ip.is_private
    except ValueError:
        return False

def extract_pcapng_to_csv(pcapng_file, output_csv=None):
    """
    Extract data from a PCAPNG file and save it to a CSV file with CICIDS-like features.
    
    Args:
        pcapng_file (str): Path to the PCAPNG file
        output_csv (str, optional): Path to the output CSV file. If None, will use the same name as the input file.
    
    Returns:
        str: Path to the created CSV file
    """
    if output_csv is None:
        # Create output filename based on input filename
        output_csv = os.path.splitext(pcapng_file)[0] + '_CICIDS.csv'
    
    print(f"Processing {pcapng_file}...")
    
    # Read the pcapng file
    cap = pyshark.FileCapture(pcapng_file)
    
    # Prepare data for packet-level CSV
    packet_data = []
    
    # Flow tracking for statistical features
    flows = defaultdict(lambda: {
        'forward_packets': [],
        'backward_packets': [],
        'start_time': None,
        'last_time': None,
        'fwd_flags': set(),
        'bwd_flags': set(),
        'fwd_bytes': 0,
        'bwd_bytes': 0,
        'fwd_packets_count': 0,
        'bwd_packets_count': 0,
        'fwd_iat': [],  # Inter-arrival times
        'bwd_iat': [],
        'active_times': [],
        'idle_times': [],
        'last_active_time': None,
        'tcp_flags_counts': defaultdict(int),
        'fwd_psh_flags': 0,
        'bwd_psh_flags': 0,
        'fwd_urg_flags': 0,
        'bwd_urg_flags': 0,
        'fwd_header_bytes': 0,
        'bwd_header_bytes': 0,
        'fwd_bulk_bytes': 0,
        'bwd_bulk_bytes': 0,
        'fwd_bulk_packets': 0,
        'bwd_bulk_packets': 0,
        'fwd_bulk_state': 0,
        'bwd_bulk_state': 0,
        'fwd_last_bulk': 0,
        'bwd_last_bulk': 0,
        'fwd_avg_bytes': 0,
        'bwd_avg_bytes': 0,
        'fwd_subflow_bytes': 0,
        'bwd_subflow_bytes': 0,
        'fwd_init_win_bytes': 0,
        'bwd_init_win_bytes': 0,
        'fwd_win_bytes': [],
        'bwd_win_bytes': [],
        'protocol': None,
        'src_port': None,
        'dst_port': None,
        'src_ip': None,
        'dst_ip': None,
        'flow_duration': 0,
        'is_attack': False  # Can be set based on known attack patterns or labels
    })
    
    # Threshold for idle time (5 seconds)
    IDLE_THRESHOLD = 5.0
    
    # Process each packet
    for i, packet in enumerate(cap):
        try:
            # Basic packet information
            timestamp = packet.sniff_time
            
            # Skip packets without IP layer
            if not hasattr(packet, 'ip') and not hasattr(packet, 'ipv6'):
                continue
                
            # Determine IP addresses
            if hasattr(packet, 'ip'):
                src_ip = packet.ip.src
                dst_ip = packet.ip.dst
                ip_proto = packet.ip.proto if hasattr(packet.ip, 'proto') else ''
            elif hasattr(packet, 'ipv6'):
                src_ip = packet.ipv6.src
                dst_ip = packet.ipv6.dst
                ip_proto = packet.ipv6.nxt if hasattr(packet.ipv6, 'nxt') else ''
            else:
                continue
                
            # Determine ports and protocol
            if hasattr(packet, 'tcp'):
                src_port = packet.tcp.srcport
                dst_port = packet.tcp.dstport
                protocol = 'TCP'
                flags = packet.tcp.flags if hasattr(packet.tcp, 'flags') else ''
                header_length = int(packet.tcp.hdr_len) if hasattr(packet.tcp, 'hdr_len') else 20
                window_size = int(packet.tcp.window_size) if hasattr(packet.tcp, 'window_size') else 0
            elif hasattr(packet, 'udp'):
                src_port = packet.udp.srcport
                dst_port = packet.udp.dstport
                protocol = 'UDP'
                flags = ''
                header_length = 8  # UDP header is always 8 bytes
                window_size = 0
            else:
                continue  # Skip non-TCP/UDP packets for flow analysis
                
            # Packet length
            packet_length = int(packet.length)
            
            # Create flow keys (bidirectional)
            forward_key = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol}"
            backward_key = f"{dst_ip}:{dst_port}-{src_ip}:{src_port}-{protocol}"
            
            # Determine flow direction and key
            flow_key = None
            direction = None
            
            if forward_key in flows:
                flow_key = forward_key
                direction = 'forward'
            elif backward_key in flows:
                flow_key = backward_key
                direction = 'backward'
            else:
                # New flow, use forward key
                flow_key = forward_key
                direction = 'forward'
                flows[flow_key]['start_time'] = timestamp
                flows[flow_key]['protocol'] = protocol
                flows[flow_key]['src_ip'] = src_ip
                flows[flow_key]['dst_ip'] = dst_ip
                flows[flow_key]['src_port'] = src_port
                flows[flow_key]['dst_port'] = dst_port
                
                # Set initial window size if TCP
                if protocol == 'TCP' and hasattr(packet.tcp, 'window_size'):
                    flows[flow_key]['fwd_init_win_bytes'] = window_size
            
            # Update flow information
            flow = flows[flow_key]
            
            # Update last time
            if flow['last_time'] is not None:
                time_diff = (timestamp - flow['last_time']).total_seconds()
                
                # Check for active/idle time
                if time_diff > IDLE_THRESHOLD:
                    if flow['last_active_time'] is not None:
                        idle_time = (timestamp - flow['last_active_time']).total_seconds()
                        flow['idle_times'].append(idle_time)
                    flow['last_active_time'] = timestamp
                else:
                    if flow['last_active_time'] is not None:
                        active_time = (timestamp - flow['last_active_time']).total_seconds()
                        flow['active_times'].append(active_time)
                        flow['last_active_time'] = timestamp
                
                # Update inter-arrival times
                if direction == 'forward':
                    flow['fwd_iat'].append(time_diff)
                else:
                    flow['bwd_iat'].append(time_diff)
            
            flow['last_time'] = timestamp
            
            # Update packet counts and bytes
            if direction == 'forward':
                flow['forward_packets'].append({
                    'timestamp': timestamp,
                    'length': packet_length,
                    'header_length': header_length,
                    'flags': flags
                })
                flow['fwd_packets_count'] += 1
                flow['fwd_bytes'] += packet_length
                flow['fwd_header_bytes'] += header_length
                flow['fwd_win_bytes'].append(window_size)
                
                # Update TCP flags
                if protocol == 'TCP':
                    flow['fwd_flags'].add(flags)
                    
                    # Count specific flags
                    if hasattr(packet.tcp, 'flags_push') and packet.tcp.flags_push == '1':
                        flow['fwd_psh_flags'] += 1
                    if hasattr(packet.tcp, 'flags_urg') and packet.tcp.flags_urg == '1':
                        flow['fwd_urg_flags'] += 1
                    
                    # Count all flag combinations
                    flow['tcp_flags_counts'][flags] += 1
                    
                    # Update backward initial window if this is a SYN-ACK
                    if hasattr(packet.tcp, 'flags_syn') and packet.tcp.flags_syn == '1' and \
                       hasattr(packet.tcp, 'flags_ack') and packet.tcp.flags_ack == '1':
                        flow['bwd_init_win_bytes'] = window_size
            else:
                flow['backward_packets'].append({
                    'timestamp': timestamp,
                    'length': packet_length,
                    'header_length': header_length,
                    'flags': flags
                })
                flow['bwd_packets_count'] += 1
                flow['bwd_bytes'] += packet_length
                flow['bwd_header_bytes'] += header_length
                flow['bwd_win_bytes'].append(window_size)
                
                # Update TCP flags
                if protocol == 'TCP':
                    flow['bwd_flags'].add(flags)
                    
                    # Count specific flags
                    if hasattr(packet.tcp, 'flags_push') and packet.tcp.flags_push == '1':
                        flow['bwd_psh_flags'] += 1
                    if hasattr(packet.tcp, 'flags_urg') and packet.tcp.flags_urg == '1':
                        flow['bwd_urg_flags'] += 1
                    
                    # Count all flag combinations
                    flow['tcp_flags_counts'][flags] += 1
            
            # Basic packet information for packet-level CSV
            packet_info = {
                'packet_number': i + 1,
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'),
                'flow_id': flow_key,
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'src_port': src_port,
                'dst_port': dst_port,
                'protocol': protocol,
                'frame_length': packet_length,
                'direction': direction,
                'tcp_flags': flags if protocol == 'TCP' else '',
            }
            
            # Add Ethernet information if available
            if hasattr(packet, 'eth'):
                packet_info.update({
                    'eth_src': packet.eth.src if hasattr(packet.eth, 'src') else '',
                    'eth_dst': packet.eth.dst if hasattr(packet.eth, 'dst') else '',
                })
            
            # Add IP information if available
            if hasattr(packet, 'ip'):
                packet_info.update({
                    'ip_version': packet.ip.version if hasattr(packet.ip, 'version') else '',
                    'ip_ttl': packet.ip.ttl if hasattr(packet.ip, 'ttl') else '',
                    'ip_flags': packet.ip.flags if hasattr(packet.ip, 'flags') else '',
                })
            elif hasattr(packet, 'ipv6'):
                packet_info.update({
                    'ip_version': '6',
                    'ip_ttl': packet.ipv6.hlim if hasattr(packet.ipv6, 'hlim') else '',
                })
            
            # Add TCP/UDP specific information
            if hasattr(packet, 'tcp'):
                packet_info.update({
                    'tcp_seq': packet.tcp.seq if hasattr(packet.tcp, 'seq') else '',
                    'tcp_ack': packet.tcp.ack if hasattr(packet.tcp, 'ack') else '',
                    'tcp_window_size': window_size,
                    'tcp_urgent_pointer': packet.tcp.urgent_pointer if hasattr(packet.tcp, 'urgent_pointer') else '',
                })
            elif hasattr(packet, 'udp'):
                packet_info.update({
                    'udp_length': packet.udp.length if hasattr(packet.udp, 'length') else '',
                })
            
            packet_data.append(packet_info)
            
            # Print progress every 1000 packets
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1} packets...")
                
        except Exception as e:
            print(f"Error processing packet {i + 1}: {str(e)}")
    
    # Close the capture
    cap.close()
    
    # Calculate flow-level features
    flow_data = []
    
    for flow_key, flow in flows.items():
        # Skip flows with too few packets
        if flow['fwd_packets_count'] + flow['bwd_packets_count'] < 2:
            continue
        
        # Calculate flow duration
        if flow['start_time'] and flow['last_time']:
            flow_duration = (flow['last_time'] - flow['start_time']).total_seconds()
            flow['flow_duration'] = flow_duration
        else:
            flow_duration = 0
        
        # Calculate packet statistics
        fwd_packet_lengths = [p['length'] for p in flow['forward_packets']]
        bwd_packet_lengths = [p['length'] for p in flow['backward_packets']]
        
        # Avoid division by zero
        fwd_packets_count = max(1, flow['fwd_packets_count'])
        bwd_packets_count = max(1, flow['bwd_packets_count'])
        total_packets = fwd_packets_count + bwd_packets_count
        
        # Calculate basic statistics
        fwd_packet_length_mean = np.mean(fwd_packet_lengths) if fwd_packet_lengths else 0
        fwd_packet_length_std = np.std(fwd_packet_lengths) if len(fwd_packet_lengths) > 1 else 0
        fwd_packet_length_max = np.max(fwd_packet_lengths) if fwd_packet_lengths else 0
        fwd_packet_length_min = np.min(fwd_packet_lengths) if fwd_packet_lengths else 0
        
        bwd_packet_length_mean = np.mean(bwd_packet_lengths) if bwd_packet_lengths else 0
        bwd_packet_length_std = np.std(bwd_packet_lengths) if len(bwd_packet_lengths) > 1 else 0
        bwd_packet_length_max = np.max(bwd_packet_lengths) if bwd_packet_lengths else 0
        bwd_packet_length_min = np.min(bwd_packet_lengths) if bwd_packet_lengths else 0
        
        # Calculate IAT (Inter-Arrival Time) statistics
        fwd_iat_mean = np.mean(flow['fwd_iat']) if flow['fwd_iat'] else 0
        fwd_iat_std = np.std(flow['fwd_iat']) if len(flow['fwd_iat']) > 1 else 0
        fwd_iat_max = np.max(flow['fwd_iat']) if flow['fwd_iat'] else 0
        fwd_iat_min = np.min(flow['fwd_iat']) if flow['fwd_iat'] else 0
        
        bwd_iat_mean = np.mean(flow['bwd_iat']) if flow['bwd_iat'] else 0
        bwd_iat_std = np.std(flow['bwd_iat']) if len(flow['bwd_iat']) > 1 else 0
        bwd_iat_max = np.max(flow['bwd_iat']) if flow['bwd_iat'] else 0
        bwd_iat_min = np.min(flow['bwd_iat']) if flow['bwd_iat'] else 0
        
        # Calculate active/idle time statistics
        active_mean = np.mean(flow['active_times']) if flow['active_times'] else 0
        active_std = np.std(flow['active_times']) if len(flow['active_times']) > 1 else 0
        active_max = np.max(flow['active_times']) if flow['active_times'] else 0
        active_min = np.min(flow['active_times']) if flow['active_times'] else 0
        
        idle_mean = np.mean(flow['idle_times']) if flow['idle_times'] else 0
        idle_std = np.std(flow['idle_times']) if len(flow['idle_times']) > 1 else 0
        idle_max = np.max(flow['idle_times']) if flow['idle_times'] else 0
        idle_min = np.min(flow['idle_times']) if flow['idle_times'] else 0
        
        # Calculate flow bytes per second and packets per second
        flow_bytes_per_sec = (flow['fwd_bytes'] + flow['bwd_bytes']) / flow_duration if flow_duration > 0 else 0
        flow_packets_per_sec = total_packets / flow_duration if flow_duration > 0 else 0
        
        # Calculate window size statistics
        fwd_win_mean = np.mean(flow['fwd_win_bytes']) if flow['fwd_win_bytes'] else 0
        fwd_win_std = np.std(flow['fwd_win_bytes']) if len(flow['fwd_win_bytes']) > 1 else 0
        fwd_win_max = np.max(flow['fwd_win_bytes']) if flow['fwd_win_bytes'] else 0
        fwd_win_min = np.min(flow['fwd_win_bytes']) if flow['fwd_win_bytes'] else 0
        
        bwd_win_mean = np.mean(flow['bwd_win_bytes']) if flow['bwd_win_bytes'] else 0
        bwd_win_std = np.std(flow['bwd_win_bytes']) if len(flow['bwd_win_bytes']) > 1 else 0
        bwd_win_max = np.max(flow['bwd_win_bytes']) if flow['bwd_win_bytes'] else 0
        bwd_win_min = np.min(flow['bwd_win_bytes']) if flow['bwd_win_bytes'] else 0
        
        # Calculate subflow statistics (assuming 1 subflow per flow for simplicity)
        fwd_subflow_packets = fwd_packets_count
        bwd_subflow_packets = bwd_packets_count
        fwd_subflow_bytes = flow['fwd_bytes']
        bwd_subflow_bytes = flow['bwd_bytes']
        
        # Calculate header length statistics
        fwd_header_length = flow['fwd_header_bytes']
        bwd_header_length = flow['bwd_header_bytes']
        
        # Calculate flags statistics
        fin_flag_count = sum(1 for flags in flow['tcp_flags_counts'] if 'F' in flags)
        syn_flag_count = sum(1 for flags in flow['tcp_flags_counts'] if 'S' in flags)
        rst_flag_count = sum(1 for flags in flow['tcp_flags_counts'] if 'R' in flags)
        psh_flag_count = sum(1 for flags in flow['tcp_flags_counts'] if 'P' in flags)
        ack_flag_count = sum(1 for flags in flow['tcp_flags_counts'] if 'A' in flags)
        urg_flag_count = sum(1 for flags in flow['tcp_flags_counts'] if 'U' in flags)
        
        # Determine if source/destination IPs are private
        src_ip_private = is_private_ip(flow['src_ip'])
        dst_ip_private = is_private_ip(flow['dst_ip'])
        
        # Create flow record
        flow_record = {
            'flow_id': flow_key,
            'src_ip': flow['src_ip'],
            'src_port': flow['src_port'],
            'dst_ip': flow['dst_ip'],
            'dst_port': flow['dst_port'],
            'protocol': flow['protocol'],
            'timestamp': flow['start_time'].strftime('%Y-%m-%d %H:%M:%S.%f') if flow['start_time'] else '',
            'flow_duration': flow_duration,
            'total_fwd_packets': fwd_packets_count,
            'total_bwd_packets': bwd_packets_count,
            'total_length_of_fwd_packets': flow['fwd_bytes'],
            'total_length_of_bwd_packets': flow['bwd_bytes'],
            'fwd_packet_length_max': fwd_packet_length_max,
            'fwd_packet_length_min': fwd_packet_length_min,
            'fwd_packet_length_mean': fwd_packet_length_mean,
            'fwd_packet_length_std': fwd_packet_length_std,
            'bwd_packet_length_max': bwd_packet_length_max,
            'bwd_packet_length_min': bwd_packet_length_min,
            'bwd_packet_length_mean': bwd_packet_length_mean,
            'bwd_packet_length_std': bwd_packet_length_std,
            'flow_bytes_per_s': flow_bytes_per_sec,
            'flow_packets_per_s': flow_packets_per_sec,
            'flow_iat_mean': np.mean(flow['fwd_iat'] + flow['bwd_iat']) if flow['fwd_iat'] or flow['bwd_iat'] else 0,
            'flow_iat_std': np.std(flow['fwd_iat'] + flow['bwd_iat']) if len(flow['fwd_iat'] + flow['bwd_iat']) > 1 else 0,
            'flow_iat_max': np.max(flow['fwd_iat'] + flow['bwd_iat']) if flow['fwd_iat'] or flow['bwd_iat'] else 0,
            'flow_iat_min': np.min(flow['fwd_iat'] + flow['bwd_iat']) if flow['fwd_iat'] or flow['bwd_iat'] else 0,
            'fwd_iat_total': sum(flow['fwd_iat']) if flow['fwd_iat'] else 0,
            'fwd_iat_mean': fwd_iat_mean,
            'fwd_iat_std': fwd_iat_std,
            'fwd_iat_max': fwd_iat_max,
            'fwd_iat_min': fwd_iat_min,
            'bwd_iat_total': sum(flow['bwd_iat']) if flow['bwd_iat'] else 0,
            'bwd_iat_mean': bwd_iat_mean,
            'bwd_iat_std': bwd_iat_std,
            'bwd_iat_max': bwd_iat_max,
            'bwd_iat_min': bwd_iat_min,
            'fwd_psh_flags': flow['fwd_psh_flags'],
            'bwd_psh_flags': flow['bwd_psh_flags'],
            'fwd_urg_flags': flow['fwd_urg_flags'],
            'bwd_urg_flags': flow['bwd_urg_flags'],
            'fin_flag_count': fin_flag_count,
            'syn_flag_count': syn_flag_count,
            'rst_flag_count': rst_flag_count,
            'psh_flag_count': psh_flag_count,
            'ack_flag_count': ack_flag_count,
            'urg_flag_count': urg_flag_count,
            'fwd_header_length': fwd_header_length,
            'bwd_header_length': bwd_header_length,
            'fwd_packets_per_s': fwd_packets_count / flow_duration if flow_duration > 0 else 0,
            'bwd_packets_per_s': bwd_packets_count / flow_duration if flow_duration > 0 else 0,
            'packet_length_min': min(fwd_packet_length_min, bwd_packet_length_min) if fwd_packet_lengths or bwd_packet_lengths else 0,
            'packet_length_max': max(fwd_packet_length_max, bwd_packet_length_max) if fwd_packet_lengths or bwd_packet_lengths else 0,
            'packet_length_mean': np.mean(fwd_packet_lengths + bwd_packet_lengths) if fwd_packet_lengths or bwd_packet_lengths else 0,
            'packet_length_std': np.std(fwd_packet_lengths + bwd_packet_lengths) if len(fwd_packet_lengths + bwd_packet_lengths) > 1 else 0,
            'packet_length_variance': np.var(fwd_packet_lengths + bwd_packet_lengths) if len(fwd_packet_lengths + bwd_packet_lengths) > 1 else 0,
            'fwd_init_win_bytes': flow['fwd_init_win_bytes'],
            'bwd_init_win_bytes': flow['bwd_init_win_bytes'],
            'fwd_win_bytes_mean': fwd_win_mean,
            'fwd_win_bytes_std': fwd_win_std,
            'fwd_win_bytes_max': fwd_win_max,
            'fwd_win_bytes_min': fwd_win_min,
            'bwd_win_bytes_mean': bwd_win_mean,
            'bwd_win_bytes_std': bwd_win_std,
            'bwd_win_bytes_max': bwd_win_max,
            'bwd_win_bytes_min': bwd_win_min,
            'active_mean': active_mean,
            'active_std': active_std,
            'active_max': active_max,
            'active_min': active_min,
            'idle_mean': idle_mean,
            'idle_std': idle_std,
            'idle_max': idle_max,
            'idle_min': idle_min,
            'fwd_avg_bytes_per_bulk': 0,  # Placeholder for bulk statistics
            'fwd_avg_packets_per_bulk': 0,
            'fwd_avg_bulk_rate': 0,
            'bwd_avg_bytes_per_bulk': 0,
            'bwd_avg_packets_per_bulk': 0,
            'bwd_avg_bulk_rate': 0,
            'subflow_fwd_packets': fwd_subflow_packets,
            'subflow_fwd_bytes': fwd_subflow_bytes,
            'subflow_bwd_packets': bwd_subflow_packets,
            'subflow_bwd_bytes': bwd_subflow_bytes,
            'src_ip_private': src_ip_private,
            'dst_ip_private': dst_ip_private,
            'is_attack': flow['is_attack']  # Placeholder for attack label
        }
        
        flow_data.append(flow_record)
    
    # Save packet-level data to CSV
    packet_df = pd.DataFrame(packet_data)
    packet_csv = os.path.splitext(output_csv)[0] + '_packets.csv'
    packet_df.to_csv(packet_csv, index=False)
    print(f"Saved {len(packet_data)} packets to {packet_csv}")
    
    # Save flow-level data to CSV (CICIDS-like)
    flow_df = pd.DataFrame(flow_data)
    flow_df.to_csv(output_csv, index=False)
    print(f"Saved {len(flow_data)} flows to {output_csv}")
    
    return output_csv, packet_csv

def process_directory(directory_path, output_dir=None):
    """
    Process all PCAPNG files in a directory and convert them to CSV.
    
    Args:
        directory_path (str): Path to the directory containing PCAPNG files
        output_dir (str, optional): Directory to save CSV files. If None, will save in the same directory.
    
    Returns:
        list: List of created CSV files
    """
    # Create output directory if specified and doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Find all PCAPNG files in the directory
    pcapng_files = glob.glob(os.path.join(directory_path, "*.pcapng"))
    
    if not pcapng_files:
        print(f"No PCAPNG files found in {directory_path}")
        return []
    
    print(f"Found {len(pcapng_files)} PCAPNG files in {directory_path}")
    
    # Process each file
    csv_files = []
    for pcapng_file in pcapng_files:
        if output_dir:
            base_name = os.path.basename(pcapng_file)
            output_csv = os.path.join(output_dir, os.path.splitext(base_name)[0] + '_CICIDS.csv')
        else:
            output_csv = None
        
        flow_csv, packet_csv = extract_pcapng_to_csv(pcapng_file, output_csv)
        csv_files.extend([flow_csv, packet_csv])
    
    return csv_files

def main():
    # Directory containing PCAPNG files
    pcapng_dir = r"d:\AInetwork\Dataset\WireShark"
    
    # Directory to save CSV files (create a new directory for CSV files)
    output_dir = r"d:\AInetwork\Dataset\CsvFiles"
    
    # Process all PCAPNG files in the directory
    csv_files = process_directory(pcapng_dir, output_dir)
    
    if csv_files:
        print("\nSummary:")
        print(f"Processed files and created {len(csv_files)} CSV files:")
        for csv_file in csv_files:
            print(f"- {csv_file}")

if __name__ == "__main__":
    start_time = datetime.now()
    print(f"Starting conversion at {start_time}")
    
    main()
    
    end_time = datetime.now()
    print(f"Conversion completed at {end_time}")
    print(f"Total time: {end_time - start_time}")

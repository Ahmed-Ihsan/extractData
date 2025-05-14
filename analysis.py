import os
import pandas as pd
import matplotlib.pyplot as plt
from NetworkTrafficEDA import NetworkTrafficEDA

def main():
    """
    Main function to run the Network Traffic EDA analysis.
    """
    # Set up paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CsvFiles")
    cicids_file = os.path.join(data_dir, "dataN6_CICIDS.csv")
    
    print("=" * 80)
    print("Network Traffic Exploratory Data Analysis")
    print("=" * 80)
    
    # Check if file exists
    if not os.path.exists(cicids_file):
        print(f"Error: File not found at {cicids_file}")
        print("Please provide the correct path to your network traffic dataset.")
        return
    
    # Initialize the EDA class
    print("\nInitializing NetworkTrafficEDA and loading data...")
    eda = NetworkTrafficEDA(data_path=cicids_file)
    
    # Basic information
    print("\n\n" + "=" * 50)
    print("BASIC DATASET INFORMATION")
    print("=" * 50)
    eda.get_basic_info()
    
    # Clean data
    print("\n\n" + "=" * 50)
    print("DATA CLEANING")
    print("=" * 50)
    eda.clean_data(drop_duplicates=True, handle_missing='median')
    
    # Convert IP addresses to numeric for analysis
    print("\n\n" + "=" * 50)
    print("IP ADDRESS CONVERSION")
    print("=" * 50)
    eda.convert_ip_to_numeric()
    
    # Analyze numeric features
    print("\n\n" + "=" * 50)
    print("NUMERIC FEATURE ANALYSIS")
    print("=" * 50)
    eda.analyze_numeric_features()
    
    # Plot distributions
    print("\n\n" + "=" * 50)
    print("FEATURE DISTRIBUTIONS")
    print("=" * 50)
    # Select a subset of important features to visualize
    important_features = eda.numeric_columns[:10]  # First 10 numeric features
    eda.plot_distributions(columns=important_features)
    
    # Correlation analysis
    print("\n\n" + "=" * 50)
    print("CORRELATION ANALYSIS")
    print("=" * 50)
    eda.plot_correlation_matrix()
    
    # Protocol distribution
    print("\n\n" + "=" * 50)
    print("PROTOCOL DISTRIBUTION")
    print("=" * 50)
    eda.analyze_protocol_distribution()
    
    # Attack distribution if available
    if eda.attack_column:
        print("\n\n" + "=" * 50)
        print("ATTACK DISTRIBUTION")
        print("=" * 50)
        eda.analyze_attack_distribution()
    
    # Traffic over time
    print("\n\n" + "=" * 50)
    print("TRAFFIC OVER TIME")
    print("=" * 50)
    if eda.time_columns:
        eda.analyze_traffic_over_time(resample_freq='1H')
    
    # Port distribution
    print("\n\n" + "=" * 50)
    print("PORT DISTRIBUTION")
    print("=" * 50)
    eda.analyze_port_distribution(top_n=15)
    
    # Network flow visualization
    print("\n\n" + "=" * 50)
    print("NETWORK FLOW VISUALIZATION")
    print("=" * 50)
    eda.visualize_network_flows(limit=1000)
    
    # PCA analysis
    print("\n\n" + "=" * 50)
    print("PCA ANALYSIS")
    print("=" * 50)
    eda.plot_pca_analysis(n_components=3, plot_type='2d')
    
    # Generate summary report
    print("\n\n" + "=" * 50)
    print("GENERATING SUMMARY REPORT")
    print("=" * 50)
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "network_traffic_report.md")
    eda.generate_summary_report(output_file=report_path)
    print(f"Report saved to: {report_path}")
    
    print("\n\n" + "=" * 80)
    print("EDA Analysis Complete!")
    print("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error running EDA analysis: {e}")

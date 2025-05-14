import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings
from typing import List, Dict, Tuple, Union, Optional
import ipaddress

class NetworkTrafficEDA:
    """
    A professional class for Exploratory Data Analysis of Network Traffic data,
    specifically designed for CICIDS and similar network security datasets.
    
    This class provides comprehensive methods for:
    - Data loading and basic inspection
    - Data cleaning and preprocessing
    - Statistical analysis
    - Feature analysis and correlation
    - Visualization of network traffic patterns
    - Attack detection and analysis
    """
    
    def __init__(self, data_path: str = None, df: pd.DataFrame = None):
        """
        Initialize the NetworkTrafficEDA class with either a path to a CSV file or a pandas DataFrame.
        
        Args:
            data_path (str, optional): Path to the CSV file containing network traffic data.
            df (pd.DataFrame, optional): Pandas DataFrame containing network traffic data.
        
        Raises:
            ValueError: If neither data_path nor df is provided.
        """
        self.data_path = data_path
        self.df = df
        self.original_df = None
        self.numeric_columns = []
        self.categorical_columns = []
        self.ip_columns = []
        self.time_columns = []
        self.attack_column = None
        
        # Set plotting style
        plt.style.use('seaborn-whitegrid')
        sns.set_style("whitegrid")
        warnings.filterwarnings('ignore')
        
        if data_path is not None:
            self.load_data(data_path)
        elif df is not None:
            self.set_dataframe(df)
        else:
            raise ValueError("Either data_path or df must be provided")
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        Args:
            data_path (str): Path to the CSV file.
            
        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        try:
            self.data_path = data_path
            self.df = pd.read_csv(data_path, low_memory=False)
            self.original_df = self.df.copy()
            print(f"Data loaded successfully from {data_path}")
            print(f"Shape: {self.df.shape}")
            self._identify_column_types()
            return self.df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def set_dataframe(self, df: pd.DataFrame) -> None:
        """
        Set the DataFrame directly.
        
        Args:
            df (pd.DataFrame): The DataFrame to analyze.
        """
        self.df = df
        self.original_df = df.copy()
        print(f"DataFrame set successfully")
        print(f"Shape: {self.df.shape}")
        self._identify_column_types()
    
    def _identify_column_types(self) -> None:
        """
        Identify and categorize column types in the dataset.
        """
        # Reset column lists
        self.numeric_columns = []
        self.categorical_columns = []
        self.ip_columns = []
        self.time_columns = []
        
        # Identify column types
        for col in self.df.columns:
            # Check for IP address columns
            if 'ip' in col.lower():
                self.ip_columns.append(col)
            # Check for time columns
            elif 'time' in col.lower() or 'date' in col.lower():
                self.time_columns.append(col)
            # Check for numeric columns
            elif pd.api.types.is_numeric_dtype(self.df[col]):
                self.numeric_columns.append(col)
            # Everything else is categorical
            else:
                self.categorical_columns.append(col)
        
        # Identify attack column if it exists
        if 'is_attack' in self.df.columns:
            self.attack_column = 'is_attack'
        elif 'label' in self.df.columns:
            self.attack_column = 'label'
        
        print(f"Numeric columns: {len(self.numeric_columns)}")
        print(f"Categorical columns: {len(self.categorical_columns)}")
        print(f"IP columns: {len(self.ip_columns)}")
        print(f"Time columns: {len(self.time_columns)}")
        if self.attack_column:
            print(f"Attack column identified: {self.attack_column}")
    
    def get_basic_info(self) -> None:
        """
        Display basic information about the dataset.
        """
        if self.df is None:
            print("No data loaded. Please load data first.")
            return
        
        print("Dataset Information:")
        print("-" * 50)
        print(f"Number of rows: {self.df.shape[0]}")
        print(f"Number of columns: {self.df.shape[1]}")
        print("-" * 50)
        print("\nData Types:")
        print(self.df.dtypes.value_counts())
        print("-" * 50)
        print("\nMissing Values:")
        missing = self.df.isnull().sum()
        print(missing[missing > 0] if missing.any() > 0 else "No missing values")
        print("-" * 50)
        
        # Display sample data
        print("\nSample Data:")
        print(self.df.head())
        
        # Display memory usage
        memory_usage = self.df.memory_usage(deep=True).sum() / (1024 * 1024)
        print(f"\nMemory Usage: {memory_usage:.2f} MB")
    
    def clean_data(self, drop_duplicates: bool = True, 
                  handle_missing: str = 'drop', 
                  drop_columns: List[str] = None) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values, duplicates, and dropping unnecessary columns.
        
        Args:
            drop_duplicates (bool): Whether to drop duplicate rows.
            handle_missing (str): How to handle missing values ('drop', 'mean', 'median', 'mode', 'zero').
            drop_columns (List[str]): List of columns to drop.
            
        Returns:
            pd.DataFrame: The cleaned DataFrame.
        """
        if self.df is None:
            print("No data loaded. Please load data first.")
            return None
        
        print("Cleaning data...")
        df_clean = self.df.copy()
        
        # Drop specified columns
        if drop_columns:
            df_clean = df_clean.drop(columns=[col for col in drop_columns if col in df_clean.columns])
            print(f"Dropped {len(drop_columns)} columns")
        
        # Handle duplicates
        if drop_duplicates:
            dup_count = df_clean.duplicated().sum()
            df_clean = df_clean.drop_duplicates()
            print(f"Dropped {dup_count} duplicate rows")
        
        # Handle missing values
        missing_count = df_clean.isnull().sum().sum()
        if missing_count > 0:
            if handle_missing == 'drop':
                df_clean = df_clean.dropna()
                print(f"Dropped rows with missing values")
            elif handle_missing == 'mean':
                for col in self.numeric_columns:
                    if df_clean[col].isnull().sum() > 0:
                        df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                print(f"Filled missing numeric values with mean")
            elif handle_missing == 'median':
                for col in self.numeric_columns:
                    if df_clean[col].isnull().sum() > 0:
                        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                print(f"Filled missing numeric values with median")
            elif handle_missing == 'mode':
                for col in df_clean.columns:
                    if df_clean[col].isnull().sum() > 0:
                        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
                print(f"Filled missing values with mode")
            elif handle_missing == 'zero':
                df_clean = df_clean.fillna(0)
                print(f"Filled missing values with zero")
        
        # Update the dataframe
        self.df = df_clean
        print(f"Data cleaned. New shape: {self.df.shape}")
        return self.df
    
    def convert_ip_to_numeric(self, ip_columns: List[str] = None) -> pd.DataFrame:
        """
        Convert IP addresses to numeric values for analysis.
        
        Args:
            ip_columns (List[str], optional): List of IP columns to convert. If None, uses all detected IP columns.
            
        Returns:
            pd.DataFrame: DataFrame with converted IP columns.
        """
        if self.df is None:
            print("No data loaded. Please load data first.")
            return None
        
        if ip_columns is None:
            ip_columns = self.ip_columns
        
        df_copy = self.df.copy()
        
        for col in ip_columns:
            if col in df_copy.columns:
                try:
                    # Convert IP addresses to integers
                    df_copy[f"{col}_numeric"] = df_copy[col].apply(
                        lambda x: int(ipaddress.ip_address(str(x).split('-')[0] if '-' in str(x) else str(x)))
                    )
                    print(f"Converted {col} to numeric")
                except Exception as e:
                    print(f"Error converting {col} to numeric: {e}")
        
        self.df = df_copy
        return self.df
    
    def analyze_numeric_features(self, columns: List[str] = None) -> pd.DataFrame:
        """
        Analyze numeric features in the dataset.
        
        Args:
            columns (List[str], optional): List of numeric columns to analyze. If None, uses all numeric columns.
            
        Returns:
            pd.DataFrame: DataFrame with statistical summary of numeric features.
        """
        if self.df is None:
            print("No data loaded. Please load data first.")
            return None
        
        if columns is None:
            columns = self.numeric_columns
        
        # Filter only existing columns
        columns = [col for col in columns if col in self.df.columns]
        
        if not columns:
            print("No numeric columns to analyze.")
            return None
        
        # Calculate statistics
        stats = self.df[columns].describe().T
        
        # Add additional statistics
        stats['skew'] = self.df[columns].skew()
        stats['kurtosis'] = self.df[columns].kurtosis()
        stats['missing'] = self.df[columns].isnull().sum()
        stats['missing_pct'] = (self.df[columns].isnull().sum() / len(self.df)) * 100
        
        print("Numeric Feature Analysis:")
        print(stats)
        
        return stats
    
    def plot_distributions(self, columns: List[str] = None, 
                          n_cols: int = 3, figsize: Tuple[int, int] = (18, 15),
                          bins: int = 30) -> None:
        """
        Plot distributions of numeric features.
        
        Args:
            columns (List[str], optional): List of columns to plot. If None, uses numeric columns.
            n_cols (int): Number of columns in the subplot grid.
            figsize (Tuple[int, int]): Figure size.
            bins (int): Number of bins for histograms.
        """
        if self.df is None:
            print("No data loaded. Please load data first.")
            return
        
        if columns is None:
            columns = self.numeric_columns[:15]  # Limit to 15 columns to avoid overcrowding
        
        # Filter only existing columns
        columns = [col for col in columns if col in self.df.columns]
        
        if not columns:
            print("No columns to plot.")
            return
        
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        for i, col in enumerate(columns):
            if i < len(axes):
                sns.histplot(self.df[col], bins=bins, kde=True, ax=axes[i])
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
                
                # Add mean and median lines
                mean_val = self.df[col].mean()
                median_val = self.df[col].median()
                axes[i].axvline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.2f}')
                axes[i].axvline(median_val, color='g', linestyle='-.', label=f'Median: {median_val:.2f}')
                axes[i].legend()
        
        # Hide unused subplots
        for j in range(len(columns), len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_matrix(self, columns: List[str] = None, 
                               figsize: Tuple[int, int] = (12, 10),
                               cmap: str = 'coolwarm') -> None:
        """
        Plot correlation matrix for numeric features.
        
        Args:
            columns (List[str], optional): List of columns to include. If None, uses numeric columns.
            figsize (Tuple[int, int]): Figure size.
            cmap (str): Colormap for the heatmap.
        """
        if self.df is None:
            print("No data loaded. Please load data first.")
            return
        
        if columns is None:
            # Use numeric columns but limit to 20 to avoid overcrowding
            if len(self.numeric_columns) > 20:
                print(f"Limiting correlation matrix to 20 columns out of {len(self.numeric_columns)}")
                columns = self.numeric_columns[:20]
            else:
                columns = self.numeric_columns
        
        # Filter only existing columns
        columns = [col for col in columns if col in self.df.columns]
        
        if not columns:
            print("No columns to plot correlation matrix.")
            return
        
        # Calculate correlation matrix
        corr_matrix = self.df[columns].corr()
        
        # Plot correlation matrix
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, annot=True, fmt=".2f", 
                   linewidths=0.5, vmin=-1, vmax=1)
        plt.title('Correlation Matrix of Numeric Features', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, top_n: int = 15) -> None:
        """
        Plot feature importance based on correlation with target variable.
        
        Args:
            top_n (int): Number of top features to display.
        """
        if self.df is None or self.attack_column is None:
            print("No data loaded or no attack column identified.")
            return
        
        # Calculate correlation with target
        if self.attack_column in self.df.columns:
            # Convert target to numeric if it's not
            if not pd.api.types.is_numeric_dtype(self.df[self.attack_column]):
                target = self.df[self.attack_column].map({'True': 1, 'False': 0})
                if target.isnull().any():  # If mapping didn't work
                    target = pd.factorize(self.df[self.attack_column])[0]
            else:
                target = self.df[self.attack_column]
            
            # Calculate correlation for numeric features
            correlations = {}
            for col in self.numeric_columns:
                if col != self.attack_column and col in self.df.columns:
                    correlations[col] = abs(np.corrcoef(self.df[col], target)[0, 1])
            
            # Sort and get top features
            sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
            top_features = sorted_features[:top_n]
            
            # Plot
            plt.figure(figsize=(12, 8))
            features, importance = zip(*top_features)
            sns.barplot(x=list(importance), y=list(features))
            plt.title(f'Top {top_n} Features by Correlation with Target', fontsize=16)
            plt.xlabel('Absolute Correlation')
            plt.tight_layout()
            plt.show()
        else:
            print(f"Attack column '{self.attack_column}' not found in dataframe.")
    
    def plot_pca_analysis(self, n_components: int = 2, 
                         plot_type: str = '2d',
                         target_col: str = None) -> None:
        """
        Perform PCA analysis and plot the results.
        
        Args:
            n_components (int): Number of PCA components.
            plot_type (str): Type of plot ('2d' or '3d').
            target_col (str, optional): Target column for coloring. If None, uses attack_column.
        """
        if self.df is None:
            print("No data loaded. Please load data first.")
            return
        
        if target_col is None:
            target_col = self.attack_column
        
        # Select numeric columns for PCA
        numeric_df = self.df[self.numeric_columns].copy()
        
        # Handle missing values
        numeric_df = numeric_df.fillna(numeric_df.mean())
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(scaled_data)
        
        # Create DataFrame with PCA results
        pca_df = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
        
        # Add target column if available
        if target_col and target_col in self.df.columns:
            pca_df[target_col] = self.df[target_col].values
        
        # Plot results
        if plot_type == '2d' and n_components >= 2:
            plt.figure(figsize=(12, 8))
            if target_col and target_col in pca_df.columns:
                scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pd.factorize(pca_df[target_col])[0], 
                                     alpha=0.6, cmap='viridis')
                plt.colorbar(scatter, label=target_col)
            else:
                plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.6)
            
            plt.title('PCA: First Two Principal Components', fontsize=16)
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            
            # Plot explained variance
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, n_components+1), np.cumsum(pca.explained_variance_ratio_), marker='o')
            plt.title('Explained Variance by Components', fontsize=16)
            plt.xlabel('Number of Components')
            plt.ylabel('Cumulative Explained Variance')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            
        elif plot_type == '3d' and n_components >= 3:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            if target_col and target_col in pca_df.columns:
                scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], 
                                   c=pd.factorize(pca_df[target_col])[0], alpha=0.6, cmap='viridis')
                plt.colorbar(scatter, label=target_col)
            else:
                ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], alpha=0.6)
            
            ax.set_title('PCA: First Three Principal Components', fontsize=16)
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%} variance)')
            plt.tight_layout()
            plt.show()
    
    def analyze_protocol_distribution(self) -> None:
        """
        Analyze the distribution of network protocols in the dataset.
        """
        if self.df is None:
            print("No data loaded. Please load data first.")
            return
        
        protocol_col = None
        
        # Find protocol column
        for col in self.df.columns:
            if 'protocol' in col.lower():
                protocol_col = col
                break
        
        if protocol_col is None:
            print("No protocol column found in the dataset.")
            return
        
        # Count protocols
        protocol_counts = self.df[protocol_col].value_counts()
        
        # Plot
        plt.figure(figsize=(12, 6))
        sns.barplot(x=protocol_counts.index, y=protocol_counts.values)
        plt.title('Distribution of Network Protocols', fontsize=16)
        plt.xlabel('Protocol')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # If attack column exists, analyze protocols by attack status
        if self.attack_column and self.attack_column in self.df.columns:
            plt.figure(figsize=(14, 7))
            protocol_attack = pd.crosstab(self.df[protocol_col], self.df[self.attack_column])
            protocol_attack.plot(kind='bar', stacked=True)
            plt.title('Protocols by Attack Status', fontsize=16)
            plt.xlabel('Protocol')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.legend(title='Attack')
            plt.tight_layout()
            plt.show()
    
    def analyze_port_distribution(self, top_n: int = 10) -> None:
        """
        Analyze the distribution of source and destination ports.
        
        Args:
            top_n (int): Number of top ports to display.
        """
        if self.df is None:
            print("No data loaded. Please load data first.")
            return
        
        src_port_col = None
        dst_port_col = None
        
        # Find port columns
        for col in self.df.columns:
            if 'src_port' in col.lower():
                src_port_col = col
            elif 'dst_port' in col.lower():
                dst_port_col = col
        
        if src_port_col is None or dst_port_col is None:
            print("Source or destination port columns not found.")
            return
        
        # Analyze source ports
        src_port_counts = self.df[src_port_col].value_counts().head(top_n)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=src_port_counts.index.astype(str), y=src_port_counts.values)
        plt.title(f'Top {top_n} Source Ports', fontsize=16)
        plt.xlabel('Source Port')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Analyze destination ports
        dst_port_counts = self.df[dst_port_col].value_counts().head(top_n)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=dst_port_counts.index.astype(str), y=dst_port_counts.values)
        plt.title(f'Top {top_n} Destination Ports', fontsize=16)
        plt.xlabel('Destination Port')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Common ports and their services
        common_ports = {
            '80': 'HTTP',
            '443': 'HTTPS',
            '22': 'SSH',
            '21': 'FTP',
            '25': 'SMTP',
            '53': 'DNS',
            '3389': 'RDP',
            '3306': 'MySQL',
            '5432': 'PostgreSQL',
            '1433': 'MSSQL'
        }
        
        # Identify common services in the dataset
        print("\nCommon Services Identified in the Dataset:")
        print("-" * 50)
        for port, service in common_ports.items():
            src_count = self.df[self.df[src_port_col] == int(port)].shape[0]
            dst_count = self.df[self.df[dst_port_col] == int(port)].shape[0]
            if src_count > 0 or dst_count > 0:
                print(f"Port {port} ({service}):")
                print(f"  - As source port: {src_count} connections")
                print(f"  - As destination port: {dst_count} connections")
    
    def analyze_flow_duration(self) -> None:
        """
        Analyze the distribution of flow durations.
        """
        if self.df is None:
            print("No data loaded. Please load data first.")
            return
        
        duration_col = None
        
        # Find duration column
        for col in self.df.columns:
            if 'duration' in col.lower() or 'time' in col.lower():
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    duration_col = col
                    break
        
        if duration_col is None:
            print("No flow duration column found.")
            return
        
        # Plot distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(self.df[duration_col], bins=50, kde=True)
        plt.title('Distribution of Flow Duration', fontsize=16)
        plt.xlabel('Duration')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
        
        # Plot log distribution for better visualization
        plt.figure(figsize=(12, 6))
        sns.histplot(np.log1p(self.df[duration_col]), bins=50, kde=True)
        plt.title('Log Distribution of Flow Duration', fontsize=16)
        plt.xlabel('Log(Duration + 1)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
        
        # If attack column exists, analyze duration by attack status
        if self.attack_column and self.attack_column in self.df.columns:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x=self.df[self.attack_column], y=self.df[duration_col])
            plt.title('Flow Duration by Attack Status', fontsize=16)
            plt.xlabel('Attack')
            plt.ylabel('Duration')
            plt.tight_layout()
            plt.show()
    
    def analyze_packet_statistics(self) -> None:
        """
        Analyze packet-related statistics.
        """
        if self.df is None:
            print("No data loaded. Please load data first.")
            return
        
        # Find relevant columns
        packet_cols = []
        for col in self.df.columns:
            if 'packet' in col.lower():
                packet_cols.append(col)
        
        if not packet_cols:
            print("No packet-related columns found.")
            return
        
        # Select a subset of columns for visualization
        selected_cols = []
        for term in ['total_fwd_packets', 'total_bwd_packets', 'packets_per_s']:
            for col in packet_cols:
                if term in col.lower():
                    selected_cols.append(col)
                    break
        
        if not selected_cols:
            selected_cols = packet_cols[:3]  # Take first 3 if specific columns not found
        
        # Plot distributions
        for col in selected_cols:
            plt.figure(figsize=(12, 6))
            sns.histplot(self.df[col], bins=50, kde=True)
            plt.title(f'Distribution of {col}', fontsize=16)
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.show()
        
        # Plot relationship between forward and backward packets
        fwd_col = next((col for col in packet_cols if 'fwd_packets' in col.lower()), None)
        bwd_col = next((col for col in packet_cols if 'bwd_packets' in col.lower()), None)
        
        if fwd_col and bwd_col:
            plt.figure(figsize=(10, 8))
            plt.scatter(self.df[fwd_col], self.df[bwd_col], alpha=0.5)
            plt.title('Forward vs Backward Packets', fontsize=16)
            plt.xlabel('Forward Packets')
            plt.ylabel('Backward Packets')
            plt.tight_layout()
            plt.show()
    
        def analyze_protocol_distribution(self) -> None:
            """
            Analyze the distribution of network protocols in the dataset.
            """
            if self.df is None:
                print("No data loaded. Please load data first.")
                return
            
            # Look for protocol column
            protocol_col = None
            for col in self.df.columns:
                if 'protocol' in col.lower() or 'proto' in col.lower():
                    protocol_col = col
                    break
            
            if protocol_col is None:
                print("No protocol column found in the dataset.")
                return
            
            # Count protocols
            protocol_counts = self.df[protocol_col].value_counts()
            
            # Plot
            plt.figure(figsize=(12, 6))
            sns.barplot(x=protocol_counts.index, y=protocol_counts.values)
            plt.title('Distribution of Network Protocols', fontsize=16)
            plt.xlabel('Protocol')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
            # Print statistics
            print("Protocol Distribution:")
            for protocol, count in protocol_counts.items():
                print(f"{protocol}: {count} ({count/len(self.df)*100:.2f}%)")
    
    def analyze_attack_distribution(self) -> None:
        """
        Analyze the distribution of attacks in the dataset.
        """
        if self.df is None or self.attack_column is None:
            print("No data loaded or no attack column identified.")
            return
        
        # Count attacks
        attack_counts = self.df[self.attack_column].value_counts()
        
        # Plot
        plt.figure(figsize=(12, 6))
        sns.barplot(x=attack_counts.index, y=attack_counts.values) 
        plt.title('Distribution of Attacks', fontsize=16)
        plt.xlabel('Attack Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print("Attack Distribution:")
        for attack, count in attack_counts.items():
            print(f"{attack}: {count} ({count/len(self.df)*100:.2f}%)")
    
    def analyze_traffic_over_time(self, time_col: str = None, 
                                 resample_freq: str = '1H',
                                 group_by_attack: bool = True) -> None:
        """
        Analyze network traffic patterns over time.
        
        Args:
            time_col (str, optional): Time column to use. If None, uses the first detected time column.
            resample_freq (str): Frequency for resampling time series ('1H' for hourly, '1D' for daily, etc.)
            group_by_attack (bool): Whether to group by attack type.
        """
        if self.df is None:
            print("No data loaded. Please load data first.")
            return
        
        # Find time column if not specified
        if time_col is None and self.time_columns:
            time_col = self.time_columns[0]
        
        if time_col is None or time_col not in self.df.columns:
            print("No valid time column found.")
            return
        
        # Convert time column to datetime if needed
        if not pd.api.types.is_datetime64_dtype(self.df[time_col]):
            try:
                self.df[time_col] = pd.to_datetime(self.df[time_col])
                print(f"Converted {time_col} to datetime format")
            except Exception as e:
                print(f"Error converting {time_col} to datetime: {e}")
                return
        
        # Set time column as index
        df_time = self.df.set_index(time_col)
        
        # Resample and count
        traffic_count = df_time.resample(resample_freq).size()
        
        # Plot
        plt.figure(figsize=(15, 6))
        traffic_count.plot()
        plt.title(f'Network Traffic Over Time (Resampled {resample_freq})', fontsize=16)
        plt.xlabel('Time')
        plt.ylabel('Number of Connections')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # Group by attack type if requested
        if group_by_attack and self.attack_column is not None:
            # Create a DataFrame with time and attack column
            attack_time_df = self.df[[time_col, self.attack_column]].copy()
            attack_time_df.set_index(time_col, inplace=True)
            
            # Group by time and attack type
            attack_time_counts = attack_time_df.groupby([pd.Grouper(freq=resample_freq), self.attack_column]).size().unstack()
            
            # Fill NaN with 0
            attack_time_counts = attack_time_counts.fillna(0)
            
            # Plot
            plt.figure(figsize=(15, 8))
            attack_time_counts.plot(kind='area', stacked=True)
            plt.title(f'Network Traffic by Attack Type Over Time (Resampled {resample_freq})', fontsize=16)
            plt.xlabel('Time')
            plt.ylabel('Number of Connections')
            plt.grid(True)
            plt.legend(title='Attack Type', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()
    
    def analyze_port_distribution(self, top_n: int = 10) -> None:
        """
        Analyze the distribution of source and destination ports.
        
        Args:
            top_n (int): Number of top ports to display.
        """
        if self.df is None:
            print("No data loaded. Please load data first.")
            return
        
        # Find port columns
        src_port_col = None
        dst_port_col = None
        
        for col in self.df.columns:
            if 'src' in col.lower() and 'port' in col.lower():
                src_port_col = col
            elif ('dst' in col.lower() or 'dest' in col.lower()) and 'port' in col.lower():
                dst_port_col = col
        
        if src_port_col is None and dst_port_col is None:
            print("No port columns found in the dataset.")
            return
        
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        
        # Plot source ports if available
        if src_port_col and src_port_col in self.df.columns:
            src_port_counts = self.df[src_port_col].value_counts().nlargest(top_n)
            sns.barplot(x=src_port_counts.index, y=src_port_counts.values, ax=axes[0])
            axes[0].set_title(f'Top {top_n} Source Ports', fontsize=14)
            axes[0].set_xlabel('Port')
            axes[0].set_ylabel('Count')
            axes[0].tick_params(axis='x', rotation=45)
        
        # Plot destination ports if available
        if dst_port_col and dst_port_col in self.df.columns:
            dst_port_counts = self.df[dst_port_col].value_counts().nlargest(top_n)
            sns.barplot(x=dst_port_counts.index, y=dst_port_counts.values, ax=axes[1])
            axes[1].set_title(f'Top {top_n} Destination Ports', fontsize=14)
            axes[1].set_xlabel('Port')
            axes[1].set_ylabel('Count')
            axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Print common port information
        common_ports = {
            20: 'FTP Data',
            21: 'FTP Control',
            22: 'SSH',
            23: 'Telnet',
            25: 'SMTP',
            53: 'DNS',
            80: 'HTTP',
            110: 'POP3',
            143: 'IMAP',
            443: 'HTTPS',
            445: 'SMB',
            3389: 'RDP'
        }
        
        print("Common ports found in the dataset:")
        if src_port_col and src_port_col in self.df.columns:
            for port in src_port_counts.index:
                if isinstance(port, (int, float)) and int(port) in common_ports:
                    print(f"Port {port} ({common_ports[int(port)]}): {src_port_counts[port]} occurrences as source")
        
        if dst_port_col and dst_port_col in self.df.columns:
            for port in dst_port_counts.index:
                if isinstance(port, (int, float)) and int(port) in common_ports:
                    print(f"Port {port} ({common_ports[int(port)]}): {dst_port_counts[port]} occurrences as destination")
    
    def visualize_network_flows(self, limit: int = 1000, 
                               src_ip_col: str = None, 
                               dst_ip_col: str = None) -> None:
        """
        Visualize network flows between source and destination IPs.
        
        Args:
            limit (int): Limit the number of flows to visualize.
            src_ip_col (str, optional): Source IP column. If None, tries to detect automatically.
            dst_ip_col (str, optional): Destination IP column. If None, tries to detect automatically.
        """
        if self.df is None:
            print("No data loaded. Please load data first.")
            return
        
        # Find IP columns if not specified
        if src_ip_col is None or dst_ip_col is None:
            for col in self.df.columns:
                if 'src' in col.lower() and ('ip' in col.lower() or 'addr' in col.lower()):
                    src_ip_col = col
                elif ('dst' in col.lower() or 'dest' in col.lower()) and ('ip' in col.lower() or 'addr' in col.lower()):
                    dst_ip_col = col
        
        if src_ip_col is None or dst_ip_col is None:
            print("Could not identify source and destination IP columns.")
            return
        
        # Sample data if needed
        if len(self.df) > limit:
            df_sample = self.df.sample(limit)
            print(f"Sampling {limit} connections from {len(self.df)} total")
        else:
            df_sample = self.df
        
        # Count flows between IPs
        flow_counts = df_sample.groupby([src_ip_col, dst_ip_col]).size().reset_index(name='count')
        flow_counts = flow_counts.sort_values('count', ascending=False)
        
        # Get top source and destination IPs
        top_src_ips = flow_counts[src_ip_col].value_counts().nlargest(10).index.tolist()
        top_dst_ips = flow_counts[dst_ip_col].value_counts().nlargest(10).index.tolist()
        
        # Filter flows involving top IPs
        top_flows = flow_counts[
            (flow_counts[src_ip_col].isin(top_src_ips)) & 
            (flow_counts[dst_ip_col].isin(top_dst_ips))
        ]
        
        # Create a network graph using plotly
        try:
            # Create nodes
            nodes = list(set(top_flows[src_ip_col].tolist() + top_flows[dst_ip_col].tolist()))
            node_indices = {node: i for i, node in enumerate(nodes)}
            
            # Create edges
            edge_x = []
            edge_y = []
            edge_weights = []
            
            for _, row in top_flows.iterrows():
                src_idx = node_indices[row[src_ip_col]]
                dst_idx = node_indices[row[dst_ip_col]]
                weight = row['count']
                
                # Add source to destination edge
                edge_x.extend([src_idx, dst_idx, None])
                edge_y.extend([0, 0, None])
                edge_weights.append(weight)
            
            # Create figure
            fig = go.Figure()
            
            # Add edges
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, color='#888'),
                hoverinfo='none',
                mode='lines'))
            
            # Add nodes
            fig.add_trace(go.Scatter(
                x=list(range(len(nodes))),
                y=[0] * len(nodes),
                mode='markers',
                marker=dict(
                    size=10,
                    color='blue',
                    line=dict(width=2, color='DarkSlateGrey')
                ),
                text=nodes,
                hoverinfo='text'
            ))
            
            # Update layout
            fig.update_layout(
                title='Network Flow Visualization',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            
            fig.show()
            
        except Exception as e:
            print(f"Error creating network visualization: {e}")
            
            # Fallback to simple table display
            print("Top Network Flows:")
            print(top_flows.head(20))
    
    def generate_summary_report(self, output_file: str = None) -> None:
        """
        Generate a comprehensive summary report of the network traffic data.
        
        Args:
            output_file (str, optional): Path to save the report. If None, displays in console.
        """
        if self.df is None:
            print("No data loaded. Please load data first.")
            return
        
        # Create report
        report = []
        report.append("# Network Traffic Analysis Summary Report")
        report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n## Dataset Overview")
        report.append(f"- Total records: {len(self.df):,}")
        report.append(f"- Number of features: {len(self.df.columns)}")
        
        # Missing values
        missing = self.df.isnull().sum()
        if missing.any() > 0:
            report.append("\n## Missing Values")
            for col, count in missing[missing > 0].items():
                report.append(f"- {col}: {count:,} ({count/len(self.df)*100:.2f}%)")
        
        # Attack distribution if available
        if self.attack_column and self.attack_column in self.df.columns:
            report.append("\n## Attack Distribution")
            attack_counts = self.df[self.attack_column].value_counts()
            for attack, count in attack_counts.items():
                report.append(f"- {attack}: {count:,} ({count/len(self.df)*100:.2f}%)")
        
        # Protocol distribution if available
        protocol_col = None
        for col in self.df.columns:
            if 'protocol' in col.lower() or 'proto' in col.lower():
                protocol_col = col
                break
        
        if protocol_col:
            report.append("\n## Protocol Distribution")
            protocol_counts = self.df[protocol_col].value_counts()
            for protocol, count in protocol_counts.items():
                report.append(f"- {protocol}: {count:,} ({count/len(self.df)*100:.2f}%)")
        
        # Statistical summary of key numeric features
        report.append("\n## Statistical Summary of Key Features")
        numeric_stats = self.df[self.numeric_columns[:10]].describe().T  # Limit to first 10 numeric columns
        report.append("```")
        report.append(numeric_stats.to_string())
        report.append("```")
        
        # Output report
        report_text = "\n".join(report)
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write(report_text)
                print(f"Report saved to {output_file}")
            except Exception as e:
                print(f"Error saving report: {e}")
                print(report_text)
        else:
            print(report_text)

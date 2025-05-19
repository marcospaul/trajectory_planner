#!/usr/bin/python3
"""Helper classes for trajectory analysis tutorial."""

import os
import glob
import argparse
import yaml
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML, IFrame
from pathlib import Path
from tqdm.auto import tqdm, trange

from trajectory_data import TrajectoryData
from trajectory_anomaly_detector import TrajectoryQualityAnalyzer, AnomalyType
from trajectory_analysis import TrajectoryAnalyzer


class ConfigLoader:
    """Load and validate configuration for trajectory analysis."""
    
    @staticmethod
    def load_config(config_path):
        """Load configuration from YAML or JSON file.
        
        :param config_path: Path to configuration file
        :return: Configuration dictionary
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Determine file type based on extension
        _, ext = os.path.splitext(config_path)
        
        if ext.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        elif ext.lower() == '.json':
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {ext}")
        
        # Validate required configuration sections
        required_sections = ['analyzer_params', 'data_directories']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        return config
    
    @staticmethod
    def create_analyzer_from_config(config):
        """Create a TrajectoryQualityAnalyzer from configuration.
        
        :param config: Configuration dictionary
        :return: Configured TrajectoryQualityAnalyzer
        """
        params = config.get('analyzer_params', {})
        
        return TrajectoryQualityAnalyzer(
            detector_type=params.get('detector_type', 'integrated'),
            max_position_jump_m=params.get('max_position_jump_m', 5.0),
            max_velocity_mps=params.get('max_velocity_mps', 40.0),
            max_acceleration_mps2=params.get('max_acceleration_mps2', 10.0),
            min_gps_jitter_m=params.get('min_gps_jitter_m', 0.1),
            max_vibration_hz=params.get('max_vibration_hz', 5.0)
        )
    
    @staticmethod
    def load_and_display_config(config_path="trajectory_config.yaml"):
        """Load configuration file and display a summary.
        
        If configuration file doesn't exist, creates a default one.
        
        :param config_path: Path to configuration file
        :return: Configuration dictionary
        """
        import yaml
        
        try:
            config = ConfigLoader.load_config(config_path)
            print("Configuration loaded successfully")
            # Display configuration summary
            print("\nAnalyzer Parameters:")
            for key, value in config['analyzer_params'].items():
                print(f"  {key}: {value}")
            
            print("\nDirectories to analyze:")
            for directory in config['data_directories']:
                print(f"  {directory}")
            
            print(f"\nAnalysis parameters: Max files per directory = {config['analysis_params']['max_files_per_dir']}")
            
        except FileNotFoundError:
            print(f"Configuration file not found: {config_path}")
            print("Using default configuration...")
            # Create default configuration
            config = {
                'analyzer_params': {
                    'detector_type': 'integrated',
                    'max_position_jump_m': 5.0,
                    'max_velocity_mps': 40.0,
                    'max_acceleration_mps2': 10.0,
                    'min_gps_jitter_m': 0.1,
                    'max_vibration_hz': 5.0
                },
                'data_directories': [
                    "./data/highway_trajectories",
                    "./data/urban_trajectories",
                    "./data/rural_trajectories"
                ],
                'analysis_params': {
                    'max_files_per_dir': 2,
                    'detailed_analysis': True,
                    'ml_analysis': True
                },
                'visualization': {
                    'map_zoom': 14,
                    'max_animation_frames': 100,
                    'figure_width': 12,
                    'figure_height': 8,
                    'dpi': 100
                }
            }
            
            # Save the default configuration to a file
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            print(f"Default configuration saved to {config_path}")
        
        return config


class TrajectoryAnalysisHelpers:
    """Helper class containing methods for analyzing trajectory data."""
    
    @staticmethod
    def create_trajectory_summary_table(trajectory):
        """Create a summary table with basic trajectory information.
        
        :param trajectory: TrajectoryData object
        :return: HTML table with trajectory summary
        """
        # Get statistics
        position_stats = trajectory.get_coordinate_stats()
        velocity_stats = trajectory.get_velocity_stats()
        accel_stats = trajectory.get_acceleration_stats()
        
        # Format HTML table
        html = f"""
        <style>
            .summary-table {{width: 100%; border-collapse: collapse; margin-bottom: 20px;}}
            .summary-table th {{background-color: #f0f0f0; text-align: left; padding: 8px;}}
            .summary-table td {{padding: 8px; border-bottom: 1px solid #ddd;}}
            .section-header {{background-color: #4a75a2; color: white; padding: 5px;}}
        </style>
        <table class="summary-table">
            <tr><th colspan="2" class="section-header">Position Information</th></tr>
            <tr><th>Number of trajectory points:</th><td>{position_stats['num_points']}</td></tr>
            <tr><th>Latitude range:</th><td>{position_stats['lat_range'][0]:.6f} to {position_stats['lat_range'][1]:.6f}</td></tr>
            <tr><th>Longitude range:</th><td>{position_stats['lon_range'][0]:.6f} to {position_stats['lon_range'][1]:.6f}</td></tr>
            <tr><th>Time range:</th><td>{position_stats['time_range']:.2f} seconds</td></tr>
        """
        
        # Add velocity information if available
        if velocity_stats.get("available", False):
            html += f"""
            <tr><th colspan="2" class="section-header">Velocity Information</th></tr>
            <tr><th>Average Speed:</th><td>{velocity_stats['avg_speed']:.2f} m/s</td></tr>
            <tr><th>Maximum Speed:</th><td>{velocity_stats['max_speed']:.2f} m/s</td></tr>
            <tr><th>Maximum North Velocity:</th><td>{velocity_stats['max_vel_north']:.2f} m/s</td></tr>
            <tr><th>Maximum East Velocity:</th><td>{velocity_stats['max_vel_east']:.2f} m/s</td></tr>
            <tr><th>Maximum Down Velocity:</th><td>{velocity_stats['max_vel_down']:.2f} m/s</td></tr>
            """
            
            # Add velocity units if available
            if velocity_stats.get("units"):
                html += "<tr><th colspan=\"2\" class=\"section-header\">Velocity Units</th></tr>"
                for component, unit in velocity_stats["units"].items():
                    html += f"<tr><th>{component}:</th><td>{unit}</td></tr>"
        
        # Add acceleration information if available
        if accel_stats.get("available", False):
            html += f"""
            <tr><th colspan="2" class="section-header">Acceleration Information</th></tr>
            <tr><th>Average Acceleration Magnitude:</th><td>{accel_stats['avg_accel_magnitude']:.2f} m/s²</td></tr>
            <tr><th>Maximum Acceleration Magnitude:</th><td>{accel_stats['max_accel_magnitude']:.2f} m/s²</td></tr>
            <tr><th>Maximum X Acceleration:</th><td>{accel_stats['max_accel_x']:.2f} m/s²</td></tr>
            <tr><th>Maximum Y Acceleration:</th><td>{accel_stats['max_accel_y']:.2f} m/s²</td></tr>
            <tr><th>Maximum Z Acceleration:</th><td>{accel_stats['max_accel_z']:.2f} m/s²</td></tr>
            """
            
            # Add acceleration units if available
            if accel_stats.get("units"):
                html += "<tr><th colspan=\"2\" class=\"section-header\">Acceleration Units</th></tr>"
                for component, unit in accel_stats["units"].items():
                    html += f"<tr><th>{component}:</th><td>{unit}</td></tr>"
        
        html += "</table>"
        return HTML(html)
    
    @staticmethod
    def plot_trajectory_components(trajectory, config=None):
        """Create plots of position, velocity, and acceleration components.
        
        :param trajectory: TrajectoryData object
        :param config: Optional configuration dictionary
        :return: Dictionary of matplotlib figures
        """
        # Get visualization settings from config or use defaults
        if config and 'visualization' in config:
            vis_config = config['visualization']
            fig_width = vis_config.get('figure_width', 12)
            fig_height = vis_config.get('figure_height', 8)
        else:
            fig_width, fig_height = 12, 8
        
        # Get statistics to check availability
        velocity_stats = trajectory.get_velocity_stats()
        accel_stats = trajectory.get_acceleration_stats()
        have_velocity = velocity_stats.get("available", False)
        have_accel = accel_stats.get("available", False)
        
        figures = {}
        
        # 1. Position (distance from start)
        fig_pos, ax_pos = plt.subplots(figsize=(fig_width, fig_height-2))
        distance = np.zeros_like(trajectory.time_diffs)
        if len(trajectory.time_diffs) > 1:
            for i in trange(1, len(trajectory.time_diffs), desc="Calculating distances", leave=False):
                dx = trajectory.longitude[i] - trajectory.longitude[i-1]
                dy = trajectory.latitude[i] - trajectory.latitude[i-1]
                distance[i] = distance[i-1] + np.sqrt(dx**2 + dy**2)
        
        ax_pos.plot(trajectory.time_diffs, distance, 'b-')
        ax_pos.set_title('Distance from Start')
        ax_pos.set_xlabel('Time (s)')
        ax_pos.set_ylabel('Distance (deg)')
        ax_pos.grid(True)
        plt.tight_layout()
        figures['position'] = fig_pos
        
        # 2. Velocity if available
        if have_velocity:
            fig_vel, ax_vel = plt.subplots(figsize=(fig_width, fig_height-2))
            velocity_time_diff = trajectory.velocity_time - trajectory.velocity_time[0]
            ax_vel.plot(velocity_time_diff, trajectory.speed_3d, 'g-')
            ax_vel.set_title('Speed vs Time')
            ax_vel.set_xlabel('Time (s)')
            unit_label = velocity_stats["units"]["speed_3d"] if velocity_stats.get("units") else "m/s"
            ax_vel.set_ylabel(f'Speed ({unit_label})')
            ax_vel.grid(True)
            plt.tight_layout()
            figures['velocity'] = fig_vel
            
            # Velocity components
            fig_vel_comp, axs = plt.subplots(3, 1, figsize=(fig_width, fig_height+2), sharex=True)
            axs[0].plot(velocity_time_diff, trajectory.vel_north, 'r-')
            axs[0].set_title('North Velocity')
            axs[0].set_ylabel(f'Velocity ({unit_label})')
            axs[0].grid(True)
            
            axs[1].plot(velocity_time_diff, trajectory.vel_east, 'g-')
            axs[1].set_title('East Velocity')
            axs[1].set_ylabel(f'Velocity ({unit_label})')
            axs[1].grid(True)
            
            axs[2].plot(velocity_time_diff, trajectory.vel_down, 'b-')
            axs[2].set_title('Down Velocity')
            axs[2].set_xlabel('Time (s)')
            axs[2].set_ylabel(f'Velocity ({unit_label})')
            axs[2].grid(True)
            
            plt.tight_layout()
            figures['velocity_components'] = fig_vel_comp
        
        # 3. Acceleration if available
        if have_accel:
            # Acceleration magnitude
            fig_accel, ax_accel = plt.subplots(figsize=(fig_width, fig_height-2))
            accel_time_diff = trajectory.acceleration_time - trajectory.acceleration_time[0]
            accel_magnitude = np.sqrt(
                trajectory.accel_x**2 + 
                trajectory.accel_y**2 + 
                trajectory.accel_z**2
            )
            ax_accel.plot(accel_time_diff, accel_magnitude, 'r-')
            ax_accel.set_title('Acceleration Magnitude vs Time')
            ax_accel.set_xlabel('Time (s)')
            unit_label = accel_stats["units"]["accel_x"] if accel_stats.get("units") else "m/s²"
            ax_accel.set_ylabel(f'Acceleration ({unit_label})')
            ax_accel.grid(True)
            plt.tight_layout()
            figures['acceleration'] = fig_accel
            
            # Acceleration components
            fig_accel_comp, axs = plt.subplots(3, 1, figsize=(fig_width, fig_height+2), sharex=True)
            axs[0].plot(accel_time_diff, trajectory.accel_x, 'r-')
            axs[0].set_title('X Acceleration')
            axs[0].set_ylabel(f'Acceleration ({unit_label})')
            axs[0].grid(True)
            
            axs[1].plot(accel_time_diff, trajectory.accel_y, 'g-')
            axs[1].set_title('Y Acceleration')
            axs[1].set_ylabel(f'Acceleration ({unit_label})')
            axs[1].grid(True)
            
            axs[2].plot(accel_time_diff, trajectory.accel_z, 'b-')
            axs[2].set_title('Z Acceleration')
            axs[2].set_xlabel('Time (s)')
            axs[2].set_ylabel(f'Acceleration ({unit_label})')
            axs[2].grid(True)
            
            plt.tight_layout()
            figures['acceleration_components'] = fig_accel_comp
        
        return figures
    
    @staticmethod
    def analyze_single_trajectory(file_path, analyzer, config=None, notebook_mode=True):
        """Perform comprehensive analysis on a single trajectory file.
        
        :param file_path: Path to trajectory file
        :param analyzer: TrajectoryQualityAnalyzer instance
        :param config: Optional configuration dictionary
        :param notebook_mode: If True, display results interactively
        :return: Dictionary of analysis results
        """
        print(f"\n\n{'='*80}")
        print(f"Analyzing file: {os.path.basename(file_path)}")
        print(f"{'='*80}\n")
        
        # Load trajectory data
        trajectory = TrajectoryData(file_path)
        trajectory.name = os.path.basename(file_path)
        
        # Create basic trajectory summary
        summary_table = TrajectoryAnalysisHelpers.create_trajectory_summary_table(trajectory)
        if notebook_mode:
            display(summary_table)
        
        # Analyze trajectory for anomalies
        print("\nDetecting anomalies...")
        with tqdm(total=5, desc="Analysis progress", leave=False) as pbar:
            quality_report = analyzer.analyze_trajectory(trajectory)
            pbar.update(2)
            
            # Create enhanced measurement statistics
            enhanced_stats = TrajectoryAnalyzer.create_enhanced_measurement_statistics_table(trajectory, quality_report)
            pbar.update(1)
            
            # Create anomaly summary
            anomaly_summary = TrajectoryAnalyzer.create_enhanced_anomaly_summary_table(quality_report)
            pbar.update(1)
            
            # Create standard measurement statistics
            std_stats = TrajectoryAnalyzer.create_measurement_statistics_table(trajectory, quality_report)
            pbar.update(1)
        
        print(f"Analysis complete. Quality score: {quality_report.quality_score:.2f}")
        
        if notebook_mode:
            # Display enhanced measurement statistics
            print("\n== Enhanced Measurement Statistics ==\n")
            display(enhanced_stats)
            
            # Display anomaly summary
            print("\n== Enhanced Anomaly Summary ==\n")
            display(anomaly_summary)
            
            # Display standard measurement statistics
            print("\n== Standard Measurement Statistics ==\n")
            display(std_stats)
            
            # Create movement component plots
            print("\n== Movement Component Plots ==\n")
            component_figures = TrajectoryAnalysisHelpers.plot_trajectory_components(trajectory, config)
            
            # Display position plot
            if 'position' in component_figures:
                display(component_figures['position'])
            
            # Display velocity plots if available
            if 'velocity' in component_figures:
                display(component_figures['velocity'])
                display(component_figures['velocity_components'])
            
            # Display acceleration plots if available
            if 'acceleration' in component_figures:
                display(component_figures['acceleration'])
                display(component_figures['acceleration_components'])
            
            # Display anomaly visualizations if anomalies were detected
            if quality_report.anomalies:
                print("\n== Anomaly Visualizations ==\n")
                
                # Multivariate anomalies plot
                print("\nMultivariate Anomalies Visualization:")
                multivariate_plot = quality_report.plot_multivariate_anomalies(trajectory)
                display(multivariate_plot)
                
                # Advanced anomaly clustering
                print("\nAdvanced Anomaly Clustering:")
                try:
                    clustering_fig = TrajectoryAnalyzer.create_advanced_anomaly_clustering(trajectory, quality_report)
                    display(clustering_fig)
                except Exception as e:
                    print(f"Advanced clustering visualization could not be generated: {e}")
                
                # Enhanced visualizations
                print("\nDetailed Quality Visualizations:")
                vis_figures = quality_report.enhanced_visualizations(trajectory)
                
                # Display each available visualization
                for vis_type in ['heatmap', 'frequency', 'distance_distribution', 'velocity_distribution', 
                                'acceleration_distribution', 'spatial_distribution', 'severity_distribution']:
                    if vis_type in vis_figures:
                        print(f"\n{vis_type.replace('_', ' ').title()} Visualization:")
                        display(vis_figures[vis_type])
                
                # Display interactive heat map if available
                try:
                    folium_filename = f"{quality_report.trajectory_name}_heatmap.html"
                    if os.path.exists(folium_filename):
                        print("\nInteractive Heat Map:")
                        display(IFrame(folium_filename, width=900, height=600))
                except Exception as e:
                    print(f"\nInteractive heat map not available: {e}")
            else:
                print("\nNo anomalies detected in this trajectory.")
        else:
            # In command-line mode, generate component figures but save rather than display them
            component_figures = TrajectoryAnalysisHelpers.plot_trajectory_components(trajectory, config)
            
            # Generate enhanced visualizations if anomalies were detected
            if quality_report.anomalies:
                quality_report.enhanced_visualizations(trajectory)
        
        return {
            'trajectory': trajectory,
            'quality_report': quality_report,
            'component_figures': component_figures
        }
    
    @staticmethod
    def analyze_directory(directory_path, analyzer, max_files=5, config=None, notebook_mode=True):
        """Analyze all trajectory files in a directory.
        
        :param directory_path: Path to directory with trajectory files
        :param analyzer: TrajectoryQualityAnalyzer instance
        :param max_files: Maximum number of files to analyze in detail
        :param config: Optional configuration dictionary
        :param notebook_mode: If True, display results interactively
        :return: Dictionary of analysis results
        """
        print(f"\n\n{'*'*100}")
        print(f"Analyzing directory: {directory_path}")
        print(f"{'*'*100}\n")
        
        # Find all MATLAB files in the directory
        mat_files = list(Path(directory_path).glob("*.mat"))
        print(f"Found {len(mat_files)} .mat files in the directory.")
        
        # First, create a summary table for the directory
        print("\nGenerating directory summary...")
        dir_summary = TrajectoryAnalyzer.create_trajectory_quality_summary_table([directory_path])
        if notebook_mode:
            display(dir_summary)
        
        # Then analyze individual files (limited to max_files)
        analysis_results = {}
        
        # Use tqdm for progress tracking
        for file_path in tqdm(mat_files[:max_files], desc=f"Analyzing files in {os.path.basename(directory_path)}"):
            results = TrajectoryAnalysisHelpers.analyze_single_trajectory(
                file_path, 
                analyzer, 
                config=config,
                notebook_mode=notebook_mode
            )
            analysis_results[str(file_path)] = results
        
        if len(mat_files) > max_files:
            print(f"\nNote: Only analyzed {max_files} out of {len(mat_files)} files. Adjust max_files parameter to analyze more.")
        
        return analysis_results
    
    @staticmethod
    def ml_enhanced_trajectory_analysis(analysis_results, config=None, notebook_mode=True):
        """Perform ML-enhanced analysis across multiple trajectories.
        
        :param analysis_results: Dictionary of analysis results from multiple trajectories
        :param config: Optional configuration dictionary
        :param notebook_mode: If True, display results interactively
        :return: Dictionary of ML analysis results
        """
        try:
            import numpy as np
            import pandas as pd
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA
            from sklearn.cluster import KMeans
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Get visualization settings from config or use defaults
            if config and 'visualization' in config:
                vis_config = config['visualization']
                fig_width = vis_config.get('figure_width', 12)
                fig_height = vis_config.get('figure_height', 8)
            else:
                fig_width, fig_height = 12, 8
            
            print("Extracting features from trajectories...")
            
            # Extract features from all trajectories
            trajectory_features = []
            trajectory_names = []
            quality_scores = []
            
            # Flatten the nested dictionaries
            flat_results = {}
            for dir_name, dir_results in analysis_results.items():
                for file_path, file_results in dir_results.items():
                    flat_results[file_path] = file_results
            
            # Collect features from each trajectory with progress bar
            for file_path, results in tqdm(flat_results.items(), desc="Processing trajectories for ML"):
                trajectory = results['trajectory']
                quality_report = results['quality_report']
                
                # Basic features
                pos_stats = trajectory.get_coordinate_stats()
                vel_stats = trajectory.get_velocity_stats()
                accel_stats = trajectory.get_acceleration_stats()
                
                # Initialize feature dictionary
                features = {
                    'num_points': pos_stats['num_points'],
                    'time_range': pos_stats['time_range'],
                    'lat_range': pos_stats['lat_range'][1] - pos_stats['lat_range'][0],
                    'lon_range': pos_stats['lon_range'][1] - pos_stats['lon_range'][0],
                    'quality_score': quality_report.quality_score,
                    'anomaly_count': len(quality_report.anomalies)
                }
                
                # Add velocity features if available
                if vel_stats.get('available', False):
                    features.update({
                        'avg_speed': vel_stats['avg_speed'],
                        'max_speed': vel_stats['max_speed'],
                        'max_vel_north': vel_stats['max_vel_north'],
                        'max_vel_east': vel_stats['max_vel_east'],
                        'max_vel_down': vel_stats['max_vel_down']
                    })
                else:
                    features.update({
                        'avg_speed': 0,
                        'max_speed': 0,
                        'max_vel_north': 0,
                        'max_vel_east': 0,
                        'max_vel_down': 0
                    })
                
                # Add acceleration features if available
                if accel_stats.get('available', False):
                    features.update({
                        'avg_accel_magnitude': accel_stats['avg_accel_magnitude'],
                        'max_accel_magnitude': accel_stats['max_accel_magnitude'],
                        'max_accel_x': accel_stats['max_accel_x'],
                        'max_accel_y': accel_stats['max_accel_y'],
                        'max_accel_z': accel_stats['max_accel_z']
                    })
                else:
                    features.update({
                        'avg_accel_magnitude': 0,
                        'max_accel_magnitude': 0,
                        'max_accel_x': 0,
                        'max_accel_y': 0,
                        'max_accel_z': 0
                    })
                
                # Add anomaly type counts
                anomaly_types = {atype.name: 0 for atype in AnomalyType}
                for anomaly in quality_report.anomalies:
                    anomaly_types[anomaly.anomaly_type.name] += 1
                
                features.update(anomaly_types)
                
                # Store features
                trajectory_features.append(features)
                trajectory_names.append(os.path.basename(file_path))
                quality_scores.append(quality_report.quality_score)
            
            # Create DataFrame
            feature_df = pd.DataFrame(trajectory_features)
            feature_df['name'] = trajectory_names
            
            if len(feature_df) < 2:
                print("Insufficient trajectories for ML analysis. Need at least 2 trajectories.")
                return {}
                
            print("Performing machine learning analysis...")
            
            with tqdm(total=4, desc="ML Analysis", leave=False) as pbar:
                # Normalize the data
                numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(feature_df[numeric_cols])
                
                # Perform PCA
                pca = PCA(n_components=min(2, len(scaled_data)))
                pca_result = pca.fit_transform(scaled_data)
                pbar.update(1)
                
                # Cluster the data
                num_clusters = min(3, len(scaled_data))
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(scaled_data)
                pbar.update(1)
                
                # Create visualization
                ml_results = {}
                
                # PCA scatter plot
                if pca_result.shape[1] >= 2:
                    fig, ax = plt.subplots(figsize=(fig_width-2, fig_height))
                    scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], 
                                       c=quality_scores, cmap='viridis', 
                                       s=100, alpha=0.7, edgecolors='k')
                    
                    # Add labels
                    for i, name in enumerate(trajectory_names):
                        ax.annotate(name, (pca_result[i, 0], pca_result[i, 1]), 
                                   fontsize=8, ha='center', va='bottom')
                    
                    # Add colorbar for quality scores
                    cbar = plt.colorbar(scatter)
                    cbar.set_label('Quality Score')
                    
                    ax.set_title('PCA of Trajectory Features (colored by quality score)')
                    ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
                    ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
                    ax.grid(alpha=0.3)
                    
                    ml_results['pca_plot'] = fig
                
                # Create cluster plot
                if pca_result.shape[1] >= 2:
                    fig, ax = plt.subplots(figsize=(fig_width-2, fig_height))
                    scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], 
                                       c=cluster_labels, cmap='tab10', 
                                       s=100, alpha=0.7, edgecolors='k')
                    
                    # Add labels
                    for i, name in enumerate(trajectory_names):
                        ax.annotate(name, (pca_result[i, 0], pca_result[i, 1]), 
                                   fontsize=8, ha='center', va='bottom')
                    
                    # Add legend for clusters
                    legend1 = ax.legend(*scatter.legend_elements(),
                                       title="Clusters")
                    ax.add_artist(legend1)
                    
                    ax.set_title('Trajectory Clusters Based on Features')
                    ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
                    ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
                    ax.grid(alpha=0.3)
                    
                    ml_results['cluster_plot'] = fig
                pbar.update(1)
                
                # Create correlation matrix
                fig, ax = plt.subplots(figsize=(fig_width+2, fig_height+4))
                corr = feature_df.select_dtypes(include=[np.number]).corr()
                mask = np.triu(np.ones_like(corr, dtype=bool))
                sns.heatmap(corr, mask=mask, cmap='coolwarm', vmax=1, vmin=-1, center=0,
                           annot=False, square=True, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
                
                ax.set_title('Feature Correlation Matrix')
                plt.tight_layout()
                ml_results['correlation_matrix'] = fig
                
                # Feature importance
                fig, ax = plt.subplots(figsize=(fig_width-2, fig_height+4))
                feature_importances = np.abs(pca.components_[0])
                feature_names = numeric_cols
                
                # Sort by importance
                sorted_idx = np.argsort(feature_importances)
                pos = np.arange(sorted_idx.shape[0]) + .5
                
                ax.barh(pos, feature_importances[sorted_idx], align='center')
                ax.set_yticks(pos)
                ax.set_yticklabels(feature_names[sorted_idx])
                ax.set_xlabel('Feature Importance (Absolute PC1 Coefficient)')
                ax.set_title('Feature Importance for Trajectory Differentiation')
                
                plt.tight_layout()
                ml_results['feature_importance'] = fig
                pbar.update(1)
            
            print("ML analysis complete.")
            
            # Display results in notebook mode
            if notebook_mode and ml_results:
                print("\n== Machine Learning Analysis Results ==\n")
                
                if 'pca_plot' in ml_results:
                    print("\nPCA of Trajectory Features:")
                    display(ml_results['pca_plot'])
                
                if 'cluster_plot' in ml_results:
                    print("\nTrajectory Clusters:")
                    display(ml_results['cluster_plot'])
                
                if 'correlation_matrix' in ml_results:
                    print("\nFeature Correlation Matrix:")
                    display(ml_results['correlation_matrix'])
                
                if 'feature_importance' in ml_results:
                    print("\nFeature Importance:")
                    display(ml_results['feature_importance'])
            
            return ml_results
        
        except Exception as e:
            print(f"Error in ML analysis: {e}")
            import traceback
            traceback.print_exc()
            return {}

    @staticmethod
    def check_and_install_dependencies():
        """Check for and install required dependencies."""
        import importlib
        import subprocess
        import sys
        
        # Core dependencies that are safe to directly import
        core_packages = [
            "numpy",
            "pandas",
            "matplotlib",
            "seaborn",
            "scipy",
            "scikit-learn",
            "folium",
            "plotly",
            "tqdm",
            "pyyaml",
            "jinja2",
        ]
        
        # Optional dependencies that may require system libraries
        optional_packages = [
            "weasyprint",
            "langchain",
            "sentence-transformers",
            "huggingface-hub",
            "pillow"
        ]
        
        print("Checking core dependencies...")
        missing_core = []
        for package in core_packages:
            try:
                importlib.import_module(package)
            except ImportError:
                missing_core.append(package)
        
        if missing_core:
            print(f"Installing missing core dependencies: {', '.join(missing_core)}")
            for package in missing_core:
                print(f"Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print("All core dependencies installed.")
        else:
            print("All core dependencies are already installed.")
        
        print("\nChecking optional dependencies...")
        for package in optional_packages:
            try:
                print(f"Checking {package}...")
                # Check if package is installed without importing it
                subprocess.check_call([
                    sys.executable, "-c", 
                    f"import pkg_resources; pkg_resources.get_distribution('{package}')"
                ], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                print(f"  ✓ {package} is installed")
            except subprocess.CalledProcessError:
                print(f"  ✗ {package} is not installed. Will be installed when needed.")
        
        # Special check for weasyprint system dependencies
        if sys.platform == 'darwin':  # macOS
            print("\nChecking system dependencies for PDF generation...")
            try:
                # Try to execute a simple WeasyPrint operation
                subprocess.check_call([
                    sys.executable, "-c", 
                    "import pkg_resources; pkg_resources.get_distribution('weasyprint')"
                ], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                
                # If we got here, try a deeper check that may trigger the system library error
                try:
                    subprocess.check_call([
                        sys.executable, "-c", 
                        "from weasyprint import HTML; HTML(string='<p>Test</p>')"
                    ], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                    print("  ✓ PDF generation libraries are properly configured")
                except subprocess.CalledProcessError:
                    print("  ✗ WeasyPrint is installed but system dependencies are missing.")
                    print("    Install system dependencies with: brew install pango gdk-pixbuf libffi")
                    print("    HTML reports will be generated instead of PDF.")
            except subprocess.CalledProcessError:
                print("  ✗ WeasyPrint is not installed. HTML reports will be generated instead of PDF.")
                print("    To enable PDF generation, install WeasyPrint and system dependencies:")
                print("    pip install weasyprint")
                print("    brew install pango gdk-pixbuf libffi")
        
        return True



    
    @staticmethod
    def check_dependencies():
        """Check and install required dependencies."""
        try:
            import importlib
            import subprocess
            import sys
            
            required_packages = [
                "numpy",
                "pandas",
                "matplotlib",
                "seaborn",
                "scipy",
                "scikit-learn",
                "folium",
                "plotly",
                "pyyaml",
                "tqdm",
            ]
            
            missing_packages = []
            for package in required_packages:
                try:
                    importlib.import_module(package)
                except ImportError:
                    missing_packages.append(package)
            
            if missing_packages:
                print(f"Installing missing dependencies: {', '.join(missing_packages)}")
                subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
                print("All required packages have been installed.")
            else:
                print("All required dependencies are already installed.")
                
        except Exception as e:
            print(f"Error checking dependencies: {e}")


    # Add these new methods to the TrajectoryAnalysisHelpers class

    @staticmethod
    def initialize_analysis(config_path="trajectory_config.yaml"):
        """Initialize the analysis environment by importing modules and loading config.
        
        :param config_path: Path to configuration file
        :return: Tuple of (config, analyzer)
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from IPython.display import display, HTML, IFrame
        from tqdm.auto import tqdm
        
        # Import custom modules
        from trajectory_data import TrajectoryData
        from trajectory_collection import TrajectoryCollection
        from trajectory_visualizer import (
            MatplotlibVisualizer,
            FoliumMapVisualizer,
            PlotlyAnimationVisualizer
        )
        from trajectory_anomaly_detector import (
            TrajectoryQualityAnalyzer,
            IntegratedAnomalyDetector,
            AnomalyType
        )
        from trajectory_analysis import TrajectoryAnalyzer
        
        # Set plotting style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['figure.dpi'] = 100
        
        # Load configuration
        config = ConfigLoader.load_and_display_config(config_path)
        
        # Create analyzer
        analyzer = ConfigLoader.create_analyzer_from_config(config)
        print(f"Analyzer created with {config['analyzer_params']['detector_type']} detector")
        
        return config, analyzer
    
    @staticmethod
    def process_all_directories(config, analyzer):
        """Process all directories specified in the configuration.
        
        :param config: Configuration dictionary
        :param analyzer: TrajectoryQualityAnalyzer instance
        :return: Dictionary of analysis results
        """
        from tqdm.auto import tqdm
        from IPython.display import display
        from trajectory_analysis import TrajectoryAnalyzer
        
        # Get the list of directories
        data_directories = config['data_directories']
        
        # Generate and display summary table across all directories
        print("\nGenerating summary across all directories...")
        summary_table = TrajectoryAnalyzer.create_trajectory_quality_summary_table(data_directories)
        display(summary_table)
        
        # Get the maximum number of files to analyze per directory from config
        max_files_per_dir = config['analysis_params']['max_files_per_dir']
        
        # Process each directory with progress tracking
        print("\nPerforming detailed analysis on each directory...")
        all_results = {}
        for directory in tqdm(data_directories, desc="Processing directories"):
            dir_results = TrajectoryAnalysisHelpers.analyze_directory(
                directory, 
                analyzer, 
                max_files=max_files_per_dir,
                config=config
            )
            all_results[directory] = dir_results
        
        return all_results
    
    @staticmethod
    def run_ml_analysis(all_results, config):
        """Run machine learning analysis on the analysis results if enabled.
        
        :param all_results: Dictionary of analysis results
        :param config: Configuration dictionary
        :return: Dictionary of ML results or None
        """
        # Check if ML analysis is enabled in config
        ml_enabled = config['analysis_params'].get('ml_analysis', True)
        
        if ml_enabled and all_results:
            # Perform ML-enhanced analysis across all trajectories
            print("\n\nPerforming machine learning analysis across all trajectories...\n")
            ml_results = TrajectoryAnalysisHelpers.ml_enhanced_trajectory_analysis(all_results, config)
            return ml_results
        else:
            print("\nML analysis is disabled in configuration or no results available for analysis.")
            return None
            
    @staticmethod
    def generate_analysis_report(config, all_results, ml_results=None, output_path='trajectory_analysis_report.html'):
        """Generate a comprehensive HTML report with all analysis results and AI-generated insights.
        
        :param config: Configuration dictionary
        :param all_results: Dictionary of analysis results per directory
        :param ml_results: Dictionary of machine learning analysis results (optional)
        :param output_path: Path to save the HTML report
        :return: Path to the generated report
        """
        import os
        import sys
        import json
        import datetime
        import tempfile
        from pathlib import Path
        import matplotlib.pyplot as plt
        from IPython.display import display, HTML
        from tqdm.auto import tqdm
        import base64
        from io import BytesIO
        import subprocess
        
        # Ensure output path has .html extension
        if not output_path.endswith('.html'):
            output_path = os.path.splitext(output_path)[0] + '.html'
        
        print("Generating comprehensive analysis report...")
        
        # Create a temporary directory for images
        temp_dir = tempfile.mkdtemp()
        image_paths = []
        
        # Function to save a figure to the temp directory and return the path
        def save_figure_to_temp(fig, name):
            path = os.path.join(temp_dir, f"{name}.png")
            fig.savefig(path, dpi=120, bbox_inches='tight')
            image_paths.append(path)
            return path
        
        # Function to convert a figure to base64 for embedding in HTML
        def fig_to_base64(fig):
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            return f"data:image/png;base64,{img_str}"
        
        # Function to convert HTML table to string for embedding in the report
        def html_to_string(html_obj):
            if hasattr(html_obj, 'data'):
                return html_obj.data
            elif hasattr(html_obj, '_repr_html_'):
                return html_obj._repr_html_()
            else:
                return str(html_obj)
        
        # 1. Prepare report data structure
        report_data = {
            'title': "Trajectory Analysis Report",
            'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'config': config,
            'summary': {
                'total_directories': len(all_results),
                'total_files': sum(len(dir_results) for dir_results in all_results.values()),
                'directories': [Path(d).name for d in all_results.keys()]  # Store directory names properly
            },
            'directory_results': {},
            'ml_insights': {} if ml_results else None,
            'ai_insights': [],
            'figures': {},
        }
        
        # 2. Process each directory's results
        print("Processing directory results...")
        for dir_path, dir_results in tqdm(all_results.items(), desc="Preparing directory data"):
            # Get proper directory name (fix empty directory name issue)
            dir_name = os.path.basename(dir_path) or dir_path
            
            dir_summary = {
                'name': dir_name,
                'path': dir_path,
                'num_files': len(dir_results),
                'files': [],
                'quality_scores': [],
                'anomaly_counts': []
            }
            
            # Process each file in the directory
            for file_path, results in dir_results.items():
                trajectory = results['trajectory']
                quality_report = results['quality_report']
                
                # Get component figures if available
                component_figures = results.get('component_figures', {})
                file_figures = {}
                
                # Convert component figures to base64
                for fig_name, fig in component_figures.items():
                    file_figures[fig_name] = fig_to_base64(fig)
                
                # Basic file info
                file_info = {
                    'name': os.path.basename(file_path),
                    'path': file_path,
                    'quality_score': quality_report.quality_score,
                    'num_anomalies': len(quality_report.anomalies),
                    'stats': {
                        'position': trajectory.get_coordinate_stats(),
                        'velocity': trajectory.get_velocity_stats(),
                        'acceleration': trajectory.get_acceleration_stats()
                    },
                    'anomalies': [],
                    'figures': file_figures,
                    'tables': {}
                }
                
                # Add quality assessment
                if quality_report.quality_score >= 0.8:
                    file_info['quality_assessment'] = "Good"
                elif quality_report.quality_score >= 0.5:
                    file_info['quality_assessment'] = "Moderate"
                else:
                    file_info['quality_assessment'] = "Poor"
                
                # Add anomalies
                for anomaly in quality_report.anomalies:
                    file_info['anomalies'].append({
                        'type': anomaly.anomaly_type.name,
                        'severity': anomaly.severity,
                        'description': anomaly.description,
                        'start_index': anomaly.start_index,
                        'end_index': anomaly.end_index
                    })
                
                # Generate tables for this file
                try:
                    # Generate enhanced measurement statistics table
                    enhanced_stats = TrajectoryAnalyzer.create_enhanced_measurement_statistics_table(
                        trajectory, quality_report)
                    file_info['tables']['enhanced_stats'] = html_to_string(enhanced_stats)
                    
                    # Generate anomaly summary table
                    anomaly_summary = TrajectoryAnalyzer.create_enhanced_anomaly_summary_table(
                        quality_report)
                    file_info['tables']['anomaly_summary'] = html_to_string(anomaly_summary)
                    
                    # Generate standard measurement statistics table
                    std_stats = TrajectoryAnalyzer.create_measurement_statistics_table(
                        trajectory, quality_report)
                    file_info['tables']['std_stats'] = html_to_string(std_stats)
                    
                    # Generate multivariate anomaly plot if there are anomalies
                    if quality_report.anomalies:
                        multivariate_plot = quality_report.plot_multivariate_anomalies(trajectory)
                        file_info['figures']['multivariate_anomalies'] = fig_to_base64(multivariate_plot)
                        
                        # Add advanced clustering if available
                        try:
                            clustering_fig = TrajectoryAnalyzer.create_advanced_anomaly_clustering(
                                trajectory, quality_report)
                            file_info['figures']['anomaly_clustering'] = fig_to_base64(clustering_fig)
                        except Exception as e:
                            print(f"Warning: Could not create anomaly clustering for {file_info['name']}: {e}")
                        
                        # Generate enhanced visualization figures
                        try:
                            vis_figures = quality_report.enhanced_visualizations(trajectory)
                            for vis_type, vis_fig in vis_figures.items():
                                file_info['figures'][f"vis_{vis_type}"] = fig_to_base64(vis_fig)
                        except Exception as e:
                            print(f"Warning: Could not create enhanced visualizations for {file_info['name']}: {e}")
                    
                except Exception as e:
                    print(f"Warning: Error generating tables/figures for {file_info['name']}: {e}")
                
                # Add file info to directory summary
                dir_summary['files'].append(file_info)
                dir_summary['quality_scores'].append(quality_report.quality_score)
                dir_summary['anomaly_counts'].append(len(quality_report.anomalies))
            
            # Calculate directory averages
            if dir_summary['quality_scores']:
                dir_summary['avg_quality_score'] = sum(dir_summary['quality_scores']) / len(dir_summary['quality_scores'])
                dir_summary['avg_anomaly_count'] = sum(dir_summary['anomaly_counts']) / len(dir_summary['anomaly_counts'])
            else:
                dir_summary['avg_quality_score'] = 1.0
                dir_summary['avg_anomaly_count'] = 0
            
            # Add directory quality assessment
            if dir_summary['avg_quality_score'] >= 0.8:
                dir_summary['quality_assessment'] = "Good"
            elif dir_summary['avg_quality_score'] >= 0.5:
                dir_summary['quality_assessment'] = "Moderate"
            else:
                dir_summary['quality_assessment'] = "Poor"
            
            # Add directory summary to report data
            report_data['directory_results'][dir_name] = dir_summary
        
        # 3. Process ML results if available
        if ml_results:
            print("Processing machine learning results...")
            report_data['ml_insights'] = {
                'available': True,
                'figures': {}
            }
            
            # Convert ML figures to base64 for embedding in HTML
            for name, fig in ml_results.items():
                report_data['ml_insights']['figures'][name] = fig_to_base64(fig)
                
                # Save figure to temp directory
                save_figure_to_temp(fig, f"ml_{name}")
        
        # 4. Generate AI insights if libraries are available
        try_llm = True
        try:
            from langchain.llms import HuggingFacePipeline
            from langchain.chains import LLMChain
            from langchain.prompts import PromptTemplate
            from transformers import pipeline
        except ImportError:
            print("LLM libraries not available. Generating report without AI insights.")
            try_llm = False
        
        if try_llm:
            print("Generating AI insights...")
            try:
                # Load a small local model for analysis
                model_name = "google/flan-t5-small"  # A lightweight model
                
                # Generate insights for each directory
                for dir_name, dir_summary in report_data['directory_results'].items():
                    # Create a text summary of the directory
                    text_summary = f"Directory: {dir_name}\n"
                    text_summary += f"Number of files: {dir_summary['num_files']}\n"
                    text_summary += f"Average quality score: {dir_summary['avg_quality_score']:.2f}\n"
                    text_summary += f"Average anomaly count: {dir_summary['avg_anomaly_count']:.2f}\n"
                    text_summary += f"Quality assessment: {dir_summary['quality_assessment']}\n\n"
                    
                    # Add file details
                    for file_info in dir_summary['files']:
                        text_summary += f"File: {file_info['name']}\n"
                        text_summary += f"Quality score: {file_info['quality_score']:.2f}\n"
                        text_summary += f"Number of anomalies: {file_info['num_anomalies']}\n"
                        
                        # Add anomaly types
                        anomaly_types = {}
                        for anomaly in file_info['anomalies']:
                            anomaly_type = anomaly['type']
                            if anomaly_type not in anomaly_types:
                                anomaly_types[anomaly_type] = 0
                            anomaly_types[anomaly_type] += 1
                        
                        if anomaly_types:
                            text_summary += "Anomaly types:\n"
                            for atype, count in anomaly_types.items():
                                text_summary += f"  - {atype}: {count}\n"
                        
                        text_summary += "\n"
                    
                    # Initialize the pipeline
                    pipe = pipeline(
                        "text2text-generation",
                        model=model_name,
                        max_length=512
                    )
                    
                    # Set up the LangChain
                    local_llm = HuggingFacePipeline(pipeline=pipe)
                    
                    # Create a prompt template
                    template = """
                    You are an expert trajectory analyst. Analyze the following trajectory data and provide insights.
                    
                    DATA:
                    {text_summary}
                    
                    Based on this data, provide:
                    1. A summary of the key findings
                    2. Potential issues to be aware of
                    3. Recommendations for improvement
                    
                    Respond with clear, concise bullet points.
                    """
                    
                    prompt = PromptTemplate(
                        input_variables=["text_summary"],
                        template=template
                    )
                    
                    # Create the chain
                    chain = LLMChain(llm=local_llm, prompt=prompt)
                    
                    # Run the chain
                    result = chain.run(text_summary=text_summary)
                    
                    # Add insights to report
                    report_data['ai_insights'].append({
                        'directory': dir_name,
                        'insights': result
                    })
                    
            except Exception as e:
                print(f"Error generating AI insights: {e}")
                print("Continuing with report generation without AI insights.")
        
        # 5. Generate CSS styling for the report
        css = """
        body {
            font-family: 'Helvetica', 'Arial', sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 20px;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #2980b9;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
        }
        h3 {
            color: #3498db;
        }
        h4 {
            color: #2c3e50;
            margin-top: 20px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 10px;
            border: 1px solid #ddd;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        .good {
            background-color: rgba(0, 255, 0, 0.2);
        }
        .moderate {
            background-color: rgba(255, 255, 0, 0.2);
        }
        .poor {
            background-color: rgba(255, 0, 0, 0.2);
        }
        .summary-box {
            background-color: #f8f9fa;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 15px;
            margin: 15px 0;
        }
        .section {
            margin-top: 30px;
        }
        img {
            max-width: 100%;
            height: auto;
            margin: 10px 0;
            border: 1px solid #ddd;
        }
        .insights {
            background-color: #e8f4f8;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
        }
        .config-table {
            font-family: monospace;
            font-size: 0.9em;
        }
        .file-section {
            background-color: #f0f7fb;
            border: 1px solid #d0e3f0;
            border-radius: 5px;
            padding: 20px;
            margin: 20px 0;
        }
        .tabs {
            display: flex;
            flex-wrap: wrap;
            margin: 20px 0;
        }
        .tab-label {
            order: 1;
            display: inline-block;
            padding: 10px 20px;
            margin-right: 2px;
            cursor: pointer;
            background: #e0e0e0;
            font-weight: bold;
            transition: background ease 0.2s;
            border-top-left-radius: 5px;
            border-top-right-radius: 5px;
        }
        .tab-label:hover {
            background: #d0d0d0;
        }
        .tab-content {
            order: 99;
            flex-grow: 1;
            width: 100%;
            display: none;
            padding: 15px;
            background: #fff;
            border: 1px solid #ddd;
        }
        .tab-input {
            display: none;
        }
        .tab-input:checked + .tab-label {
            background: #fff;
            border: 1px solid #ddd;
            border-bottom: 1px solid #fff;
            z-index: 1;
        }
        .tab-input:checked + .tab-label + .tab-content {
            display: block;
        }
        """
        
        # 6. Generate HTML report
        print("Generating HTML report...")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>{report_data['title']}</title>
            <style>{css}</style>
        </head>
        <body>
            <h1>{report_data['title']}</h1>
            <p>Generated on: {report_data['date']}</p>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <div class="summary-box">
                    <p>This report provides a comprehensive analysis of trajectory data across {report_data['summary']['total_directories']} directories,
                    examining a total of {report_data['summary']['total_files']} files.</p>
                    
                    <h3>Overview by Directory</h3>
                    <table>
                        <tr>
                            <th>Directory</th>
                            <th>Files Analyzed</th>
                            <th>Avg Quality Score</th>
                            <th>Avg Anomaly Count</th>
                            <th>Quality Assessment</th>
                        </tr>
        """
        
        # Add directory overview rows
        for dir_name, dir_summary in report_data['directory_results'].items():
            quality_class = dir_summary['quality_assessment'].lower()
            html_content += f"""
            <tr>
                <td>{dir_name}</td>
                <td>{dir_summary['num_files']}</td>
                <td class="{quality_class}">{dir_summary['avg_quality_score']:.2f}</td>
                <td>{dir_summary['avg_anomaly_count']:.1f}</td>
                <td class="{quality_class}">{dir_summary['quality_assessment']}</td>
            </tr>
            """
        
        html_content += """
                    </table>
                </div>
            </div>
            
            <div class="section">
                <h2>Configuration Settings</h2>
                <div class="summary-box">
                    <h3>Analyzer Parameters</h3>
                    <table class="config-table">
                        <tr>
                            <th>Parameter</th>
                            <th>Value</th>
                        </tr>
        """
        
        # Add analyzer parameters
        for param, value in config['analyzer_params'].items():
            html_content += f"""
            <tr>
                <td>{param}</td>
                <td>{value}</td>
            </tr>
            """
        
        html_content += """
                    </table>
                </div>
            </div>
        """
        
        # Add directory details
        html_content += """
            <div class="section">
                <h2>Directory Analysis</h2>
        """
        
        for dir_name, dir_summary in report_data['directory_results'].items():
            quality_class = dir_summary['quality_assessment'].lower()
            html_content += f"""
                <div class="section">
                    <h3>Directory: {dir_name}</h3>
                    <div class="summary-box">
                        <p>Summary: {dir_summary['num_files']} files analyzed with an average quality score of 
                        <span class="{quality_class}">{dir_summary['avg_quality_score']:.2f}</span> and
                        an average of {dir_summary['avg_anomaly_count']:.1f} anomalies per file.</p>
                        
                        <h4>Files Analyzed</h4>
                        <table>
                            <tr>
                                <th>File</th>
                                <th>Quality Score</th>
                                <th>Anomalies</th>
                                <th>Quality Assessment</th>
                            </tr>
            """
            
            # Add file rows
            for file_info in dir_summary['files']:
                quality_class = file_info['quality_assessment'].lower()
                html_content += f"""
                            <tr>
                                <td><a href="#{file_info['name'].replace('.', '_')}">{file_info['name']}</a></td>
                                <td class="{quality_class}">{file_info['quality_score']:.2f}</td>
                                <td>{file_info['num_anomalies']}</td>
                                <td class="{quality_class}">{file_info['quality_assessment']}</td>
                            </tr>
                """
            
            html_content += """
                        </table>
                    </div>
                """
            
            # AI Insights for this directory if available
            dir_insights = [insight for insight in report_data['ai_insights'] if insight['directory'] == dir_name]
            if dir_insights:
                html_content += f"""
                    <div class="insights">
                        <h4>AI-Generated Insights</h4>
                        <pre>{dir_insights[0]['insights']}</pre>
                    </div>
                """
            
            # Add details for each file
            for file_info in dir_summary['files']:
                file_id = file_info['name'].replace('.', '_')
                html_content += f"""
                    <div id="{file_id}" class="file-section">
                        <h4>File: {file_info['name']}</h4>
                        <p>Quality Score: <span class="{file_info['quality_assessment'].lower()}">{file_info['quality_score']:.2f}</span></p>
                        
                        <h5>Trajectory Information</h5>
                        <table>
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                            </tr>
                            <tr>
                                <td>Number of points</td>
                                <td>{file_info['stats']['position']['num_points']}</td>
                            </tr>
                            <tr>
                                <td>Time range</td>
                                <td>{file_info['stats']['position']['time_range']:.2f} seconds</td>
                            </tr>
                    """
                    
                # Add velocity stats if available
                if file_info['stats']['velocity'].get('available', False):
                    html_content += f"""
                        <tr>
                            <td>Average Speed</td>
                            <td>{file_info['stats']['velocity']['avg_speed']:.2f} m/s</td>
                        </tr>
                        <tr>
                            <td>Maximum Speed</td>
                            <td>{file_info['stats']['velocity']['max_speed']:.2f} m/s</td>
                        </tr>
                    """
                
                # Add acceleration stats if available
                if file_info['stats']['acceleration'].get('available', False):
                    html_content += f"""
                        <tr>
                            <td>Average Acceleration</td>
                            <td>{file_info['stats']['acceleration']['avg_accel_magnitude']:.2f} m/s²</td>
                        </tr>
                        <tr>
                            <td>Maximum Acceleration</td>
                            <td>{file_info['stats']['acceleration']['max_accel_magnitude']:.2f} m/s²</td>
                        </tr>
                    """
                
                html_content += """
                    </table>
                """
                
                # Add tabbed content for analysis details
                if file_info['tables'] or file_info['figures']:
                    html_content += """
                        <div class="tabs">
                    """
                    
                    # Add tabs for tables
                    if 'enhanced_stats' in file_info['tables']:
                        html_content += f"""
                            <input class="tab-input" type="radio" name="tabs-{file_id}" id="tab1-{file_id}" checked>
                            <label class="tab-label" for="tab1-{file_id}">Measurement Statistics</label>
                            <div class="tab-content">
                                {file_info['tables']['enhanced_stats']}
                            </div>
                        """
                    
                    if 'anomaly_summary' in file_info['tables']:
                        tab_checked = "" if 'enhanced_stats' in file_info['tables'] else "checked"
                        html_content += f"""
                            <input class="tab-input" type="radio" name="tabs-{file_id}" id="tab2-{file_id}" {tab_checked}>
                            <label class="tab-label" for="tab2-{file_id}">Anomaly Summary</label>
                            <div class="tab-content">
                                {file_info['tables']['anomaly_summary']}
                            </div>
                        """
                    
                    # Add tabs for basic plots
                    if file_info['figures']:
                        # Tab for position plots
                        if 'position' in file_info['figures']:
                            tab_checked = "" if 'enhanced_stats' in file_info['tables'] or 'anomaly_summary' in file_info['tables'] else "checked"
                            html_content += f"""
                                <input class="tab-input" type="radio" name="tabs-{file_id}" id="tab3-{file_id}" {tab_checked}>
                                <label class="tab-label" for="tab3-{file_id}">Position Plot</label>
                                <div class="tab-content">
                                    <img src="{file_info['figures']['position']}" alt="Position Plot">
                                </div>
                            """
                        
                        # Tab for velocity plots
                        if 'velocity' in file_info['figures']:
                            html_content += f"""
                                <input class="tab-input" type="radio" name="tabs-{file_id}" id="tab4-{file_id}">
                                <label class="tab-label" for="tab4-{file_id}">Velocity Plot</label>
                                <div class="tab-content">
                                    <img src="{file_info['figures']['velocity']}" alt="Velocity Plot">
                                    <img src="{file_info['figures']['velocity_components']}" alt="Velocity Components">
                                </div>
                            """
                        
                        # Tab for acceleration plots
                        if 'acceleration' in file_info['figures']:
                            html_content += f"""
                                <input class="tab-input" type="radio" name="tabs-{file_id}" id="tab5-{file_id}">
                                <label class="tab-label" for="tab5-{file_id}">Acceleration Plot</label>
                                <div class="tab-content">
                                    <img src="{file_info['figures']['acceleration']}" alt="Acceleration Plot">
                                    <img src="{file_info['figures']['acceleration_components']}" alt="Acceleration Components">
                                </div>
                            """
                        
                        # Tab for anomaly plots
                        if 'multivariate_anomalies' in file_info['figures']:
                            html_content += f"""
                                <input class="tab-input" type="radio" name="tabs-{file_id}" id="tab6-{file_id}">
                                <label class="tab-label" for="tab6-{file_id}">Anomaly Analysis</label>
                                <div class="tab-content">
                                    <img src="{file_info['figures']['multivariate_anomalies']}" alt="Multivariate Anomalies">
                            """
                            
                            if 'anomaly_clustering' in file_info['figures']:
                                html_content += f"""
                                    <img src="{file_info['figures']['anomaly_clustering']}" alt="Anomaly Clustering">
                                """
                                
                            html_content += """
                                </div>
                            """
                        
                        # Tab for additional visualizations
                        vis_plots = {k: v for k, v in file_info['figures'].items() if k.startswith('vis_')}
                        if vis_plots:
                            html_content += f"""
                                <input class="tab-input" type="radio" name="tabs-{file_id}" id="tab7-{file_id}">
                                <label class="tab-label" for="tab7-{file_id}">Detailed Visualizations</label>
                                <div class="tab-content">
                            """
                            
                            for vis_name, vis_img in vis_plots.items():
                                pretty_name = vis_name.replace('vis_', '').replace('_', ' ').title()
                                html_content += f"""
                                    <h5>{pretty_name}</h5>
                                    <img src="{vis_img}" alt="{pretty_name}">
                                """
                                
                            html_content += """
                                </div>
                            """
                    
                    html_content += """
                        </div>
                    """
                
                # Add anomalies section if there are any
                if file_info['anomalies']:
                    html_content += """
                        <h5>Detected Anomalies</h5>
                        <table>
                            <tr>
                                <th>Type</th>
                                <th>Severity</th>
                                <th>Description</th>
                            </tr>
                    """
                    
                    # Add anomalies
                    for anomaly in file_info['anomalies']:
                        severity_class = "good" if anomaly['severity'] < 0.5 else "moderate" if anomaly['severity'] < 0.8 else "poor"
                        html_content += f"""
                            <tr>
                                <td>{anomaly['type']}</td>
                                <td class="{severity_class}">{anomaly['severity']:.2f}</td>
                                <td>{anomaly['description']}</td>
                            </tr>
                        """
                    
                    html_content += """
                        </table>
                    """
                
                html_content += """
                    </div>
                """
                
            html_content += """
                </div>
            """
        
        # Add ML Insights section if available
        if ml_results:
            html_content += """
                <div class="section">
                    <h2>Machine Learning Insights</h2>
                    <div class="summary-box">
                        <p>The following visualizations show patterns and relationships identified through machine learning analysis.</p>
            """
            
            # Add ML figures
            for name, base64_img in report_data['ml_insights']['figures'].items():
                pretty_name = name.replace('_', ' ').title()
                html_content += f"""
                        <h3>{pretty_name}</h3>
                        <img src="{base64_img}" alt="{name}">
                """
            
            html_content += """
                    </div>
                </div>
            """
        
        # Add a conclusions section
        html_content += """
            <div class="section">
                <h2>Conclusions and Recommendations</h2>
                <div class="summary-box">
        """
        
        # Generate some basic conclusions based on overall quality
        overall_quality = sum(dir_summary['avg_quality_score'] for dir_summary in report_data['directory_results'].values())
        overall_quality /= len(report_data['directory_results']) if report_data['directory_results'] else 1
        
        if overall_quality >= 0.8:
            html_content += """
                    <p>Overall, the trajectory data is of <strong class="good">good quality</strong>. The following recommendations are provided:</p>
                    <ul>
                        <li>Continue monitoring for any emerging anomalies</li>
                        <li>Consider using this data for further analysis with high confidence</li>
                        <li>Document the data collection methodology for future reference</li>
                    </ul>
            """
        elif overall_quality >= 0.5:
            html_content += """
                    <p>Overall, the trajectory data is of <strong class="moderate">moderate quality</strong>. The following recommendations are provided:</p>
                    <ul>
                        <li>Investigate the sources of the identified anomalies</li>
                        <li>Consider applying filtering techniques to improve data quality</li>
                        <li>Use caution when using this data for critical applications</li>
                        <li>Review sensor calibration and data collection procedures</li>
                    </ul>
            """
        else:
            html_content += """
                    <p>Overall, the trajectory data is of <strong class="poor">poor quality</strong>. The following recommendations are provided:</p>
                    <ul>
                        <li>Investigate and address the sources of the numerous anomalies</li>
                        <li>Consider recollecting the data with improved procedures</li>
                        <li>Check for sensor malfunctions or calibration issues</li>
                        <li>Apply robust filtering techniques if the data must be used</li>
                        <li>Consult with domain experts to determine if the data is usable for your application</li>
                    </ul>
            """
        
        # Close the HTML
        html_content += """
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save the HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"HTML report saved to '{output_path}'")
        
        # Clean up temporary files
        for path in image_paths:
            try:
                os.remove(path)
            except:
                pass
        try:
            os.rmdir(temp_dir)
        except:
            pass
        
        return output_path

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description='Trajectory Analysis Tool')
    parser.add_argument('--config', type=str, default='trajectory_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--dir', type=str, 
                        help='Directory to analyze (overrides config file)')
    parser.add_argument('--max-files', type=int, default=0,
                        help='Maximum number of files to analyze per directory (0 for all)')
    parser.add_argument('--output', type=str, default='trajectory_analysis_results',
                        help='Output directory for analysis results')
    parser.add_argument('--skip-ml', action='store_true',
                        help='Skip machine learning analysis')
    
    args = parser.parse_args()
    
    # Check for dependencies
    TrajectoryAnalysisHelpers.check_dependencies()
    
    # Load configuration
    config = ConfigLoader.load_config(args.config)
    
    # Create analyzer from config
    analyzer = ConfigLoader.create_analyzer_from_config(config)
    
    # Process directories
    if args.dir:
        directories = [args.dir]
    else:
        directories = config.get('data_directories', [])
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Determine max files to analyze
    max_files = args.max_files
    if max_files <= 0:
        # Use config value or default
        max_files = config.get('analysis_params', {}).get('max_files_per_dir', 2)
    
    # Analyze each directory
    all_results = {}
    for directory in directories:
        print(f"\nAnalyzing directory: {directory}")
        dir_results = TrajectoryAnalysisHelpers.analyze_directory(
            directory, 
            analyzer, 
            max_files=max_files,
            config=config,
            notebook_mode=False  # Command line mode
        )
        all_results[directory] = dir_results
        
        # Save the directory summary
        summary_file = os.path.join(args.output, f"{os.path.basename(directory)}_summary.html")
        dir_summary = TrajectoryAnalyzer.create_trajectory_quality_summary_table([directory])
        with open(summary_file, 'w') as f:
            f.write(dir_summary.data)
        print(f"Directory summary saved to {summary_file}")
    
    # Perform ML analysis if requested
    if not args.skip_ml and len(all_results) > 0:
        print("\nPerforming machine learning analysis...")
        ml_results = TrajectoryAnalysisHelpers.ml_enhanced_trajectory_analysis(
            all_results,
            config=config,
            notebook_mode=False  # Command line mode
        )
        
        # Save ML figures
        if ml_results:
            ml_dir = os.path.join(args.output, "ml_analysis")
            os.makedirs(ml_dir, exist_ok=True)
            
            for name, fig in ml_results.items():
                fig_path = os.path.join(ml_dir, f"{name}.png")
                fig.savefig(fig_path, dpi=300)
                print(f"Saved {name} figure to {fig_path}")
    
    print(f"\nAnalysis complete. Results saved to {args.output}/")


if __name__ == "__main__":
    main()

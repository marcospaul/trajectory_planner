#!/usr/bin/python3
#
# Copyright (c) 2025 Ford Motor Company
#
# CONFIDENTIAL - FORD MOTOR COMPANY
#
# This is an unpublished work, which is a trade secret, created in 2025. Ford Motor Company owns all rights to this work and intends
# to maintain it in confidence to preserve its trade secret status. Ford Motor Company reserves the right to protect this work as an
# unpublished copyrighted work in the event of an inadvertent or deliberate unauthorized publication. Ford Motor Company also
# reserves its rights under the copyright laws to protect this work as a published work. Those having access to this work may not
# copy it, use it, or disclose the information contained in it without the written authorization of Ford Motor Company.
#
"""Classes and functions for advanced trajectory analysis and visualization."""

from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap

from trajectory_data import TrajectoryData
from trajectory_anomaly_detector import TrajectoryQualityReport


class TrajectoryAnalyzer:
    """Class for advanced trajectory analysis and reporting."""
    
    @staticmethod
    def create_enhanced_measurement_statistics_table(trajectory, quality_report):
        """Create an enhanced comprehensive table of measurement statistics with anomaly information.
        
        :param trajectory: The trajectory data
        :param quality_report: Quality report containing anomalies
        :return: Styled HTML table with statistics and anomaly information
        """
        # Prepare data structure for statistics
        stats_data = {
            'Measurement': [],
            'Min': [],
            'Max': [],
            'Mean': [],
            'Std Dev': [],
            'Has Outliers': [],
            'Outlier Severity': [],  # New column
            'Related Anomalies': [],  # New column
            'Avg Anomaly Score': [],  # New column
            'Unit': []
        }
        
        # Build a mapping of index ranges to anomalies for quick lookups
        anomaly_map = {}
        for anomaly in quality_report.anomalies:
            for idx in range(anomaly.start_index, anomaly.end_index + 1):
                if idx not in anomaly_map:
                    anomaly_map[idx] = []
                anomaly_map[idx].append(anomaly)
        
        # Position data (always available)
        # Calculate distance from start for position
        position_distances = []
        if len(trajectory.latitude) > 1:
            ref_lat, ref_lon = trajectory.latitude[0], trajectory.longitude[0]
            for i in range(len(trajectory.latitude)):
                # Approximate conversion of lat/lon to meters
                lat_dist = (trajectory.latitude[i] - ref_lat) * 111111
                lon_dist = (trajectory.longitude[i] - ref_lon) * (111111 * np.cos(np.radians(ref_lat)))
                dist = np.sqrt(lat_dist**2 + lon_dist**2)
                position_distances.append(dist)
        
        # Add position statistics
        stats_data['Measurement'].append('Position (distance)')
        stats_data['Min'].append(np.min(position_distances) if position_distances else 0)
        stats_data['Max'].append(np.max(position_distances) if position_distances else 0)
        stats_data['Mean'].append(np.mean(position_distances) if position_distances else 0)
        stats_data['Std Dev'].append(np.std(position_distances) if position_distances else 0)
        
        # Check for outliers in position data using Z-score
        has_outliers = False
        outlier_severity = 0.0
        if position_distances:
            z_scores = np.abs(stats.zscore(position_distances))
            outlier_indices = np.where(z_scores > 3)[0]
            if len(outlier_indices) > 0:
                has_outliers = True
                outlier_severity = np.mean(z_scores[outlier_indices]) / 10  # Normalize to 0-1 scale
        
        stats_data['Has Outliers'].append(has_outliers)
        stats_data['Outlier Severity'].append(outlier_severity)
        stats_data['Unit'].append('meters')
        
        # Find related anomalies for position
        position_anomalies = []
        position_anomaly_scores = []
        
        for idx in range(len(trajectory.latitude)):
            if idx in anomaly_map and anomaly_map[idx]:
                for anomaly in anomaly_map[idx]:
                    if anomaly.anomaly_type.name in ['POSITION_JUMP', 'GPS_JITTER', 'ZIGZAG_PATTERN']:
                        if anomaly.anomaly_type.name not in position_anomalies:
                            position_anomalies.append(anomaly.anomaly_type.name)
                            position_anomaly_scores.append(anomaly.severity)
        
        stats_data['Related Anomalies'].append(", ".join(position_anomalies) if position_anomalies else "None")
        stats_data['Avg Anomaly Score'].append(np.mean(position_anomaly_scores) if position_anomaly_scores else 0.0)
        
        # Derived velocity (always available)
        derived_velocities = []
        for i in range(1, len(trajectory.latitude)):
            if i < len(position_distances):
                dist_diff = position_distances[i] - position_distances[i-1]
                time_diff = max(trajectory.time[i] - trajectory.time[i-1], 0.001)
                derived_velocities.append(dist_diff / time_diff)
        
        # Add derived velocity statistics
        stats_data['Measurement'].append('Derived Velocity')
        stats_data['Min'].append(np.min(derived_velocities) if derived_velocities else 0)
        stats_data['Max'].append(np.max(derived_velocities) if derived_velocities else 0)
        stats_data['Mean'].append(np.mean(derived_velocities) if derived_velocities else 0)
        stats_data['Std Dev'].append(np.std(derived_velocities) if derived_velocities else 0)
        
        # Check for outliers in derived velocity
        has_outliers = False
        outlier_severity = 0.0
        if derived_velocities:
            z_scores = np.abs(stats.zscore(derived_velocities))
            outlier_indices = np.where(z_scores > 3)[0]
            if len(outlier_indices) > 0:
                has_outliers = True
                outlier_severity = np.mean(z_scores[outlier_indices]) / 10
        
        stats_data['Has Outliers'].append(has_outliers)
        stats_data['Outlier Severity'].append(outlier_severity)
        stats_data['Unit'].append('m/s')
        
        # Find related anomalies for derived velocity
        velocity_anomalies = []
        velocity_anomaly_scores = []
        
        for idx in range(len(trajectory.latitude) - 1):
            if idx in anomaly_map and anomaly_map[idx]:
                for anomaly in anomaly_map[idx]:
                    if anomaly.anomaly_type.name in ['UNREALISTIC_VELOCITY', 'INCONSISTENT_MOVEMENT']:
                        if anomaly.anomaly_type.name not in velocity_anomalies:
                            velocity_anomalies.append(anomaly.anomaly_type.name)
                            velocity_anomaly_scores.append(anomaly.severity)
        
        stats_data['Related Anomalies'].append(", ".join(velocity_anomalies) if velocity_anomalies else "None")
        stats_data['Avg Anomaly Score'].append(np.mean(velocity_anomaly_scores) if velocity_anomaly_scores else 0.0)
        
        # Measured velocity if available
        if hasattr(trajectory, 'speed_3d') and len(trajectory.speed_3d) > 0:
            # Add measured speed statistics
            stats_data['Measurement'].append('Measured Speed')
            stats_data['Min'].append(np.min(trajectory.speed_3d))
            stats_data['Max'].append(np.max(trajectory.speed_3d))
            stats_data['Mean'].append(np.mean(trajectory.speed_3d))
            stats_data['Std Dev'].append(np.std(trajectory.speed_3d))
            
            # Check for outliers
            z_scores = np.abs(stats.zscore(trajectory.speed_3d))
            outlier_indices = np.where(z_scores > 3)[0]
            has_outliers = len(outlier_indices) > 0
            outlier_severity = np.mean(z_scores[outlier_indices]) / 10 if has_outliers else 0.0
            
            stats_data['Has Outliers'].append(has_outliers)
            stats_data['Outlier Severity'].append(outlier_severity)
            stats_data['Unit'].append('m/s')
            
            # Find related anomalies for measured speed
            speed_anomalies = []
            speed_anomaly_scores = []
            
            for idx in range(len(trajectory.speed_3d)):
                if idx in anomaly_map and anomaly_map[idx]:
                    for anomaly in anomaly_map[idx]:
                        if anomaly.anomaly_type.name in ['UNREALISTIC_VELOCITY', 'INCONSISTENT_MOVEMENT']:
                            if anomaly.anomaly_type.name not in speed_anomalies:
                                speed_anomalies.append(anomaly.anomaly_type.name)
                                speed_anomaly_scores.append(anomaly.severity)
            
            stats_data['Related Anomalies'].append(", ".join(speed_anomalies) if speed_anomalies else "None")
            stats_data['Avg Anomaly Score'].append(np.mean(speed_anomaly_scores) if speed_anomaly_scores else 0.0)
            
            # Add velocity components if available
            if hasattr(trajectory, 'vel_north') and len(trajectory.vel_north) > 0:
                for component, name in zip(
                    [trajectory.vel_north, trajectory.vel_east, trajectory.vel_down],
                    ['North Velocity', 'East Velocity', 'Down Velocity']
                ):
                    stats_data['Measurement'].append(name)
                    stats_data['Min'].append(np.min(component))
                    stats_data['Max'].append(np.max(component))
                    stats_data['Mean'].append(np.mean(component))
                    stats_data['Std Dev'].append(np.std(component))
                    
                    # Check for outliers
                    z_scores = np.abs(stats.zscore(component))
                    outlier_indices = np.where(z_scores > 3)[0]
                    has_outliers = len(outlier_indices) > 0
                    outlier_severity = np.mean(z_scores[outlier_indices]) / 10 if has_outliers else 0.0
                    
                    stats_data['Has Outliers'].append(has_outliers)
                    stats_data['Outlier Severity'].append(outlier_severity)
                    stats_data['Unit'].append('m/s')
                    
                    # Use the same anomalies as for speed
                    stats_data['Related Anomalies'].append(", ".join(speed_anomalies) if speed_anomalies else "None")
                    stats_data['Avg Anomaly Score'].append(np.mean(speed_anomaly_scores) if speed_anomaly_scores else 0.0)
        
        # Measured acceleration if available
        if hasattr(trajectory, 'accel_x') and len(trajectory.accel_x) > 0:
            # Calculate acceleration magnitude
            accel_magnitude = np.sqrt(
                trajectory.accel_x**2 + 
                trajectory.accel_y**2 + 
                trajectory.accel_z**2
            )
            
            # Add acceleration magnitude statistics
            stats_data['Measurement'].append('Acceleration Magnitude')
            stats_data['Min'].append(np.min(accel_magnitude))
            stats_data['Max'].append(np.max(accel_magnitude))
            stats_data['Mean'].append(np.mean(accel_magnitude))
            stats_data['Std Dev'].append(np.std(accel_magnitude))
            
            # Check for outliers
            z_scores = np.abs(stats.zscore(accel_magnitude))
            outlier_indices = np.where(z_scores > 3)[0]
            has_outliers = len(outlier_indices) > 0
            outlier_severity = np.mean(z_scores[outlier_indices]) / 10 if has_outliers else 0.0
            
            stats_data['Has Outliers'].append(has_outliers)
            stats_data['Outlier Severity'].append(outlier_severity)
            stats_data['Unit'].append('m/s²')
            
            # Find related anomalies for acceleration
            accel_anomalies = []
            accel_anomaly_scores = []
            
            for idx in range(len(accel_magnitude)):
                if idx in anomaly_map and anomaly_map[idx]:
                    for anomaly in anomaly_map[idx]:
                        if anomaly.anomaly_type.name in ['UNREALISTIC_ACCELERATION', 'INCONSISTENT_MOVEMENT', 'VIBRATION']:
                            if anomaly.anomaly_type.name not in accel_anomalies:
                                accel_anomalies.append(anomaly.anomaly_type.name)
                                accel_anomaly_scores.append(anomaly.severity)
            
            stats_data['Related Anomalies'].append(", ".join(accel_anomalies) if accel_anomalies else "None")
            stats_data['Avg Anomaly Score'].append(np.mean(accel_anomaly_scores) if accel_anomaly_scores else 0.0)
            
            # Add acceleration components
            for component, name in zip(
                [trajectory.accel_x, trajectory.accel_y, trajectory.accel_z],
                ['X Acceleration', 'Y Acceleration', 'Z Acceleration']
            ):
                stats_data['Measurement'].append(name)
                stats_data['Min'].append(np.min(component))
                stats_data['Max'].append(np.max(component))
                stats_data['Mean'].append(np.mean(component))
                stats_data['Std Dev'].append(np.std(component))
                
                # Check for outliers
                z_scores = np.abs(stats.zscore(component))
                outlier_indices = np.where(z_scores > 3)[0]
                has_outliers = len(outlier_indices) > 0
                outlier_severity = np.mean(z_scores[outlier_indices]) / 10 if has_outliers else 0.0
                
                stats_data['Has Outliers'].append(has_outliers)
                stats_data['Outlier Severity'].append(outlier_severity)
                stats_data['Unit'].append('m/s²')
                
                # Use the same anomalies as for acceleration magnitude
                stats_data['Related Anomalies'].append(", ".join(accel_anomalies) if accel_anomalies else "None")
                stats_data['Avg Anomaly Score'].append(np.mean(accel_anomaly_scores) if accel_anomaly_scores else 0.0)
        
        # Create DataFrame
        stats_df = pd.DataFrame(stats_data).reset_index(drop=True)
        
        # Create HTML styled table to avoid Styler issues
        from IPython.display import HTML
        
        # Build a custom HTML table with styling
        html = f"<h3>Enhanced Measurement Statistics for {quality_report.trajectory_name}</h3>"
        html += "<table style='border-collapse: collapse; width: 100%; font-size: 14px;'>"
        
        # Add header row
        html += "<tr>"
        for col in stats_df.columns:
            html += f"<th style='background-color: #f0f0f0; color: #333; font-weight: bold; padding: 8px; border: 1px solid #ddd; text-align: center;'>{col}</th>"
        html += "</tr>"
        
        # Add data rows
        for _, row in stats_df.iterrows():
            html += "<tr>"
            
            for col in stats_df.columns:
                # Set cell style based on column and value
                style = "padding: 6px; border: 1px solid #ddd; "
                cell_value = row[col]
                
                # Format value based on column type
                if col == 'Has Outliers':
                    formatted_value = "YES" if cell_value else "NO"
                    style += "background-color: rgba(255, 0, 0, 0.2); " if cell_value else "background-color: rgba(0, 255, 0, 0.2); "
                elif col == 'Outlier Severity':
                    formatted_value = f"{cell_value:.2f}"
                    if cell_value >= 0.5:
                        style += "background-color: rgba(255, 0, 0, 0.2); "
                    elif cell_value >= 0.2:
                        style += "background-color: rgba(255, 255, 0, 0.2); "
                    else:
                        style += "background-color: rgba(0, 255, 0, 0.2); "
                elif col == 'Avg Anomaly Score':
                    formatted_value = f"{cell_value:.2f}"
                    if cell_value >= 0.8:
                        style += "background-color: rgba(255, 0, 0, 0.2); "
                    elif cell_value >= 0.5:
                        style += "background-color: rgba(255, 255, 0, 0.2); "
                    else:
                        style += "background-color: rgba(0, 255, 0, 0.2); "
                elif col == 'Related Anomalies':
                    formatted_value = cell_value
                    style += "text-align: left; "
                    if cell_value != "None":
                        style += "background-color: rgba(255, 240, 200, 0.5); font-weight: bold; "
                elif col in ['Min', 'Max', 'Mean', 'Std Dev']:
                    formatted_value = f"{cell_value:.4f}"
                else:
                    formatted_value = str(cell_value)
                
                html += f"<td style='{style}'>{formatted_value}</td>"
            
            html += "</tr>"
        
        html += "</table>"
        
        return HTML(html)
    
        
    @staticmethod
    def create_enhanced_anomaly_summary_table(quality_report):
        """Create an enhanced summary table of anomaly frequencies with detailed quality information.
        
        :param quality_report: Quality report containing anomalies
        :return: Styled HTML table with anomaly summary
        """
        if not quality_report.anomalies:
            # Create an empty DataFrame with a message
            return HTML(f"<h3>Anomaly Summary for {quality_report.trajectory_name}</h3><p>No anomalies detected in this trajectory.</p>")
        
        # Count anomalies by type
        anomaly_counts = {}
        for anomaly in quality_report.anomalies:
            anomaly_type = anomaly.anomaly_type.name
            if anomaly_type not in anomaly_counts:
                anomaly_counts[anomaly_type] = {
                    'count': 0,
                    'total_severity': 0,
                    'severities': [],
                    'indices': [],
                    'durations': []
                }
            
            anomaly_counts[anomaly_type]['count'] += 1
            anomaly_counts[anomaly_type]['total_severity'] += anomaly.severity
            anomaly_counts[anomaly_type]['severities'].append(anomaly.severity)
            anomaly_counts[anomaly_type]['indices'].append((anomaly.start_index, anomaly.end_index))
            anomaly_counts[anomaly_type]['durations'].append(anomaly.end_index - anomaly.start_index + 1)
        
        # Prepare data for the summary table
        summary_data = {
            'Anomaly Type': [],
            'Count': [],
            'Frequency (%)': [],
            'Avg Severity': [],
            'Max Severity': [],
            'Avg Duration': [],  # New column
            'Quality Impact': [],
            'Description': []  # New column
        }
        
        # Calculate total anomalies for frequency calculation
        total_anomalies = sum(info['count'] for info in anomaly_counts.values())
        
        # Anomaly type descriptions
        anomaly_descriptions = {
            'POSITION_JUMP': 'Sudden, unrealistic changes in position',
            'UNREALISTIC_VELOCITY': 'Speeds exceeding maximum expected values',
            'UNREALISTIC_ACCELERATION': 'Acceleration rates beyond physical capabilities',
            'GPS_JITTER': 'Small, random variations in stationary positions',
            'ZIGZAG_PATTERN': 'Erratic back-and-forth movements indicating noise',
            'OUTLIER': 'Statistical outliers in the data',
            'MISSING_DATA': 'Gaps in the sensor data',
            'INCONSISTENT_MOVEMENT': 'Mismatches between position, velocity, and acceleration',
            'VIBRATION': 'High-frequency oscillations in acceleration',
            'DATA_DRIFT': 'Systematic drift between sensors',
            'TEMPORAL_INCONSISTENCY': 'Issues with timing or sampling',
        }
        
        # Fill in the data
        for anomaly_type, info in anomaly_counts.items():
            summary_data['Anomaly Type'].append(anomaly_type)
            summary_data['Count'].append(info['count'])
            summary_data['Frequency (%)'].append(100 * info['count'] / total_anomalies)
            summary_data['Avg Severity'].append(info['total_severity'] / info['count'])
            summary_data['Max Severity'].append(max(info['severities']))
            summary_data['Avg Duration'].append(np.mean(info['durations']))
            
            # Calculate quality impact
            avg_severity = info['total_severity'] / info['count']
            if avg_severity >= 0.8:
                impact = "HIGH"
            elif avg_severity >= 0.5:
                impact = "MEDIUM"
            else:
                impact = "LOW"
            summary_data['Quality Impact'].append(impact)
            
            # Add description
            description = anomaly_descriptions.get(anomaly_type, "Unknown anomaly type")
            summary_data['Description'].append(description)
        
        # Create DataFrame and sort by count
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Count', ascending=False).reset_index(drop=True)
        
        # Add overall quality score row
        summary_df = pd.concat([
            summary_df,
            pd.DataFrame([{
                'Anomaly Type': 'OVERALL QUALITY',
                'Count': total_anomalies,
                'Frequency (%)': 100.0,
                'Avg Severity': np.mean([info['total_severity'] / info['count'] for info in anomaly_counts.values()]),
                'Max Severity': max([max(info['severities']) for info in anomaly_counts.values()]),
                'Avg Duration': np.mean([np.mean(info['durations']) for info in anomaly_counts.values()]),
                'Quality Impact': 'GOOD' if quality_report.quality_score >= 0.8 else 
                                 'MODERATE' if quality_report.quality_score >= 0.5 else 'POOR',
                'Description': f"Overall quality score: {quality_report.quality_score:.2f}"
            }])
        ]).reset_index(drop=True)
        
        # Create HTML styled table to avoid Styler issues
        from IPython.display import HTML
        
        # Build a custom HTML table with styling
        html = f"<h3>Enhanced Anomaly Summary for {quality_report.trajectory_name} (Quality Score: {quality_report.quality_score:.2f})</h3>"
        html += "<table style='border-collapse: collapse; width: 100%; font-size: 14px;'>"
        
        # Add header row
        html += "<tr>"
        for col in summary_df.columns:
            html += f"<th style='background-color: #f0f0f0; color: #333; font-weight: bold; padding: 8px; border: 1px solid #ddd; text-align: center;'>{col}</th>"
        html += "</tr>"
        
        # Add data rows
        for idx, row in summary_df.iterrows():
            is_last_row = idx == len(summary_df) - 1
            row_style = "font-weight: bold;" if is_last_row else ""
            
            html += f"<tr style='{row_style}'>"
            
            for col in summary_df.columns:
                # Set cell style based on column and value
                style = f"padding: 6px; border: 1px solid #ddd; {row_style} "
                cell_value = row[col]
                
                # Format value based on column type
                if col == 'Quality Impact':
                    formatted_value = cell_value
                    if cell_value in ['HIGH', 'POOR']:
                        style += "background-color: rgba(255, 0, 0, 0.2); "
                    elif cell_value in ['MEDIUM', 'MODERATE']:
                        style += "background-color: rgba(255, 255, 0, 0.2); "
                    else:
                        style += "background-color: rgba(0, 255, 0, 0.2); "
                elif col in ['Avg Severity', 'Max Severity']:
                    formatted_value = f"{cell_value:.3f}"
                    if cell_value >= 0.8:
                        style += "background-color: rgba(255, 0, 0, 0.2); "
                    elif cell_value >= 0.5:
                        style += "background-color: rgba(255, 255, 0, 0.2); "
                    else:
                        style += "background-color: rgba(0, 255, 0, 0.2); "
                elif col == 'Frequency (%)':
                    formatted_value = f"{cell_value:.1f}%"
                elif col == 'Avg Duration':
                    formatted_value = f"{cell_value:.1f} points"
                elif col == 'Description':
                    formatted_value = cell_value
                    style += "text-align: left; "
                else:
                    formatted_value = str(cell_value)
                
                html += f"<td style='{style}'>{formatted_value}</td>"
            
            html += "</tr>"
        
        html += "</table>"
        
        return HTML(html)



    
    @staticmethod
    def create_measurement_statistics_table(trajectory: TrajectoryData, quality_report: TrajectoryQualityReport) -> pd.DataFrame:
        """Create a comprehensive table of measurement statistics with outlier highlighting.
        
        :param trajectory: The trajectory data
        :param quality_report: Quality report containing anomalies
        :return: Styled pandas DataFrame with statistics
        """
        # Prepare data structure for statistics
        stats_data = {
            'Measurement': [],
            'Min': [],
            'Max': [],
            'Mean': [],
            'Std Dev': [],
            'Has Outliers': [],
            'Unit': []
        }
        
        # Position data (always available)
        # Calculate distance from start for position
        position_distances = []
        if len(trajectory.latitude) > 1:
            ref_lat, ref_lon = trajectory.latitude[0], trajectory.longitude[0]
            for i in range(len(trajectory.latitude)):
                # Approximate conversion of lat/lon to meters
                lat_dist = (trajectory.latitude[i] - ref_lat) * 111111
                lon_dist = (trajectory.longitude[i] - ref_lon) * (111111 * np.cos(np.radians(ref_lat)))
                dist = np.sqrt(lat_dist**2 + lon_dist**2)
                position_distances.append(dist)
        
        # Add position statistics
        stats_data['Measurement'].append('Position (distance)')
        stats_data['Min'].append(np.min(position_distances) if position_distances else 0)
        stats_data['Max'].append(np.max(position_distances) if position_distances else 0)
        stats_data['Mean'].append(np.mean(position_distances) if position_distances else 0)
        stats_data['Std Dev'].append(np.std(position_distances) if position_distances else 0)
        
        # Check for outliers in position data using Z-score
        has_outliers = False
        if position_distances:
            z_scores = np.abs(stats.zscore(position_distances))
            if np.any(z_scores > 3):
                has_outliers = True
        stats_data['Has Outliers'].append(has_outliers)
        stats_data['Unit'].append('meters')
        
        # Derived velocity (always available)
        derived_velocities = []
        for i in range(1, len(trajectory.latitude)):
            if i < len(position_distances):
                dist_diff = position_distances[i] - position_distances[i-1]
                time_diff = max(trajectory.time[i] - trajectory.time[i-1], 0.001)
                derived_velocities.append(dist_diff / time_diff)
        
        # Add derived velocity statistics
        stats_data['Measurement'].append('Derived Velocity')
        stats_data['Min'].append(np.min(derived_velocities) if derived_velocities else 0)
        stats_data['Max'].append(np.max(derived_velocities) if derived_velocities else 0)
        stats_data['Mean'].append(np.mean(derived_velocities) if derived_velocities else 0)
        stats_data['Std Dev'].append(np.std(derived_velocities) if derived_velocities else 0)
        
        # Check for outliers in derived velocity
        has_outliers = False
        if derived_velocities:
            z_scores = np.abs(stats.zscore(derived_velocities))
            if np.any(z_scores > 3):
                has_outliers = True
        stats_data['Has Outliers'].append(has_outliers)
        stats_data['Unit'].append('m/s')
        
        # Measured velocity if available
        if hasattr(trajectory, 'speed_3d') and len(trajectory.speed_3d) > 0:
            # Add measured speed statistics
            stats_data['Measurement'].append('Measured Speed')
            stats_data['Min'].append(np.min(trajectory.speed_3d))
            stats_data['Max'].append(np.max(trajectory.speed_3d))
            stats_data['Mean'].append(np.mean(trajectory.speed_3d))
            stats_data['Std Dev'].append(np.std(trajectory.speed_3d))
            
            # Check for outliers
            z_scores = np.abs(stats.zscore(trajectory.speed_3d))
            has_outliers = np.any(z_scores > 3)
            stats_data['Has Outliers'].append(has_outliers)
            stats_data['Unit'].append('m/s')
            
            # Add velocity components if available
            if hasattr(trajectory, 'vel_north') and len(trajectory.vel_north) > 0:
                for component, name in zip(
                    [trajectory.vel_north, trajectory.vel_east, trajectory.vel_down],
                    ['North Velocity', 'East Velocity', 'Down Velocity']
                ):
                    stats_data['Measurement'].append(name)
                    stats_data['Min'].append(np.min(component))
                    stats_data['Max'].append(np.max(component))
                    stats_data['Mean'].append(np.mean(component))
                    stats_data['Std Dev'].append(np.std(component))
                    
                    # Check for outliers
                    z_scores = np.abs(stats.zscore(component))
                    has_outliers = np.any(z_scores > 3)
                    stats_data['Has Outliers'].append(has_outliers)
                    stats_data['Unit'].append('m/s')
        
        # Measured acceleration if available
        if hasattr(trajectory, 'accel_x') and len(trajectory.accel_x) > 0:
            # Calculate acceleration magnitude
            accel_magnitude = np.sqrt(
                trajectory.accel_x**2 + 
                trajectory.accel_y**2 + 
                trajectory.accel_z**2
            )
            
            # Add acceleration magnitude statistics
            stats_data['Measurement'].append('Acceleration Magnitude')
            stats_data['Min'].append(np.min(accel_magnitude))
            stats_data['Max'].append(np.max(accel_magnitude))
            stats_data['Mean'].append(np.mean(accel_magnitude))
            stats_data['Std Dev'].append(np.std(accel_magnitude))
            
            # Check for outliers
            z_scores = np.abs(stats.zscore(accel_magnitude))
            has_outliers = np.any(z_scores > 3)
            stats_data['Has Outliers'].append(has_outliers)
            stats_data['Unit'].append('m/s²')
            
            # Add acceleration components
            for component, name in zip(
                [trajectory.accel_x, trajectory.accel_y, trajectory.accel_z],
                ['X Acceleration', 'Y Acceleration', 'Z Acceleration']
            ):
                stats_data['Measurement'].append(name)
                stats_data['Min'].append(np.min(component))
                stats_data['Max'].append(np.max(component))
                stats_data['Mean'].append(np.mean(component))
                stats_data['Std Dev'].append(np.std(component))
                
                # Check for outliers
                z_scores = np.abs(stats.zscore(component))
                has_outliers = np.any(z_scores > 3)
                stats_data['Has Outliers'].append(has_outliers)
                stats_data['Unit'].append('m/s²')
        
        # Create DataFrame
        #stats_df = pd.DataFrame(stats_data)
        stats_df = pd.DataFrame(stats_data).reset_index(drop=True)

        
        # Style the DataFrame
        def highlight_outliers(val):
            """Highlight outliers in red, normal values in green."""
            if isinstance(val, bool):
                color = 'background-color: rgba(255, 0, 0, 0.2)' if val else 'background-color: rgba(0, 255, 0, 0.2)'
                return color
            return ''
        
        # Apply the styling
        styled_df = stats_df.style.map(highlight_outliers, subset=['Has Outliers'])
        
        # Format numeric columns
        styled_df = styled_df.format({
            'Min': '{:.4f}',
            'Max': '{:.4f}',
            'Mean': '{:.4f}',
            'Std Dev': '{:.4f}'
        })
        
        # Add a caption
        styled_df = styled_df.set_caption(f"Measurement Statistics for {quality_report.trajectory_name}")
        
        # Set table style
        styled_df = styled_df.set_table_styles([
            {'selector': 'caption', 'props': [('caption-side', 'top'), ('font-weight', 'bold'), ('font-size', '16px')]},
            {'selector': 'th', 'props': [('background-color', '#f0f0f0'), ('color', '#333'), ('font-weight', 'bold')]},
            {'selector': 'td', 'props': [('padding', '5px')]},
        ])
        
        return styled_df    
    @staticmethod
    def create_anomaly_summary_table(quality_report: TrajectoryQualityReport) -> pd.DataFrame:
        """Create a summary table of anomaly frequencies with color-coding based on quality scores.
        
        :param quality_report: Quality report containing anomalies
        :return: Styled pandas DataFrame with anomaly summary
        """
        if not quality_report.anomalies:
            # Create an empty DataFrame with a message
            empty_df = pd.DataFrame({'No anomalies detected': ['This trajectory has no anomalies']})
            return empty_df.style.set_caption(f"Anomaly Summary for {quality_report.trajectory_name}")
    
        # Count anomalies by type
        anomaly_counts = {}
        for anomaly in quality_report.anomalies:
            anomaly_type = anomaly.anomaly_type.name
            if anomaly_type not in anomaly_counts:
                anomaly_counts[anomaly_type] = {
                    'count': 0,
                    'total_severity': 0,
                    'severities': []
                }
            
            anomaly_counts[anomaly_type]['count'] += 1
            anomaly_counts[anomaly_type]['total_severity'] += anomaly.severity
            anomaly_counts[anomaly_type]['severities'].append(anomaly.severity)
        
        # Prepare data for the summary table
        summary_data = {
            'Anomaly Type': [],
            'Count': [],
            'Frequency (%)': [],
            'Avg Severity': [],
            'Max Severity': [],
            'Quality Impact': []
        }
        
        # Calculate total anomalies for frequency calculation
        total_anomalies = sum(info['count'] for info in anomaly_counts.values())
        
        # Fill in the data
        for anomaly_type, info in anomaly_counts.items():
            summary_data['Anomaly Type'].append(anomaly_type)
            summary_data['Count'].append(info['count'])
            summary_data['Frequency (%)'].append(100 * info['count'] / total_anomalies)
            summary_data['Avg Severity'].append(info['total_severity'] / info['count'])
            summary_data['Max Severity'].append(max(info['severities']))
            
            # Calculate quality impact
            avg_severity = info['total_severity'] / info['count']
            if avg_severity >= 0.8:
                impact = "HIGH"
            elif avg_severity >= 0.5:
                impact = "MEDIUM"
            else:
                impact = "LOW"
            summary_data['Quality Impact'].append(impact)
        
        # Create DataFrame and sort by count
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Count', ascending=False)
        
        # Add overall quality score row
        summary_df = pd.concat([
            summary_df,
            pd.DataFrame([{
                'Anomaly Type': 'OVERALL QUALITY',
                'Count': total_anomalies,
                'Frequency (%)': 100.0,
                'Avg Severity': np.mean([info['total_severity'] / info['count'] for info in anomaly_counts.values()]),
                'Max Severity': max([max(info['severities']) for info in anomaly_counts.values()]),
                'Quality Impact': 'GOOD' if quality_report.quality_score >= 0.8 else 
                                 'MODERATE' if quality_report.quality_score >= 0.5 else 'POOR'
            }])
        ]).reset_index(drop=True) 
        
        # Style the DataFrame
        def color_quality_impact(val):
            """Color code based on quality impact."""
            if val == 'HIGH' or val == 'POOR':
                return 'background-color: rgba(255, 0, 0, 0.2)'
            elif val == 'MEDIUM' or val == 'MODERATE':
                return 'background-color: rgba(255, 255, 0, 0.2)'
            elif val == 'LOW' or val == 'GOOD':
                return 'background-color: rgba(0, 255, 0, 0.2)'
            return ''
        
        def color_severity(val):
            """Color code based on severity value."""
            if val >= 0.8:
                return 'background-color: rgba(255, 0, 0, 0.2)'
            elif val >= 0.5:
                return 'background-color: rgba(255, 255, 0, 0.2)'
            else:
                return 'background-color: rgba(0, 255, 0, 0.2)'
        
        # Apply the styling
        styled_df = summary_df.style.map(color_quality_impact, subset=['Quality Impact']) 
        styled_df = styled_df.map(color_severity, subset=['Avg Severity', 'Max Severity'])
        
        # Format numeric columns
        styled_df = styled_df.format({
            'Frequency (%)': '{:.1f}%',
            'Avg Severity': '{:.3f}',
            'Max Severity': '{:.3f}'
        })
        
        # Highlight the overall quality row
        styled_df = styled_df.apply(lambda x: ['font-weight: bold' if x.name == len(summary_df)-1 else '' for i in range(len(x))], axis=1)
        
        # Add a caption
        styled_df = styled_df.set_caption(f"Anomaly Summary for {quality_report.trajectory_name} (Quality Score: {quality_report.quality_score:.2f})")
        
        # Set table style
        styled_df = styled_df.set_table_styles([
            {'selector': 'caption', 'props': [('caption-side', 'top'), ('font-weight', 'bold'), ('font-size', '16px')]},
            {'selector': 'th', 'props': [('background-color', '#f0f0f0'), ('color', '#333'), ('font-weight', 'bold')]},
            {'selector': 'td', 'props': [('padding', '5px')]},
        ])
        
        return styled_df
    
    @staticmethod
    def create_trajectory_quality_summary_table(directories, pattern="*.mat", analyzer=None):
        """Create a summary table of quality scores for trajectory files across multiple directories.
        
        :param directories: List of directory paths to analyze
        :param pattern: File pattern to match (default: "*.mat")
        :param analyzer: Optional TrajectoryQualityAnalyzer instance (creates new one if None)
        :return: HTML table with quality score summary organized by directory
        """
        # Import required modules inside the function
        from IPython.display import HTML
        import numpy as np
        import os
        import glob
        from pathlib import Path
        from trajectory_data import TrajectoryData
        from trajectory_anomaly_detector import TrajectoryQualityAnalyzer
        
        # Create analyzer if not provided
        if analyzer is None:
            analyzer = TrajectoryQualityAnalyzer(
                detector_type="integrated",
                max_position_jump_m=5.0,
                max_velocity_mps=40.0,
                max_acceleration_mps2=10.0,
                min_gps_jitter_m=0.1,
                max_vibration_hz=5.0
            )
        
        # Initialize data structure to store results organized by directory
        directory_results = {}
        
        # Process each directory
        for directory in directories:
            # Create full path to directory
            dir_path = Path(directory)
            dir_name = dir_path.name
            
            # Find all files matching the pattern in this directory
            file_paths = list(dir_path.glob(pattern))
            
            # Results for this directory
            results = []
            
            # Process each file
            for file_path in file_paths:
                try:
                    # Extract file name from path
                    file_name = file_path.name
                    
                    # Load trajectory data
                    trajectory = TrajectoryData(file_path)
                    trajectory.name = file_name
                    
                    # Analyze trajectory
                    quality_report = analyzer.analyze_trajectory(trajectory)
                    
                    # Calculate quality metrics
                    if quality_report.anomalies:
                        quality_scores = [1.0 - (anomaly.severity * 0.1) for anomaly in quality_report.anomalies]
                        avg_quality = np.mean(quality_scores)
                        min_quality = np.min(quality_scores)
                        max_quality = np.max(quality_scores)
                        std_quality = np.std(quality_scores)
                    else:
                        # Perfect score if no anomalies
                        avg_quality = 1.0
                        min_quality = 1.0
                        max_quality = 1.0
                        std_quality = 0.0
                    
                    # Determine anomaly status
                    if avg_quality < 0.5:
                        anomaly_status = "Critical Anomaly Found"
                    else:
                        anomaly_status = "No Critical Anomaly"
                    
                    # Store results
                    results.append({
                        'file_name': file_name,
                        'anomaly_status': anomaly_status,
                        'avg_quality': avg_quality,
                        'min_quality': min_quality,
                        'max_quality': max_quality,
                        'std_quality': std_quality
                    })
                    
                except Exception as e:
                    # Handle errors
                    results.append({
                        'file_name': file_path.name,
                        'anomaly_status': f"Error: {str(e)}",
                        'avg_quality': 0.0,
                        'min_quality': 0.0,
                        'max_quality': 0.0,
                        'std_quality': 0.0
                    })
            
            # Store results for this directory if any files were processed
            if results:
                directory_results[dir_name] = results
        
        # Create HTML table with directory grouping
        html = """
        <style>
            .quality-table {
                border-collapse: collapse;
                width: 100%;
                font-family: Arial, sans-serif;
                margin-bottom: 30px;
            }
            .quality-table th, .quality-table td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            .quality-table th {
                background-color: #f2f2f2;
                font-weight: bold;
            }
            .quality-table tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            .good-quality {
                background-color: rgba(0, 255, 0, 0.2);
            }
            .medium-quality {
                background-color: rgba(255, 255, 0, 0.2);
            }
            .poor-quality {
                background-color: rgba(255, 0, 0, 0.2);
            }
            .directory-header {
                background-color: #4a75a2;
                color: white;
                padding: 10px;
                font-size: 16px;
                font-weight: bold;
                margin-top: 20px;
                border-radius: 5px;
            }
            .summary-header {
                background-color: #2c3e50;
                color: white;
                padding: 12px;
                font-size: 18px;
                font-weight: bold;
                text-align: center;
                margin-bottom: 10px;
                border-radius: 5px;
            }
        </style>
        <div class="summary-header">Trajectory Quality Summary Across Directories</div>
        """
        
        # Process each directory
        for dir_name, results in directory_results.items():
            # Sort results by quality score
            sorted_results = sorted(results, key=lambda x: x['avg_quality'])
            
            # Add directory header
            html += f'<div class="directory-header">Directory: {dir_name} ({len(results)} files)</div>'
            
            # Create table for this directory
            html += """
            <table class="quality-table">
                <tr>
                    <th>File Name</th>
                    <th>Anomaly Status</th>
                    <th>Average Quality Score</th>
                    <th>Min Quality Score</th>
                    <th>Max Quality Score</th>
                    <th>Std Deviation</th>
                </tr>
            """
            
            # Add rows for each file
            for result in sorted_results:
                # Determine color class based on average quality score
                if result['avg_quality'] >= 0.8:
                    color_class = "good-quality"
                elif result['avg_quality'] >= 0.5:
                    color_class = "medium-quality"
                else:
                    color_class = "poor-quality"
                
                # Add row for this result
                html += f"""
                <tr>
                    <td>{result['file_name']}</td>
                    <td>{result['anomaly_status']}</td>
                    <td class="{color_class}">{result['avg_quality']:.3f}</td>
                    <td>{result['min_quality']:.3f}</td>
                    <td>{result['max_quality']:.3f}</td>
                    <td>{result['std_quality']:.3f}</td>
                </tr>
                """
            
            html += "</table>"
        
        # Add overall summary if multiple directories
        if len(directory_results) > 1:
            html += '<div class="directory-header">Overall Summary</div>'
            html += """
            <table class="quality-table">
                <tr>
                    <th>Directory</th>
                    <th>Files Analyzed</th>
                    <th>Files with Critical Anomalies</th>
                    <th>Average Quality Score</th>
                </tr>
            """
            
            for dir_name, results in directory_results.items():
                # Calculate summary statistics
                total_files = len(results)
                critical_anomalies = sum(1 for r in results if r['avg_quality'] < 0.5)
                avg_dir_quality = np.mean([r['avg_quality'] for r in results])
                
                # Determine color class
                if avg_dir_quality >= 0.8:
                    color_class = "good-quality"
                elif avg_dir_quality >= 0.5:
                    color_class = "medium-quality"
                else:
                    color_class = "poor-quality"
                
                html += f"""
                <tr>
                    <td>{dir_name}</td>
                    <td>{total_files}</td>
                    <td>{critical_anomalies}</td>
                    <td class="{color_class}">{avg_dir_quality:.3f}</td>
                </tr>
                """
            
            html += "</table>"
        
        return HTML(html)

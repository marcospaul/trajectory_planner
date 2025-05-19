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
"""Classes for detecting anomalies in trajectory data using integrated movement data."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union, NamedTuple
from dataclasses import dataclass
from enum import Enum, auto
import math
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

from trajectory_data import TrajectoryData
from trajectory_collection import TrajectoryCollection

from typing import Dict  # Add this if not already present
import matplotlib.pyplot as plt  # Add this if not already present

class AnomalyType(Enum):
    """Enumeration of different types of anomalies that can be detected."""
    
    POSITION_JUMP = auto()
    UNREALISTIC_VELOCITY = auto()
    UNREALISTIC_ACCELERATION = auto()
    GPS_JITTER = auto()
    OUTLIER = auto()
    MISSING_DATA = auto()
    ZIGZAG_PATTERN = auto()
    INCONSISTENT_MOVEMENT = auto()  # New: when position/velocity/acceleration don't align
    VIBRATION = auto()  # New: high-frequency, low-amplitude oscillations
    DATA_DRIFT = auto()  # New: systematic drift between sensors
    TEMPORAL_INCONSISTENCY = auto()  # New: time-related issues


@dataclass
class AnomalyInfo:
    """Class containing information about a detected anomaly."""
    
    anomaly_type: AnomalyType
    start_index: int
    end_index: int
    severity: float  # 0.0 to 1.0 scale where 1.0 is most severe
    description: str
    additional_data: Dict[str, Any] = None


class TrajectoryQualityReport:
    """Class for storing and displaying the quality report for a trajectory."""
    
    def __init__(self, trajectory_name: str) -> None:
        """Initialize a new quality report.
        
        :param trajectory_name: Name of the trajectory being analyzed
        """
        self.trajectory_name = trajectory_name
        self.anomalies: List[AnomalyInfo] = []
        self.summary_stats: Dict[str, Any] = {}
        self.quality_score = 1.0  # Perfect score by default
    
    def add_anomaly(self, anomaly: AnomalyInfo) -> None:
        """Add an anomaly to the report.
        
        :param anomaly: Information about the detected anomaly
        """
        self.anomalies.append(anomaly)
        # Update quality score based on severity
        self.quality_score = min(self.quality_score, 1.0 - (anomaly.severity * 0.1))
    
    def add_summary_statistic(self, name: str, value: Any) -> None:
        """Add a summary statistic to the report.
        
        :param name: Name of the statistic
        :param value: Value of the statistic
        """
        self.summary_stats[name] = value
    
    def is_good_quality(self) -> bool:
        """Check if the trajectory has good quality.
        
        :return: True if no severe anomalies were detected
        """
        return self.quality_score >= 0.8
    
    def get_report_text(self) -> str:
        """Generate a text report of the trajectory quality.
        
        :return: Formatted text report
        """
        report_lines = [
            f"Quality Report for Trajectory: {self.trajectory_name}",
            f"Overall Quality Score: {self.quality_score:.2f} out of 1.00",
            f"Status: {'GOOD' if self.is_good_quality() else 'NEEDS REVIEW'}",
            "\nSummary Statistics:",
        ]
        
        for stat_name, stat_value in sorted(self.summary_stats.items()):
            if isinstance(stat_value, float):
                report_lines.append(f"  {stat_name}: {stat_value:.4f}")
            else:
                report_lines.append(f"  {stat_name}: {stat_value}")
        
        if self.anomalies:
            report_lines.append("\nDetected Anomalies:")
            for i, anomaly in enumerate(sorted(self.anomalies, key=lambda x: x.severity, reverse=True), 1):
                report_lines.append(f"  {i}. {anomaly.anomaly_type.name} (Severity: {anomaly.severity:.2f})")
                report_lines.append(f"     {anomaly.description}")
                report_lines.append(f"     Location: Indices {anomaly.start_index} to {anomaly.end_index}")
        else:
            report_lines.append("\nNo anomalies detected. The trajectory data appears consistent.")
        
        return "\n".join(report_lines)
    
    def plot_anomalies(self, trajectory: TrajectoryData) -> plt.Figure:
        """Create a visualization of the trajectory with anomalies highlighted.
        
        :param trajectory: The trajectory data to visualize
        :return: Matplotlib figure showing the trajectory with anomalies highlighted
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot the full path
        ax.plot(
            trajectory.longitude, 
            trajectory.latitude, 
            'b-', 
            linewidth=2,
            alpha=0.6,
            label="Trajectory"
        )
        
        # Highlight anomalies
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.anomalies)))
        for i, anomaly in enumerate(self.anomalies):
            start_idx = anomaly.start_index
            end_idx = min(anomaly.end_index + 1, len(trajectory.latitude))
            
            ax.plot(
                trajectory.longitude[start_idx:end_idx],
                trajectory.latitude[start_idx:end_idx],
                'o-',
                color=colors[i],
                linewidth=3,
                markersize=8,
                label=f"{anomaly.anomaly_type.name} (Severity: {anomaly.severity:.2f})"
            )
        
        # Add labels and style
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Trajectory Anomalies: {self.trajectory_name}")
        ax.grid(True)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def enhanced_visualizations(self, trajectory: TrajectoryData) -> Dict[str, plt.Figure]:
        """Create enhanced visualizations for anomaly analysis with heat maps and statistics.
        
        :param trajectory: The trajectory data to visualize
        :return: Dictionary of matplotlib figures with different visualizations
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.colors import LinearSegmentedColormap
        from scipy.stats import gaussian_kde
        import folium
        from folium.plugins import HeatMap
        import pandas as pd
        from collections import Counter
        
        # Check if we have anomalies to visualize
        if not self.anomalies:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No anomalies detected", ha='center', va='center')
            return {"no_anomalies": fig}
        
        # Create a dictionary to store all visualizations
        visualizations = {}
        
        # [Rest of the function code goes here - same as before]
        
        return visualizations
        
    def enhanced_anomaly_visualization(trajectory: TrajectoryData, report: TrajectoryQualityReport) -> Dict[str, plt.Figure]:
        """Create enhanced visualizations for anomaly analysis.
        
        :param trajectory: The trajectory data to visualize
        :param report: The quality report containing anomalies
        :return: Dictionary of matplotlib figures with different visualizations
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.colors import LinearSegmentedColormap
        from scipy.stats import gaussian_kde
        import folium
        from folium.plugins import HeatMap
        import pandas as pd
        from collections import Counter
        
        # Check if we have anomalies to visualize
        if not report.anomalies:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No anomalies detected", ha='center', va='center')
            return {"no_anomalies": fig}
        
        # Create a dictionary to store all visualizations
        visualizations = {}
        
        # 1. Prepare data for visualizations
        # Create a DataFrame with all anomalies
        anomaly_data = []
        
        for anomaly in report.anomalies:
            # Get all points in the anomaly range
            for idx in range(anomaly.start_index, anomaly.end_index + 1):
                if idx < len(trajectory.latitude):
                    entry = {
                        "latitude": trajectory.latitude[idx],
                        "longitude": trajectory.longitude[idx],
                        "type": anomaly.anomaly_type.name,
                        "severity": anomaly.severity,
                        "index": idx,
                        "time": trajectory.time[idx] if idx < len(trajectory.time) else 0
                    }
                    
                    # Add velocity information if available
                    if hasattr(trajectory, 'speed_3d') and idx < len(trajectory.speed_3d):
                        entry["speed"] = trajectory.speed_3d[idx]
                        entry["vel_north"] = trajectory.vel_north[idx] if idx < len(trajectory.vel_north) else 0
                        entry["vel_east"] = trajectory.vel_east[idx] if idx < len(trajectory.vel_east) else 0
                    
                    # Add acceleration information if available
                    if hasattr(trajectory, 'accel_x') and idx < len(trajectory.accel_x):
                        entry["accel_x"] = trajectory.accel_x[idx]
                        entry["accel_y"] = trajectory.accel_y[idx]
                        entry["accel_z"] = trajectory.accel_z[idx]
                        entry["accel_magnitude"] = np.sqrt(
                            trajectory.accel_x[idx]**2 + 
                            trajectory.accel_y[idx]**2 + 
                            trajectory.accel_z[idx]**2
                        )
                    
                    anomaly_data.append(entry)
        
        if not anomaly_data:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Anomaly data could not be processed", ha='center', va='center')
            return {"error": fig}
        
        anomalies_df = pd.DataFrame(anomaly_data)
        
        # Calculate distance from start for each point
        first_lat, first_lon = trajectory.latitude[0], trajectory.longitude[0]
        distances = []
        
        for idx in range(len(trajectory.latitude)):
            lat, lon = trajectory.latitude[idx], trajectory.longitude[idx]
            # Simple approximation for distance
            # 111,111 meters per degree of latitude
            # 111,111 * cos(latitude) meters per degree of longitude
            lat_dist = (lat - first_lat) * 111111
            lon_dist = (lon - first_lon) * (111111 * np.cos(np.radians(first_lat)))
            distances.append((lat_dist, lon_dist))
        
        distances = np.array(distances)
        
        # Add distance information to anomalies
        for i, row in anomalies_df.iterrows():
            idx = row['index']
            if idx < len(distances):
                anomalies_df.at[i, 'lat_dist'] = distances[idx, 0]
                anomalies_df.at[i, 'lon_dist'] = distances[idx, 1]
        
        # 2. Create heat map of anomaly frequency overlaid on trajectory
        fig1, ax1 = plt.subplots(figsize=(12, 10))
        
        # Plot the trajectory
        ax1.plot(trajectory.longitude, trajectory.latitude, 'k-', alpha=0.3, linewidth=1)
        
        # Create a 2D histogram for the heatmap
        if len(anomalies_df) > 5:  # Need at least a few points for meaningful density
            # Create a custom colormap from blue to red
            colors = [(0, 0, 0.8), (0, 0.8, 0.8), (0.8, 0.8, 0), (0.8, 0, 0)]
            cmap_name = 'anomaly_density'
            cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
            
            # Use kernel density estimation for smoother results
            try:
                # Create grid of points
                x_min, x_max = trajectory.longitude.min(), trajectory.longitude.max()
                y_min, y_max = trajectory.latitude.min(), trajectory.latitude.max()
                
                # Add some margin
                margin_x = (x_max - x_min) * 0.05
                margin_y = (y_max - y_min) * 0.05
                x_min -= margin_x
                x_max += margin_x
                y_min -= margin_y
                y_max += margin_y
                
                xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                
                # Kernel density estimation
                values = np.vstack([anomalies_df['longitude'], anomalies_df['latitude']])
                kernel = gaussian_kde(values)
                f = np.reshape(kernel(positions).T, xx.shape)
                
                # Plot the kernel density estimate
                ax1.contourf(xx, yy, f, cmap=cm, alpha=0.7)
                
                # Add a color bar
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider = make_axes_locatable(ax1)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(ax1.contourf(xx, yy, f, cmap=cm, alpha=0), cax=cax, label="Anomaly Density")
            except Exception as e:
                # Fall back to scatter plot if KDE fails
                print(f"KDE failed: {e}. Falling back to scatter plot.")
                scatter = ax1.scatter(
                    anomalies_df['longitude'], 
                    anomalies_df['latitude'],
                    c=anomalies_df['severity'],
                    cmap='hot',
                    alpha=0.6,
                    s=50
                )
                plt.colorbar(scatter, label="Anomaly Severity")
        else:
            # For very few points, just use a scatter plot
            scatter = ax1.scatter(
                anomalies_df['longitude'], 
                anomalies_df['latitude'],
                c=anomalies_df['severity'],
                cmap='hot',
                alpha=0.6,
                s=50
            )
            plt.colorbar(scatter, label="Anomaly Severity")
        
        # Customize the plot
        ax1.set_title(f"Anomaly Heat Map for {report.trajectory_name}")
        ax1.set_xlabel("Longitude")
        ax1.set_ylabel("Latitude")
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        
        visualizations["heatmap"] = fig1
        
        # 3. Create interactive Folium heatmap
        try:
            center_lat = np.mean(trajectory.latitude)
            center_lon = np.mean(trajectory.longitude)
            
            folium_map = folium.Map(location=[center_lat, center_lon], zoom_start=13)
            
            # Add trajectory line
            folium.PolyLine(
                locations=[[lat, lon] for lat, lon in zip(trajectory.latitude, trajectory.longitude)],
                color='blue',
                weight=2,
                opacity=0.7
            ).add_to(folium_map)
            
            # Add heat map layer
            heat_data = [[row.latitude, row.longitude, row.severity] for _, row in anomalies_df.iterrows()]
            HeatMap(heat_data, radius=15).add_to(folium_map)
            
            # Save to HTML for viewing in the notebook
            folium_filename = f"{report.trajectory_name}_heatmap.html"
            folium_map.save(folium_filename)
            print(f"Interactive heat map saved to {folium_filename}")
        except Exception as e:
            print(f"Folium map creation failed: {e}")
        
        # 4. Anomaly frequency by type
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        
        # Count occurrences of each anomaly type
        anomaly_counts = Counter(anomalies_df['type'])
        
        # Sort by frequency
        types, counts = zip(*sorted(anomaly_counts.items(), key=lambda x: x[1], reverse=True))
        
        # Create bar chart
        bars = ax2.bar(types, counts, color='skyblue')
        
        # Add count labels on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{counts[i]}', ha='center', va='bottom')
        
        # Customize plot
        ax2.set_title("Frequency of Anomaly Types")
        ax2.set_xlabel("Anomaly Type")
        ax2.set_ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        visualizations["frequency"] = fig2
        
        # 5. Distribution of anomalies by lateral and longitudinal distance
        if 'lat_dist' in anomalies_df.columns and 'lon_dist' in anomalies_df.columns:
            fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Color by anomaly type
            anomaly_types = anomalies_df['type'].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(anomaly_types)))
            color_map = dict(zip(anomaly_types, colors))
            
            # Lateral distance distribution
            for atype in anomaly_types:
                subset = anomalies_df[anomalies_df['type'] == atype]
                sns.kdeplot(data=subset, x='lat_dist', ax=ax3a, label=atype, color=color_map[atype])
            
            ax3a.set_title("Lateral Distance Distribution of Anomalies")
            ax3a.set_xlabel("Lateral Distance from Start (m)")
            ax3a.set_ylabel("Density")
            ax3a.grid(True, alpha=0.3)
            
            # Longitudinal distance distribution
            for atype in anomaly_types:
                subset = anomalies_df[anomalies_df['type'] == atype]
                sns.kdeplot(data=subset, x='lon_dist', ax=ax3b, label=atype, color=color_map[atype])
            
            ax3b.set_title("Longitudinal Distance Distribution of Anomalies")
            ax3b.set_xlabel("Longitudinal Distance from Start (m)")
            ax3b.set_ylabel("Density")
            ax3b.grid(True, alpha=0.3)
            
            # Create shared legend
            handles, labels = ax3a.get_legend_handles_labels()
            fig3.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=3)
            
            # Remove duplicate legends
            ax3a.get_legend().remove()
            ax3b.get_legend().remove()
            
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.15)  # Make space for the common legend
            
            visualizations["distance_distribution"] = fig3
        
        # 6. Create velocity and acceleration distributions if available
        if 'speed' in anomalies_df.columns:
            # Velocity distribution by anomaly type
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            
            for atype in anomaly_types:
                subset = anomalies_df[anomalies_df['type'] == atype]
                if not subset.empty:
                    sns.kdeplot(data=subset, x='speed', ax=ax4, label=atype, color=color_map[atype])
            
            ax4.set_title("Velocity Distribution by Anomaly Type")
            ax4.set_xlabel("Speed (m/s)")
            ax4.set_ylabel("Density")
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            
            plt.tight_layout()
            visualizations["velocity_distribution"] = fig4
        
        if 'accel_magnitude' in anomalies_df.columns:
            # Acceleration distribution by anomaly type
            fig5, ax5 = plt.subplots(figsize=(10, 6))
            
            for atype in anomaly_types:
                subset = anomalies_df[anomalies_df['type'] == atype]
                if not subset.empty:
                    sns.kdeplot(data=subset, x='accel_magnitude', ax=ax5, label=atype, color=color_map[atype])
            
            ax5.set_title("Acceleration Distribution by Anomaly Type")
            ax5.set_xlabel("Acceleration Magnitude (m/s²)")
            ax5.set_ylabel("Density")
            ax5.grid(True, alpha=0.3)
            ax5.legend()
            
            plt.tight_layout()
            visualizations["acceleration_distribution"] = fig5
        
        # 7. Create a 2D scatter plot of anomalies by lat/lon distance colored by type
        if 'lat_dist' in anomalies_df.columns and 'lon_dist' in anomalies_df.columns:
            fig6, ax6 = plt.subplots(figsize=(10, 8))
            
            for atype, color in zip(anomaly_types, colors):
                subset = anomalies_df[anomalies_df['type'] == atype]
                ax6.scatter(subset['lon_dist'], subset['lat_dist'], c=[color], label=atype, alpha=0.7, s=50)
            
            # Plot trajectory path in distance coordinates
            path_x = distances[:, 1]  # lon_dist
            path_y = distances[:, 0]  # lat_dist
            ax6.plot(path_x, path_y, 'k--', alpha=0.3, linewidth=1)
            
            # Mark start and end
            ax6.scatter([path_x[0]], [path_y[0]], c='green', s=100, marker='^', label='Start')
            ax6.scatter([path_x[-1]], [path_y[-1]], c='red', s=100, marker='s', label='End')
            
            ax6.set_title("Spatial Distribution of Anomalies")
            ax6.set_xlabel("Longitudinal Distance (m)")
            ax6.set_ylabel("Lateral Distance (m)")
            ax6.grid(True, alpha=0.3)
            ax6.legend()
            ax6.axis('equal')  # Equal scaling
            
            plt.tight_layout()
            visualizations["spatial_distribution"] = fig6
        
        # 8. Create a severity distribution plot
        fig7, ax7 = plt.subplots(figsize=(10, 6))
        
        sns.boxplot(data=anomalies_df, x='type', y='severity', ax=ax7)
        ax7.set_title("Anomaly Severity by Type")
        ax7.set_xlabel("Anomaly Type")
        ax7.set_ylabel("Severity")
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        visualizations["severity_distribution"] = fig7
        
        # Return all visualizations
        return visualizations

    
    def plot_multivariate_anomalies(self, trajectory: TrajectoryData) -> plt.Figure:
        """Create detailed visualization of anomalies across position, velocity, and acceleration.
        
        :param trajectory: The trajectory data to visualize
        :return: Matplotlib figure showing anomalies in multiple dimensions
        """
        # Determine what data we have available
        has_velocity = len(trajectory.speed_3d) > 0
        has_acceleration = len(trajectory.accel_x) > 0
        
        # Create main figure
        fig = plt.figure(figsize=(14, 10))
        
        # Create grid of subplots based on available data
        num_plots = 1 + int(has_velocity) + int(has_acceleration)
        grid_spec = plt.GridSpec(num_plots, 1, height_ratios=[1] * num_plots)
        
        # Create position plot (top)
        ax_pos = fig.add_subplot(grid_spec[0])
        
        # Calculate distance from start for position plot
        distance = np.zeros_like(trajectory.time_diffs)
        if len(trajectory.time_diffs) > 1:
            for i in range(1, len(trajectory.time_diffs)):
                dx = trajectory.longitude[i] - trajectory.longitude[i-1]
                dy = trajectory.latitude[i] - trajectory.latitude[i-1]
                distance[i] = distance[i-1] + np.sqrt(dx**2 + dy**2)
        
        # Plot position data
        ax_pos.plot(trajectory.time_diffs, distance, 'b-', alpha=0.7)
        ax_pos.set_ylabel('Distance')
        ax_pos.set_title('Position (Distance from Start)')
        ax_pos.grid(True)
        
        # Create velocity plot if available
        if has_velocity:
            ax_vel = fig.add_subplot(grid_spec[1], sharex=ax_pos)
            
            # Create relative time for velocity data
            vel_time = trajectory.velocity_time - trajectory.velocity_time[0]
            
            # Plot velocity data
            ax_vel.plot(vel_time, trajectory.speed_3d, 'g-', alpha=0.7)
            ax_vel.set_ylabel('Speed')
            ax_vel.set_title('Velocity (Speed)')
            ax_vel.grid(True)
        
        # Create acceleration plot if available
        if has_acceleration:
            ax_idx = 1 + int(has_velocity)
            ax_acc = fig.add_subplot(grid_spec[ax_idx], sharex=ax_pos)
            
            # Create relative time for acceleration data
            acc_time = trajectory.acceleration_time - trajectory.acceleration_time[0]
            
            # Calculate acceleration magnitude
            acc_mag = np.sqrt(
                trajectory.accel_x**2 + 
                trajectory.accel_y**2 + 
                trajectory.accel_z**2
            )
            
            # Plot acceleration data
            ax_acc.plot(acc_time, acc_mag, 'r-', alpha=0.7)
            ax_acc.set_ylabel('Acceleration')
            ax_acc.set_title('Acceleration (Magnitude)')
            ax_acc.grid(True)
            ax_acc.set_xlabel('Time (s)')
        else:
            # Add x-label to last plot
            if has_velocity:
                ax_vel.set_xlabel('Time (s)')
            else:
                ax_pos.set_xlabel('Time (s)')
        
        # Highlight anomalies on all relevant plots
        for anomaly in self.anomalies:
            start_idx = anomaly.start_index
            end_idx = anomaly.end_index
            
            # Get time range for this anomaly
            if start_idx < len(trajectory.time_diffs) and end_idx < len(trajectory.time_diffs):
                t_start = trajectory.time_diffs[start_idx]
                t_end = trajectory.time_diffs[end_idx]
                
                # Choose color based on anomaly type
                if anomaly.anomaly_type == AnomalyType.POSITION_JUMP:
                    color = 'orange'
                elif anomaly.anomaly_type == AnomalyType.UNREALISTIC_VELOCITY:
                    color = 'magenta'
                elif anomaly.anomaly_type == AnomalyType.UNREALISTIC_ACCELERATION:
                    color = 'cyan'
                elif anomaly.anomaly_type == AnomalyType.INCONSISTENT_MOVEMENT:
                    color = 'yellow'
                else:
                    color = 'gray'
                
                # Mark anomaly on position plot
                ax_pos.axvspan(t_start, t_end, alpha=0.3, color=color)
                
                # Mark anomaly on velocity plot if available
                if has_velocity:
                    ax_vel.axvspan(t_start, t_end, alpha=0.3, color=color)
                
                # Mark anomaly on acceleration plot if available
                if has_acceleration:
                    ax_acc.axvspan(t_start, t_end, alpha=0.3, color=color)
        
        # Add a legend for anomaly types
        legend_elements = [
            plt.Line2D([0], [0], color='orange', lw=4, label='Position Jump'),
            plt.Line2D([0], [0], color='magenta', lw=4, label='Unrealistic Velocity'),
            plt.Line2D([0], [0], color='cyan', lw=4, label='Unrealistic Acceleration'),
            plt.Line2D([0], [0], color='yellow', lw=4, label='Inconsistent Movement'),
            plt.Line2D([0], [0], color='gray', lw=4, label='Other Anomaly')
        ]
        
        # Add legend to top plot
        ax_pos.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        return fig


class IntegratedTrajectoryDetector:
    """Base class for integrated trajectory anomaly detection.
    
    This detector uses position, velocity, and acceleration data together
    for more comprehensive and accurate anomaly detection.
    """
    
    def __init__(
        self, 
        max_position_jump_m: float = 5.0,
        max_velocity_mps: float = 40.0,
        max_acceleration_mps2: float = 10.0,
        min_gps_jitter_m: float = 0.1,
        max_vibration_hz: float = 5.0,
        earth_radius_m: float = 6371000.0
    ) -> None:
        """Initialize the anomaly detector with thresholds.
        
        :param max_position_jump_m: Maximum allowed position jump in meters
        :param max_velocity_mps: Maximum allowed velocity in meters per second
        :param max_acceleration_mps2: Maximum allowed acceleration in meters per second squared
        :param min_gps_jitter_m: Minimum distance to consider as GPS jitter in meters
        :param max_vibration_hz: Maximum frequency of vibration to detect in Hz
        :param earth_radius_m: Earth radius in meters for distance calculations
        """
        self.max_position_jump_m = max_position_jump_m
        self.max_velocity_mps = max_velocity_mps
        self.max_acceleration_mps2 = max_acceleration_mps2
        self.min_gps_jitter_m = min_gps_jitter_m
        self.max_vibration_hz = max_vibration_hz
        self.earth_radius_m = earth_radius_m
    
    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate the great circle distance between two points in meters.
        
        :param lat1: Latitude of point 1 in degrees
        :param lon1: Longitude of point 1 in degrees
        :param lat2: Latitude of point 2 in degrees
        :param lon2: Longitude of point 2 in degrees
        :return: Distance between the points in meters
        """
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return self.earth_radius_m * c
    
    def calculate_movement_metrics(self, trajectory: TrajectoryData) -> pd.DataFrame:
        """Calculate derived metrics from position, velocity, and acceleration data.
        
        :param trajectory: Trajectory data to analyze
        :return: DataFrame with calculated metrics
        """
        # Create base DataFrame with position data
        df = pd.DataFrame({
            'time': trajectory.time,
            'time_diff': trajectory.time_diffs,
            'latitude': trajectory.latitude,
            'longitude': trajectory.longitude
        })
        
        # Calculate distances between consecutive points
        distances = []
        for i in range(len(df) - 1):
            dist = self.haversine_distance(
                df.iloc[i]['latitude'], df.iloc[i]['longitude'],
                df.iloc[i+1]['latitude'], df.iloc[i+1]['longitude']
            )
            distances.append(dist)
        distances.append(0.0)  # Add zero for the last point
        df['distance_m'] = distances
        
        # Calculate time differences between consecutive points
        time_diffs = []
        for i in range(len(df) - 1):
            td = df.iloc[i+1]['time'] - df.iloc[i]['time']
            time_diffs.append(max(td, 0.001))  # Avoid division by zero
        time_diffs.append(0.001)  # Add a small value for the last point
        df['time_diff_s'] = time_diffs
        
        # Calculate derived velocities (m/s) and accelerations (m/s²)
        df['derived_velocity_mps'] = df['distance_m'] / df['time_diff_s']
        
        # Calculate derived accelerations
        accelerations = []
        for i in range(len(df) - 1):
            if i == 0:
                accelerations.append(0.0)
            else:
                v_prev = df.iloc[i-1]['derived_velocity_mps']
                v_curr = df.iloc[i]['derived_velocity_mps']
                t_diff = df.iloc[i]['time_diff_s']
                accelerations.append((v_curr - v_prev) / t_diff)
        accelerations.append(0.0)  # Add zero for the last point
        df['derived_acceleration_mps2'] = accelerations
        
        # Add measured velocity data if available
        if len(trajectory.speed_3d) > 0:
            # Create a temporary DataFrame with velocity data
            vel_df = pd.DataFrame({
                'time': trajectory.velocity_time,
                'speed': trajectory.speed_3d,
                'vel_north': trajectory.vel_north,
                'vel_east': trajectory.vel_east,
                'vel_down': trajectory.vel_down
            })
            
            # Merge with main DataFrame on nearest time
            df = pd.merge_asof(df.sort_values('time'), 
                              vel_df.sort_values('time'), 
                              on='time', 
                              direction='nearest',
                              suffixes=('', '_vel'))
        
        # Add measured acceleration data if available
        if len(trajectory.accel_x) > 0:
            # Calculate acceleration magnitude
            accel_mag = np.sqrt(
                trajectory.accel_x**2 + 
                trajectory.accel_y**2 + 
                trajectory.accel_z**2
            )
            
            # Create a temporary DataFrame with acceleration data
            accel_df = pd.DataFrame({
                'time': trajectory.acceleration_time,
                'accel_x': trajectory.accel_x,
                'accel_y': trajectory.accel_y,
                'accel_z': trajectory.accel_z,
                'accel_magnitude': accel_mag
            })
            
            # Merge with main DataFrame on nearest time
            df = pd.merge_asof(df.sort_values('time'), 
                              accel_df.sort_values('time'), 
                              on='time', 
                              direction='nearest',
                              suffixes=('', '_accel'))
        
        # Calculate velocity-acceleration consistency if both are available
        if ('speed' in df.columns and 'accel_magnitude' in df.columns):
            # Calculate expected acceleration from velocity changes
            expected_accel = []
            for i in range(len(df) - 1):
                if i == 0:
                    expected_accel.append(0.0)
                else:
                    v_diff = df.iloc[i]['speed'] - df.iloc[i-1]['speed']
                    t_diff = max(df.iloc[i]['time'] - df.iloc[i-1]['time'], 0.001)
                    expected_accel.append(v_diff / t_diff)
            expected_accel.append(0.0)
            
            df['expected_accel'] = expected_accel
            df['accel_consistency'] = np.abs(df['accel_magnitude'] - df['expected_accel'])
        
        # Calculate position-velocity consistency if both are available
        if ('speed' in df.columns):
            # Calculate expected velocity from position changes
            df['velocity_consistency'] = np.abs(df['speed'] - df['derived_velocity_mps'])
        
        return df
    
    def detect_anomalies(self, trajectory: TrajectoryData) -> TrajectoryQualityReport:
        """Detect anomalies across position, velocity, and acceleration data.
        
        :param trajectory: Trajectory data to analyze
        :return: Quality report with detected anomalies
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def _detect_inconsistent_movement(self, metrics_df: pd.DataFrame, report: TrajectoryQualityReport) -> None:
        """Detect inconsistencies between position, velocity, and acceleration data.
        
        :param metrics_df: DataFrame with movement metrics
        :param report: Quality report to update
        """
        # Check for velocity consistency (measured vs. derived)
        if 'velocity_consistency' in metrics_df.columns:
            # Calculate threshold based on maximum derived velocity
            threshold = max(2.0, 0.2 * metrics_df['derived_velocity_mps'].max())
            
            # Find indices where inconsistency exceeds threshold
            inconsistent_indices = np.where(metrics_df['velocity_consistency'] > threshold)[0]
            
            # Group consecutive indices
            if len(inconsistent_indices) > 0:
                groups = np.split(inconsistent_indices, np.where(np.diff(inconsistent_indices) != 1)[0] + 1)
                
                for group in groups:
                    if len(group) > 3:  # Require multiple consecutive points for significance
                        start_idx = group[0]
                        end_idx = group[-1]
                        
                        # Calculate severity based on level of inconsistency
                        max_inconsistency = metrics_df.iloc[start_idx:end_idx+1]['velocity_consistency'].max()
                        avg_inconsistency = metrics_df.iloc[start_idx:end_idx+1]['velocity_consistency'].mean()
                        
                        severity = min(1.0, max_inconsistency / (threshold * 3))
                        
                        report.add_anomaly(AnomalyInfo(
                            anomaly_type=AnomalyType.INCONSISTENT_MOVEMENT,
                            start_index=start_idx,
                            end_index=end_idx,
                            severity=severity,
                            description=(
                                f"Inconsistency between measured and derived velocities. "
                                f"Average difference of {avg_inconsistency:.2f} m/s and maximum "
                                f"difference of {max_inconsistency:.2f} m/s detected."
                            )
                        ))
        
        # Check for acceleration consistency (measured vs. expected from velocity)
        if 'accel_consistency' in metrics_df.columns:
            # Calculate threshold based on maximum derived acceleration
            threshold = max(1.0, 0.2 * metrics_df['accel_magnitude'].max())
            
            # Find indices where inconsistency exceeds threshold
            inconsistent_indices = np.where(metrics_df['accel_consistency'] > threshold)[0]
            
            # Group consecutive indices
            if len(inconsistent_indices) > 0:
                groups = np.split(inconsistent_indices, np.where(np.diff(inconsistent_indices) != 1)[0] + 1)
                
                for group in groups:
                    if len(group) > 3:  # Require multiple consecutive points for significance
                        start_idx = group[0]
                        end_idx = group[-1]
                        
                        # Calculate severity based on level of inconsistency
                        max_inconsistency = metrics_df.iloc[start_idx:end_idx+1]['accel_consistency'].max()
                        avg_inconsistency = metrics_df.iloc[start_idx:end_idx+1]['accel_consistency'].mean()
                        
                        severity = min(1.0, max_inconsistency / (threshold * 3))
                        
                        report.add_anomaly(AnomalyInfo(
                            anomaly_type=AnomalyType.INCONSISTENT_MOVEMENT,
                            start_index=start_idx,
                            end_index=end_idx,
                            severity=severity,
                            description=(
                                f"Inconsistency between measured and expected accelerations. "
                                f"Average difference of {avg_inconsistency:.2f} m/s² and maximum "
                                f"difference of {max_inconsistency:.2f} m/s² detected."
                            )
                        ))
    
    def _detect_vibration(self, trajectory: TrajectoryData, report: TrajectoryQualityReport) -> None:
        """Detect high-frequency vibrations in acceleration data.
        
        :param trajectory: Trajectory data with acceleration measurements
        :param report: Quality report to update
        """
        # Check if we have acceleration data
        if len(trajectory.accel_x) < 20:  # Need enough points for frequency analysis
            return
            
        from scipy import signal
        
        # Perform frequency analysis on acceleration components
        # Use Welch's method to compute power spectral density
        for axis_name, axis_data in [
            ('X', trajectory.accel_x),
            ('Y', trajectory.accel_y),
            ('Z', trajectory.accel_z)
        ]:
            # Calculate sampling frequency
            if len(trajectory.acceleration_time) > 1:
                # Average time difference between samples
                sampling_freq = 1.0 / np.mean(np.diff(trajectory.acceleration_time))
            else:
                continue  # Skip if can't determine sampling frequency
                
            # Calculate power spectral density
            f, psd = signal.welch(axis_data, fs=sampling_freq, nperseg=min(256, len(axis_data)//4))
            
            # Find the dominant frequency
            dominant_idx = np.argmax(psd)
            dominant_freq = f[dominant_idx]
            dominant_power = psd[dominant_idx]
            
            # Check if the dominant frequency is in the vibration range and has significant power
            if 0.5 < dominant_freq < self.max_vibration_hz:
                # Calculate a baseline power (median)
                baseline_power = np.median(psd)
                
                # If dominant power is significantly higher than baseline, report vibration
                if dominant_power > baseline_power * 5:
                    # Find time segments where this vibration is most prominent
                    # Use a bandpass filter around the dominant frequency
                    sos = signal.butter(4, [dominant_freq * 0.8, dominant_freq * 1.2], 
                                        'bandpass', fs=sampling_freq, output='sos')
                    filtered = signal.sosfilt(sos, axis_data)
                    
                    # Calculate filtered signal energy
                    energy = filtered**2
                    
                    # Find segments with high energy
                    threshold = np.mean(energy) + 2 * np.std(energy)
                    high_energy_indices = np.where(energy > threshold)[0]
                    
                    if len(high_energy_indices) > 0:
                        # Group consecutive indices
                        groups = np.split(high_energy_indices, np.where(np.diff(high_energy_indices) != 1)[0] + 1)
                        
                        for group in groups:
                            if len(group) > 5:  # Require multiple consecutive points
                                start_idx = group[0]
                                end_idx = group[-1]
                                
                                # Calculate severity based on amplitude and duration
                                amplitude = np.max(np.abs(filtered[start_idx:end_idx+1]))
                                duration = trajectory.acceleration_time[end_idx] - trajectory.acceleration_time[start_idx]
                                
                                severity = min(1.0, (amplitude / np.std(axis_data)) * (duration / 10))
                                
                                report.add_anomaly(AnomalyInfo(
                                    anomaly_type=AnomalyType.VIBRATION,
                                    start_index=start_idx,
                                    end_index=end_idx,
                                    severity=severity,
                                    description=(
                                        f"Vibration detected in {axis_name}-axis acceleration at frequency "
                                        f"{dominant_freq:.2f} Hz with amplitude {amplitude:.2f} times normal. "
                                        f"Duration: {duration:.2f} seconds."
                                    ),
                                    additional_data={
                                        "frequency": dominant_freq,
                                        "amplitude": amplitude,
                                        "duration": duration,
                                        "axis": axis_name
                                    }
                                ))


class IntegratedAnomalyDetector(IntegratedTrajectoryDetector):
    """Detect anomalies using integrated position, velocity, and acceleration data."""
    
    def detect_anomalies(self, trajectory: TrajectoryData) -> TrajectoryQualityReport:
        """Detect anomalies across all available data sources.
        
        :param trajectory: Trajectory data to analyze
        :return: Quality report with detected anomalies
        """
        # Get or create trajectory name
        trajectory_name = getattr(trajectory, 'name', 'Unnamed Trajectory')
        
        # Create quality report
        report = TrajectoryQualityReport(trajectory_name)
        
        # Calculate metrics
        metrics_df = self.calculate_movement_metrics(trajectory)
        
        # Check what data we have
        has_velocity = 'speed' in metrics_df.columns
        has_acceleration = 'accel_magnitude' in metrics_df.columns
        
        # Add summary statistics to report
        report.add_summary_statistic("Total Points", len(metrics_df))
        report.add_summary_statistic("Total Distance (m)", metrics_df['distance_m'].sum())
        
        # Position-derived velocity statistics
        report.add_summary_statistic("Max Velocity (derived, m/s)", metrics_df['derived_velocity_mps'].max())
        report.add_summary_statistic("Avg Velocity (derived, m/s)", metrics_df['derived_velocity_mps'].mean())
        
        # Direct velocity measurements if available
        if has_velocity:
            report.add_summary_statistic("Max Velocity (measured, m/s)", metrics_df['speed'].max())
            report.add_summary_statistic("Avg Velocity (measured, m/s)", metrics_df['speed'].mean())
            
            if 'velocity_consistency' in metrics_df.columns:
                report.add_summary_statistic("Velocity Measurement Consistency", 
                                           metrics_df['velocity_consistency'].mean())
        
        # Direct acceleration measurements if available
        if has_acceleration:
            report.add_summary_statistic("Max Acceleration (measured, m/s²)", metrics_df['accel_magnitude'].max())
            report.add_summary_statistic("Avg Acceleration (measured, m/s²)", metrics_df['accel_magnitude'].mean())
            
            if 'accel_consistency' in metrics_df.columns:
                report.add_summary_statistic("Acceleration Measurement Consistency", 
                                           metrics_df['accel_consistency'].mean())
        
        report.add_summary_statistic("Duration (s)", metrics_df['time'].iloc[-1] - metrics_df['time'].iloc[0])
        
        # Anomaly detection - run detection methods based on available data
        
        # 1. Position-based anomalies (always available)
        self._detect_position_jumps(metrics_df, report)
        self._detect_zigzag_patterns(metrics_df, trajectory, report)
        self._detect_gps_jitter(metrics_df, trajectory, report)
        
        # 2. Use measured velocities if available, otherwise derived
        if has_velocity:
            self._detect_unrealistic_measured_velocities(metrics_df, report)
        else:
            self._detect_unrealistic_derived_velocities(metrics_df, report)
        
        # 3. Use measured accelerations if available, otherwise derived
        if has_acceleration:
            self._detect_unrealistic_measured_accelerations(metrics_df, report)
        else:
            self._detect_unrealistic_derived_accelerations(metrics_df, report)
        
        # 4. Detect inconsistencies between data sources
        if has_velocity or has_acceleration:
            self._detect_inconsistent_movement(metrics_df, report)
            
        # 5. Detect vibrations in acceleration data
        if has_acceleration:
            self._detect_vibration(trajectory, report)
            
        return report
    
    def _detect_position_jumps(self, metrics_df: pd.DataFrame, report: TrajectoryQualityReport) -> None:
        """Detect sudden jumps in position.
        
        :param metrics_df: DataFrame with movement metrics
        :param report: Quality report to update
        """
        # Calculate z-scores for distances
        z_scores = np.abs(stats.zscore(metrics_df['distance_m']))
        
        # Find indices where z-score is high and distance exceeds threshold
        jump_indices = np.where((z_scores > 3.0) & (metrics_df['distance_m'] > self.max_position_jump_m))[0]
        
        # Group consecutive indices
        if len(jump_indices) > 0:
            groups = np.split(jump_indices, np.where(np.diff(jump_indices) != 1)[0] + 1)
            
            for group in groups:
                if len(group) > 0:
                    start_idx = group[0]
                    end_idx = group[-1]
                    
                    distance = metrics_df.iloc[start_idx]['distance_m']
                    time_diff = metrics_df.iloc[start_idx]['time_diff_s']
                    
                    # Calculate severity based on how much it exceeds the threshold
                    severity = min(1.0, distance / (self.max_position_jump_m * 5))
                    
                    report.add_anomaly(AnomalyInfo(
                        anomaly_type=AnomalyType.POSITION_JUMP,
                        start_index=start_idx,
                        end_index=end_idx,
                        severity=severity,
                        description=(
                            f"Sudden position jump of {distance:.2f} meters over {time_diff:.2f} seconds. "
                            f"This exceeds the maximum expected jump of {self.max_position_jump_m} meters."
                        )
                    ))
    
    def _detect_unrealistic_derived_velocities(self, metrics_df: pd.DataFrame, report: TrajectoryQualityReport) -> None:
        """Detect unrealistically high velocities using calculated velocity data.
        
        :param metrics_df: DataFrame with movement metrics
        :param report: Quality report to update
        """
        # Find indices where derived velocity exceeds threshold
        velocity_indices = np.where(metrics_df['derived_velocity_mps'] > self.max_velocity_mps)[0]
        
        # Group consecutive indices
        if len(velocity_indices) > 0:
            groups = np.split(velocity_indices, np.where(np.diff(velocity_indices) != 1)[0] + 1)
            
            for group in groups:
                if len(group) > 0:
                    start_idx = group[0]
                    end_idx = group[-1]
                    
                    max_velocity = metrics_df.iloc[start_idx:end_idx+1]['derived_velocity_mps'].max()
                    
                    # Calculate severity based on how much it exceeds the threshold
                    severity = min(1.0, max_velocity / (self.max_velocity_mps * 3))
                    
                    report.add_anomaly(AnomalyInfo(
                        anomaly_type=AnomalyType.UNREALISTIC_VELOCITY,
                        start_index=start_idx,
                        end_index=end_idx,
                        severity=severity,
                        description=(
                            f"Unrealistic velocity of {max_velocity:.2f} m/s detected (calculated from position). "
                            f"This exceeds the maximum expected velocity of {self.max_velocity_mps} m/s."
                        )
                    ))
    
    def _detect_unrealistic_measured_velocities(self, metrics_df: pd.DataFrame, report: TrajectoryQualityReport) -> None:
        """Detect unrealistically high velocities using measured velocity data.
        
        :param metrics_df: DataFrame with movement metrics
        :param report: Quality report to update
        """
        # Find indices where measured velocity exceeds threshold
        velocity_indices = np.where(metrics_df['speed'] > self.max_velocity_mps)[0]
        
        # Group consecutive indices
        if len(velocity_indices) > 0:
            groups = np.split(velocity_indices, np.where(np.diff(velocity_indices) != 1)[0] + 1)
            
            for group in groups:
                if len(group) > 0:
                    start_idx = group[0]
                    end_idx = group[-1]
                    
                    max_velocity = metrics_df.iloc[start_idx:end_idx+1]['speed'].max()
                    
                    # Calculate severity based on how much it exceeds the threshold
                    severity = min(1.0, max_velocity / (self.max_velocity_mps * 3))
                    
                    report.add_anomaly(AnomalyInfo(
                        anomaly_type=AnomalyType.UNREALISTIC_VELOCITY,
                        start_index=start_idx,
                        end_index=end_idx,
                        severity=severity,
                        description=(
                            f"Unrealistic velocity of {max_velocity:.2f} m/s detected (direct measurement). "
                            f"This exceeds the maximum expected velocity of {self.max_velocity_mps} m/s."
                        )
                    ))
    
    def _detect_unrealistic_derived_accelerations(self, metrics_df: pd.DataFrame, report: TrajectoryQualityReport) -> None:
        """Detect unrealistically high accelerations using derived acceleration data.
        
        :param metrics_df: DataFrame with movement metrics
        :param report: Quality report to update
        """
        # Find indices where acceleration exceeds threshold
        accel_indices = np.where(np.abs(metrics_df['derived_acceleration_mps2']) > self.max_acceleration_mps2)[0]
        
        # Group consecutive indices
        if len(accel_indices) > 0:
            groups = np.split(accel_indices, np.where(np.diff(accel_indices) != 1)[0] + 1)
            
            for group in groups:
                if len(group) > 0:
                    start_idx = group[0]
                    end_idx = group[-1]
                    
                    max_accel = np.abs(metrics_df.iloc[start_idx:end_idx+1]['derived_acceleration_mps2']).max()
                    
                    # Calculate severity based on how much it exceeds the threshold
                    severity = min(1.0, max_accel / (self.max_acceleration_mps2 * 3))
                    
                    report.add_anomaly(AnomalyInfo(
                        anomaly_type=AnomalyType.UNREALISTIC_ACCELERATION,
                        start_index=start_idx,
                        end_index=end_idx,
                        severity=severity,
                        description=(
                            f"Unrealistic acceleration of {max_accel:.2f} m/s² detected (derived from velocity). "
                            f"This exceeds the maximum expected acceleration of {self.max_acceleration_mps2} m/s²."
                        )
                    ))
    
    def _detect_unrealistic_measured_accelerations(self, metrics_df: pd.DataFrame, report: TrajectoryQualityReport) -> None:
        """Detect unrealistically high accelerations using measured acceleration data.
        
        :param metrics_df: DataFrame with movement metrics
        :param report: Quality report to update
        """
        # Find indices where acceleration exceeds threshold
        accel_indices = np.where(metrics_df['accel_magnitude'] > self.max_acceleration_mps2)[0]
        
        # Group consecutive indices
        if len(accel_indices) > 0:
            groups = np.split(accel_indices, np.where(np.diff(accel_indices) != 1)[0] + 1)
            
            for group in groups:
                if len(group) > 0:
                    start_idx = group[0]
                    end_idx = group[-1]
                    
                    max_accel = metrics_df.iloc[start_idx:end_idx+1]['accel_magnitude'].max()
                    
                    # Get component accelerations if available for more detail
                    components = ""
                    if 'accel_x' in metrics_df.columns:
                        max_x = metrics_df.iloc[start_idx:end_idx+1]['accel_x'].max()
                        max_y = metrics_df.iloc[start_idx:end_idx+1]['accel_y'].max()
                        max_z = metrics_df.iloc[start_idx:end_idx+1]['accel_z'].max()
                        components = f" Components - X: {max_x:.2f}, Y: {max_y:.2f}, Z: {max_z:.2f} m/s²."
                    
                    # Calculate severity based on how much it exceeds the threshold
                    severity = min(1.0, max_accel / (self.max_acceleration_mps2 * 3))
                    
                    report.add_anomaly(AnomalyInfo(
                        anomaly_type=AnomalyType.UNREALISTIC_ACCELERATION,
                        start_index=start_idx,
                        end_index=end_idx,
                        severity=severity,
                        description=(
                            f"Unrealistic acceleration of {max_accel:.2f} m/s² detected (direct measurement)."
                            f"{components} This exceeds the maximum expected acceleration of {self.max_acceleration_mps2} m/s²."
                        )
                    ))
    
    def _detect_gps_jitter(self, metrics_df: pd.DataFrame, trajectory: TrajectoryData, report: TrajectoryQualityReport) -> None:
        """Detect GPS jitter (small movements when should be stationary).
        
        :param metrics_df: DataFrame with movement metrics
        :param trajectory: Original trajectory data
        :param report: Quality report to update
        """
        # Check if we have velocity data (either measured or derived)
        velocity_column = 'speed' if 'speed' in metrics_df.columns else 'derived_velocity_mps'
        
        # Calculate moving average of velocity
        window_size = min(5, len(metrics_df) - 1)
        if window_size <= 0:
            return
            
        metrics_df['velocity_ma'] = metrics_df[velocity_column].rolling(window=window_size, center=True).mean()
        metrics_df['velocity_ma'] = metrics_df['velocity_ma'].fillna(metrics_df[velocity_column])
        
        # Look for points where velocity is very low but position changes more than minimal jitter
        jitter_indices = np.where(
            (metrics_df['velocity_ma'] < 0.5) &  # Very low average velocity
            (metrics_df['distance_m'] > self.min_gps_jitter_m) &  # But position changes more than min jitter
            (metrics_df['distance_m'] < self.max_position_jump_m)  # But not a full position jump
        )[0]
        
        # Group consecutive indices
        if len(jitter_indices) > 0:
            groups = np.split(jitter_indices, np.where(np.diff(jitter_indices) != 1)[0] + 1)
            
            for group in groups:
                if len(group) > 3:  # Only report if jitter persists for multiple points
                    start_idx = group[0]
                    end_idx = group[-1]
                    
                    total_distance = metrics_df.iloc[start_idx:end_idx+1]['distance_m'].sum()
                    avg_velocity = metrics_df.iloc[start_idx:end_idx+1]['velocity_ma'].mean()
                    
                    # Additional check using acceleration data if available
                    is_confirmed_jitter = True
                    accel_details = ""
                    
                    if 'accel_magnitude' in metrics_df.columns:
                        # Look at acceleration during the potential jitter
                        accel_segment = metrics_df.iloc[start_idx:end_idx+1]['accel_magnitude']
                        
                        # High-frequency, low-magnitude accelerations are typical of GPS jitter
                        accel_mean = accel_segment.mean()
                        accel_std = accel_segment.std()
                        
                        # Check for high variability with low magnitude
                        if accel_std > 0.5 and accel_mean < 2.0:
                            accel_details = f" Acceleration data shows characteristic jitter pattern (mean: {accel_mean:.2f}, std: {accel_std:.2f})."
                        else:
                            # If acceleration doesn't fit jitter pattern, it might be real movement
                            is_confirmed_jitter = False
                    
                    if is_confirmed_jitter:
                        # Calculate severity based on duration and distance
                        duration = end_idx - start_idx + 1
                        severity = min(1.0, (duration / 10) * (total_distance / (self.min_gps_jitter_m * 10)))
                        
                        report.add_anomaly(AnomalyInfo(
                            anomaly_type=AnomalyType.GPS_JITTER,
                            start_index=start_idx,
                            end_index=end_idx,
                            severity=severity,
                            description=(
                                f"GPS jitter detected. Position changing by {total_distance:.2f} meters "
                                f"over {duration} points, with an average velocity of only {avg_velocity:.2f} m/s."
                                f"{accel_details}"
                            )
                        ))
    
    def _detect_zigzag_patterns(self, metrics_df: pd.DataFrame, trajectory: TrajectoryData, report: TrajectoryQualityReport) -> None:
        """Detect zigzag patterns (rapid direction changes).
        
        :param metrics_df: DataFrame with movement metrics
        :param trajectory: Original trajectory data
        :param report: Quality report to update
        """
        if len(trajectory.latitude) < 5:
            return
            
        # Calculate bearing changes
        bearings = []
        bearing_changes = []
        
        for i in range(len(trajectory.latitude) - 1):
            lat1, lon1 = trajectory.latitude[i], trajectory.longitude[i]
            lat2, lon2 = trajectory.latitude[i+1], trajectory.longitude[i+1]
            
            # Calculate initial bearing
            y = math.sin(math.radians(lon2) - math.radians(lon1)) * math.cos(math.radians(lat2))
            x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - \
                math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
                math.cos(math.radians(lon2) - math.radians(lon1))
            bearing = (math.degrees(math.atan2(y, x)) + 360) % 360
            bearings.append(bearing)
            
            if i > 0:
                # Calculate bearing change
                change = min((bearings[i] - bearings[i-1]) % 360, (bearings[i-1] - bearings[i]) % 360)
                bearing_changes.append(change)
            else:
                bearing_changes.append(0)
        
        # Add a final value
        bearings.append(bearings[-1] if bearings else 0)
        bearing_changes.append(0)
        
        # Add to dataframe
        metrics_df['bearing'] = bearings
        metrics_df['bearing_change'] = bearing_changes
        
        # Refine detection using acceleration data if available
        acceleration_check = False
        if 'accel_magnitude' in metrics_df.columns:
            acceleration_check = True
        
        # Identify zigzag patterns (large bearing changes in short distances)
        zigzag_indices = np.where(
            (metrics_df['bearing_change'] > 45) &  # Significant direction change
            (metrics_df['distance_m'] < self.max_position_jump_m * 0.5) &  # Short distance
            (metrics_df['derived_velocity_mps'] > 1.0)  # Not stationary
        )[0]
        
        # Group consecutive or near-consecutive indices
        if len(zigzag_indices) > 0:
            groups = []
            current_group = [zigzag_indices[0]]
            
            for i in range(1, len(zigzag_indices)):
                if zigzag_indices[i] - zigzag_indices[i-1] < 5:  # Allow small gaps
                    current_group.append(zigzag_indices[i])
                else:
                    if len(current_group) > 1:  # Only keep groups with multiple points
                        groups.append(current_group)
                    current_group = [zigzag_indices[i]]
            
            if len(current_group) > 1:
                groups.append(current_group)
            
            for group in groups:
                if len(group) > 1:
                    start_idx = group[0]
                    end_idx = group[-1]
                    
                    max_bearing_change = metrics_df.iloc[start_idx:end_idx+1]['bearing_change'].max()
                    avg_distance = metrics_df.iloc[start_idx:end_idx+1]['distance_m'].mean()
                    
                    # Check acceleration if available
                    accel_evidence = ""
                    is_real_zigzag = True
                    
                    if acceleration_check:
                        # Look at lateral acceleration patterns
                        accel_segment = metrics_df.iloc[start_idx:end_idx+1]
                        
                        if 'accel_x' in accel_segment.columns and 'accel_y' in accel_segment.columns:
                            # Calculate correlation between bearing changes and lateral acceleration
                            # (This is a simplification - ideally we'd need vehicle heading)
                            lateral_accel = np.sqrt(accel_segment['accel_x']**2 + accel_segment['accel_y']**2)
                            
                            # If lateral acceleration doesn't match zigzag pattern, it might be GPS error
                            if lateral_accel.max() < 2.0:
                                accel_evidence = " Acceleration data suggests this may be GPS error rather than real movement."
                                is_real_zigzag = False
                            else:
                                accel_evidence = f" Confirmed by lateral acceleration patterns (max: {lateral_accel.max():.2f} m/s²)."
                    
                    if is_real_zigzag:
                        # Calculate severity based on bearing changes and frequency
                        severity = min(1.0, (max_bearing_change / 90) * (len(group) / 10))
                        
                        report.add_anomaly(AnomalyInfo(
                            anomaly_type=AnomalyType.ZIGZAG_PATTERN,
                            start_index=start_idx,
                            end_index=end_idx,
                            severity=severity,
                            description=(
                                f"Zigzag pattern detected with {len(group)} direction changes. "
                                f"Maximum bearing change of {max_bearing_change:.1f}° over an average distance "
                                f"of only {avg_distance:.2f} meters.{accel_evidence}"
                            )
                        ))


class TrajectoryQualityAnalyzer:
    """Class to analyze quality across multiple trajectories."""
    
    def __init__(
        self, 
        detector_type: str = "integrated",
        max_position_jump_m: float = 5.0,
        max_velocity_mps: float = 40.0,
        max_acceleration_mps2: float = 10.0,
        min_gps_jitter_m: float = 0.1,
        max_vibration_hz: float = 5.0
    ) -> None:
        """Initialize the quality analyzer.
        
        :param detector_type: Type of detector to use ("integrated" for combined data)
        :param max_position_jump_m: Maximum allowed position jump in meters
        :param max_velocity_mps: Maximum allowed velocity in meters per second
        :param max_acceleration_mps2: Maximum allowed acceleration in meters per second squared
        :param min_gps_jitter_m: Minimum distance to consider as GPS jitter in meters
        :param max_vibration_hz: Maximum frequency of vibration to detect in Hz
        """
        self.detector_params = {
            "max_position_jump_m": max_position_jump_m,
            "max_velocity_mps": max_velocity_mps,
            "max_acceleration_mps2": max_acceleration_mps2,
            "min_gps_jitter_m": min_gps_jitter_m,
            "max_vibration_hz": max_vibration_hz
        }
        
        self.detector = IntegratedAnomalyDetector(**self.detector_params)
        self.reports: Dict[str, TrajectoryQualityReport] = {}
    
    def analyze_trajectory(self, trajectory: TrajectoryData) -> TrajectoryQualityReport:
        """Analyze the quality of a single trajectory.
        
        :param trajectory: Trajectory data to analyze
        :return: Quality report for the trajectory
        """
        # Get or create trajectory name
        trajectory_name = getattr(trajectory, 'name', 'Unnamed Trajectory')
        
        # Run anomaly detection
        report = self.detector.detect_anomalies(trajectory)
        
        # Store report
        self.reports[trajectory_name] = report
        
        return report
    
    def analyze_collection(self, collection: TrajectoryCollection) -> Dict[str, TrajectoryQualityReport]:
        """Analyze the quality of all trajectories in a collection.
        
        :param collection: Collection of trajectories to analyze
        :return: Dictionary mapping trajectory names to quality reports
        """
        for name in collection.get_trajectory_names():
            trajectory = collection.get_trajectory(name)
            trajectory.name = name  # Ensure trajectory has a name attribute
            self.analyze_trajectory(trajectory)
        
        return self.reports
    
    def generate_summary_report(self) -> str:
        """Generate a summary report of all analyzed trajectories.
        
        :return: Formatted text summary report
        """
        if not self.reports:
            return "No trajectories have been analyzed."
        
        report_lines = [
            "Trajectory Quality Analysis Summary",
            f"Number of trajectories analyzed: {len(self.reports)}",
            ""
        ]
        
        # Count trajectories by quality
        good_quality = sum(1 for report in self.reports.values() if report.is_good_quality())
        poor_quality = len(self.reports) - good_quality
        
        report_lines.append(f"Good quality trajectories: {good_quality}")
        report_lines.append(f"Poor quality trajectories: {poor_quality}")
        report_lines.append("")
        
        # List trajectories by quality score
        report_lines.append("Trajectories sorted by quality score:")
        sorted_reports = sorted(
            self.reports.items(), 
            key=lambda x: x[1].quality_score,
            reverse=True
        )
        
        for name, report in sorted_reports:
            status = "GOOD" if report.is_good_quality() else "NEEDS REVIEW"
            anomaly_count = len(report.anomalies)
            report_lines.append(
                f"  {name}: Score {report.quality_score:.2f} - {status} - {anomaly_count} anomalies"
            )
        
        # Summarize anomaly types
        anomaly_types = {}
        for report in self.reports.values():
            for anomaly in report.anomalies:
                anomaly_name = anomaly.anomaly_type.name
                if anomaly_name in anomaly_types:
                    anomaly_types[anomaly_name] += 1
                else:
                    anomaly_types[anomaly_name] = 1
        
        if anomaly_types:
            report_lines.append("\nDetected Anomaly Types:")
            for anomaly_name, count in sorted(anomaly_types.items(), key=lambda x: x[1], reverse=True):
                report_lines.append(f"  {anomaly_name}: {count} occurrences")
        
        return "\n".join(report_lines)
    
    def plot_quality_comparison(self) -> plt.Figure:
        """Create a bar chart comparing quality scores of trajectories.
        
        :return: Matplotlib figure with quality comparison
        """
        if not self.reports:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No trajectories have been analyzed", 
                    ha='center', va='center')
            return fig
        
        # Extract names and scores
        names = list(self.reports.keys())
        scores = [report.quality_score for report in self.reports.values()]
        anomaly_counts = [len(report.anomalies) for report in self.reports.values()]
        
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]
        names = [names[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]
        anomaly_counts = [anomaly_counts[i] for i in sorted_indices]
        
        # Create figure
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot quality scores
        bars = ax1.bar(names, scores, color='skyblue', alpha=0.7)
        ax1.set_ylim(0, 1.0)
        ax1.set_ylabel('Quality Score')
        ax1.set_xlabel('Trajectory')
        ax1.set_title('Trajectory Quality Comparison')
        
        # Color bars based on quality
        for i, bar in enumerate(bars):
            if scores[i] >= 0.8:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        # Add a second axis for anomaly counts
        ax2 = ax1.twinx()
        ax2.plot(names, anomaly_counts, 'ro-', label='Anomaly Count')
        ax2.set_ylabel('Number of Anomalies')
        
        # Add threshold line
        ax1.axhline(y=0.8, color='gray', linestyle='--', alpha=0.7, 
                    label='Quality Threshold')
        
        # Add a legend
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper right')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return fig

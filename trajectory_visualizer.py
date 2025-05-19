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
"""Classes for visualizing trajectory data in various formats."""

from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import folium
from folium.plugins import TimestampedGeoJson
from datetime import datetime, timedelta
import plotly.graph_objects as go
import colorsys

from trajectory_data import TrajectoryData
from trajectory_collection import TrajectoryCollection


class TrajectoryVisualizer:
    """Base class for trajectory visualization tools."""
    
    def __init__(self, trajectory_data: TrajectoryData) -> None:
        """Initialize visualizer with trajectory data.
        
        :param trajectory_data: Processed trajectory data object
        """
        self.data = trajectory_data
        
    def get_center_coordinates(self) -> Tuple[float, float]:
        """Get the center coordinates of the trajectory.
        
        :return: Tuple of (center_latitude, center_longitude)
        """
        center_lat = float(np.mean(self.data.latitude))
        center_lon = float(np.mean(self.data.longitude))
        return center_lat, center_lon


class MatplotlibVisualizer(TrajectoryVisualizer):
    """Class for creating static trajectory plots using Matplotlib."""
    
    def plot_trajectory(self, figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """Create a static plot of the trajectory path.
        
        :param figsize: Figure size as (width, height) in inches
        :return: Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot the full path
        ax.plot(
            self.data.longitude, 
            self.data.latitude, 
            'b-', 
            linewidth=2
        )
        
        # Plot start and end points
        ax.plot(
            self.data.longitude[0], 
            self.data.latitude[0], 
            'go', 
            markersize=10, 
            label="Start"
        )
        
        ax.plot(
            self.data.longitude[-1], 
            self.data.latitude[-1], 
            'ro', 
            markersize=10, 
            label="End"
        )
        
        # Add labels and style
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Target Trajectory")
        ax.grid(True)
        ax.legend()
        ax.axis("equal")  # Equal scaling for lat/lon
        
        plt.tight_layout()
        return fig


class FoliumMapVisualizer(TrajectoryVisualizer):
    """Class for creating interactive maps using Folium."""
    
    def create_map(self, zoom_start: int = 14) -> folium.Map:
        """Create a basic interactive map with the trajectory.
        
        :param zoom_start: Initial zoom level for the map
        :return: Folium map object
        """
        center_lat, center_lon = self.get_center_coordinates()
        
        # Create a map centered at the mean coordinates
        trajectory_map = folium.Map(
            location=[center_lat, center_lon], 
            zoom_start=zoom_start
        )
        
        # Add the trajectory as a line
        points = list(zip(self.data.latitude, self.data.longitude))
        folium.PolyLine(
            points, 
            color="blue", 
            weight=5, 
            opacity=0.7
        ).add_to(trajectory_map)
        
        # Add markers for start and end points
        folium.Marker(
            location=[self.data.latitude[0], self.data.longitude[0]],
            popup="Start",
            icon=folium.Icon(color="green", icon="play")
        ).add_to(trajectory_map)
        
        folium.Marker(
            location=[self.data.latitude[-1], self.data.longitude[-1]],
            popup="End",
            icon=folium.Icon(color="red", icon="stop")
        ).add_to(trajectory_map)
        
        return trajectory_map


class PlotlyAnimationVisualizer(TrajectoryVisualizer):
    """Class for creating animated visualizations using Plotly."""
    
    def create_animation(self, max_frames: int = 100) -> go.Figure:
        """Create an animated visualization of the trajectory.
        
        :param max_frames: Maximum number of frames to include in the animation
        :return: Plotly figure object with animation
        """
        # Get indices for frames
        frame_indices = self.data.get_points_for_visualization(max_frames)
        
        # Create frames for animation
        frames = []
        for idx in frame_indices:
            # For each frame, show all points up to the current one
            frames.append(
                go.Frame(
                    data=[
                        # The path so far
                        go.Scattermap(
                            lat=self.data.latitude[:idx+1],
                            lon=self.data.longitude[:idx+1],
                            mode="lines",
                            line=dict(width=2, color="blue"),
                            opacity=0.7,
                            name="Path",
                            showlegend=False
                        ),
                        # The current position
                        go.Scattermap(
                            lat=[self.data.latitude[idx]],
                            lon=[self.data.longitude[idx]],
                            mode="markers",
                            marker=dict(size=12, color="red"),
                            name="Current Position",
                            showlegend=False
                        )
                    ],
                    name=f"frame{idx}",
                    # Add timing information for smooth animation
                    layout=dict(
                        title=f"Time: +{self.data.time_diffs[idx]:.2f} seconds"
                    )
                )
            )
        
        center_lat, center_lon = self.get_center_coordinates()
        
        # Create the figure with initial data
        fig = go.Figure(
            data=[
                go.Scattermap(
                    lat=[self.data.latitude[0]],
                    lon=[self.data.longitude[0]],
                    mode="markers",
                    marker=dict(size=12, color="red"),
                    name="Starting Position"
                )
            ],
            frames=frames,
            layout=dict(
                title="Target Movement Animation",
                mapbox=dict(
                    style="open-street-map",
                    center=dict(lat=center_lat, lon=center_lon),
                    zoom=13
                ),
                updatemenus=[
                    dict(
                        type="buttons",
                        showactive=False,
                        buttons=[
                            dict(
                                label="Play",
                                method="animate",
                                args=[None, dict(
                                    frame=dict(duration=50, redraw=True),
                                    fromcurrent=True,
                                    mode="immediate"
                                )]
                            ),
                            dict(
                                label="Pause",
                                method="animate",
                                args=[[None], dict(
                                    frame=dict(duration=0, redraw=True),
                                    mode="immediate"
                                )]
                            )
                        ],
                        direction="left",
                        pad=dict(r=10, t=10),
                        x=0.1,
                        y=0,
                    )
                ],
                height=600,
                margin=dict(l=0, r=0, t=50, b=0)
            )
        )
        
        return fig


class MultiTrajectoryMplVisualizer:
    """Class for creating static plots of multiple trajectories using Matplotlib."""
    
    def __init__(self, trajectory_collection: TrajectoryCollection) -> None:
        """Initialize visualizer with a trajectory collection.
        
        :param trajectory_collection: Collection of trajectories to visualize
        """
        self.collection = trajectory_collection
    
    def plot_trajectories(self, figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """Create a static plot of multiple trajectory paths.
        
        :param figsize: Figure size as (width, height) in inches
        :return: Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        for name in self.collection.get_trajectory_names():
            trajectory = self.collection.get_trajectory(name)
            color = self.collection.get_color(name)
            
            # Plot the full path
            ax.plot(
                trajectory.longitude, 
                trajectory.latitude, 
                '-', 
                color=color,
                linewidth=2,
                label=name
            )
            
            # Plot start and end points
            ax.plot(
                trajectory.longitude[0], 
                trajectory.latitude[0], 
                'o', 
                color=color,
                markersize=8,
                markeredgecolor='black'
            )
            
            ax.plot(
                trajectory.longitude[-1], 
                trajectory.latitude[-1], 
                's', 
                color=color,
                markersize=8,
                markeredgecolor='black'
            )
        
        # Add labels and style
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Multiple Target Trajectories")
        ax.grid(True)
        ax.legend()
        ax.axis("equal")  # Equal scaling for lat/lon
        
        plt.tight_layout()
        return fig


class MultiFoliumMapVisualizer:
    """Class for creating interactive maps of multiple trajectories using Folium."""
    
    def __init__(self, trajectory_collection: TrajectoryCollection) -> None:
        """Initialize visualizer with a trajectory collection.
        
        :param trajectory_collection: Collection of trajectories to visualize
        """
        self.collection = trajectory_collection
    
    def create_map(self, zoom_start: int = 14) -> folium.Map:
        """Create an interactive map with multiple trajectories.
        
        :param zoom_start: Initial zoom level for the map
        :return: Folium map object
        """
        try:
            center_lat, center_lon = self.collection.get_center_coordinates()
        except ValueError:
            # Default to a reasonable location if collection is empty
            center_lat, center_lon = 0, 0
        
        # Create a map centered at the mean coordinates
        multi_map = folium.Map(
            location=[center_lat, center_lon], 
            zoom_start=zoom_start
        )
        
        # Add a layer control
        folium.LayerControl().add_to(multi_map)
        
        # Add each trajectory
        for name in self.collection.get_trajectory_names():
            trajectory = self.collection.get_trajectory(name)
            color = self.collection.get_color(name)
            
            # Add feature group for this trajectory
            fg = folium.FeatureGroup(name=name)
            
            # Add the trajectory as a line
            points = list(zip(trajectory.latitude, trajectory.longitude))
            folium.PolyLine(
                points, 
                color=color, 
                weight=5, 
                opacity=0.7,
                tooltip=name
            ).add_to(fg)
            
            # Add markers for start and end points
            folium.Marker(
                location=[trajectory.latitude[0], trajectory.longitude[0]],
                popup=f"{name} Start",
                icon=folium.Icon(color="green", icon="play"),
                tooltip=f"{name} Start"
            ).add_to(fg)
            
            folium.Marker(
                location=[trajectory.latitude[-1], trajectory.longitude[-1]],
                popup=f"{name} End",
                icon=folium.Icon(color="red", icon="stop"),
                tooltip=f"{name} End"
            ).add_to(fg)
            
            # Add the feature group to the map
            fg.add_to(multi_map)
        
        return multi_map


class MultiPlotlyAnimationVisualizer:
    """Class for creating animated visualizations of multiple trajectories using Plotly."""
    
    def __init__(self, trajectory_collection: TrajectoryCollection) -> None:
        """Initialize visualizer with a trajectory collection.
        
        :param trajectory_collection: Collection of trajectories to visualize
        """
        self.collection = trajectory_collection
    
    def create_animation(self, max_frames: int = 50) -> go.Figure:
        """Create an animated visualization of multiple trajectories.
        
        This method creates a time-synchronized animation where all trajectories play together.
        Each trajectory's time is normalized based on its total duration.
        
        :param max_frames: Maximum number of frames to include in the animation
        :return: Plotly figure object with animation
        """
        if not self.collection.trajectories:
            raise ValueError("Cannot create animation from empty collection")
        
        # Find the trajectory with the longest time range to use as reference
        max_time_range = 0
        for trajectory in self.collection.trajectories.values():
            if len(trajectory.time_diffs) > 0:
                max_time_range = max(max_time_range, trajectory.time_diffs[-1])
        
        # Create normalized time points
        time_points = np.linspace(0, 1, max_frames)
        
        # Create frames for animation
        frames = []
        for frame_idx, norm_time in enumerate(time_points):
            # Calculate the absolute time for this frame
            abs_time = norm_time * max_time_range
            
            frame_data = []
            
            # For each trajectory, add path and current position
            for name in self.collection.get_trajectory_names():
                trajectory = self.collection.get_trajectory(name)
                color = self.collection.get_color(name)
                
                # Find the closest time point in this trajectory's timeline
                if len(trajectory.time_diffs) > 0:
                    # Scale time based on trajectory's own duration
                    traj_rel_time = abs_time * (trajectory.time_diffs[-1] / max_time_range)
                    
                    # Find index of closest time point
                    time_idx = np.abs(trajectory.time_diffs - traj_rel_time).argmin()
                    time_idx = min(time_idx, len(trajectory.latitude) - 1)
                else:
                    time_idx = 0
                
                # Add path up to current point
                frame_data.append(
                    go.Scattermap(
                        lat=trajectory.latitude[:time_idx+1],
                        lon=trajectory.longitude[:time_idx+1],
                        mode="lines",
                        line=dict(width=2, color=color),
                        opacity=0.7,
                        name=f"{name} Path",
                        showlegend=(frame_idx == 0)  # Only show in legend for first frame
                    )
                )
                
                # Add current position - FIXED: removed 'line' property
                frame_data.append(
                    go.Scattermap(
                        lat=[trajectory.latitude[time_idx]],
                        lon=[trajectory.longitude[time_idx]],
                        mode="markers",
                        marker=dict(size=10, color=color),
                        name=f"{name} Position",
                        showlegend=(frame_idx == 0)  # Only show in legend for first frame
                    )
                )
            
            frames.append(
                go.Frame(
                    data=frame_data,
                    name=f"frame{frame_idx}",
                    layout=dict(
                        title=f"Time: {abs_time:.2f} seconds"
                    )
                )
            )
        
        # Initial data (first frame) - FIXED: removed 'line' property if present
        initial_data = []
        for name in self.collection.get_trajectory_names():
            trajectory = self.collection.get_trajectory(name)
            color = self.collection.get_color(name)
            
            # Just show starting points
            initial_data.append(
                go.Scattermap(
                    lat=[trajectory.latitude[0]],
                    lon=[trajectory.longitude[0]],
                    mode="markers",
                    marker=dict(size=10, color=color),
                    name=name
                )
            )
        
        # Center coordinates for the map
        center_lat, center_lon = self.collection.get_center_coordinates()
        
        # Create the figure with initial data
        fig = go.Figure(
            data=initial_data,
            frames=frames,
            layout=dict(
                title="Multiple Target Movement Animation",
                mapbox=dict(
                    style="open-street-map",
                    center=dict(lat=center_lat, lon=center_lon),
                    zoom=13
                ),
                updatemenus=[
                    dict(
                        type="buttons",
                        showactive=False,
                        buttons=[
                            dict(
                                label="Play",
                                method="animate",
                                args=[None, dict(
                                    frame=dict(duration=100, redraw=True),
                                    fromcurrent=True,
                                    mode="immediate"
                                )]
                            ),
                            dict(
                                label="Pause",
                                method="animate",
                                args=[[None], dict(
                                    frame=dict(duration=0, redraw=True),
                                    mode="immediate"
                                )]
                            )
                        ],
                        direction="left",
                        pad=dict(r=10, t=10),
                        x=0.1,
                        y=0,
                    )
                ],
                height=600,
                margin=dict(l=0, r=0, t=50, b=0),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
        )
        
        return fig

#!/usr/bin/python3
#
# Copyright (c) 2023 Ford Motor Company
#
# CONFIDENTIAL - FORD MOTOR COMPANY
#
# This is an unpublished work, which is a trade secret, created in 2023. Ford Motor Company owns all rights to this work and intends
# to maintain it in confidence to preserve its trade secret status. Ford Motor Company reserves the right to protect this work as an
# unpublished copyrighted work in the event of an inadvertent or deliberate unauthorized publication. Ford Motor Company also
# reserves its rights under the copyright laws to protect this work as a published work. Those having access to this work may not
# copy it, use it, or disclose the information contained in it without the written authorization of Ford Motor Company.
#
"""Class for managing and processing multiple trajectory data files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import colorsys

from trajectory_data import TrajectoryData


class TrajectoryCollection:
    """Class for managing a collection of trajectory data files.
    
    This class handles loading multiple trajectory files from a directory and provides
    methods to access and process the collection as a whole.
    """
    
    def __init__(self) -> None:
        """Initialize an empty trajectory collection."""
        self.trajectories: Dict[str, TrajectoryData] = {}
        self.trajectory_colors: Dict[str, str] = {}
    
    def scan_directory(self, directory_path: str | Path, pattern: str = "*.mat") -> List[Path]:
        """Scan a directory for MAT files matching the pattern.
        
        :param directory_path: Path to the directory to scan
        :param pattern: Glob pattern to match files
        :return: List of file paths matching the pattern
        :raises FileNotFoundError: If the directory does not exist
        """
        directory = Path(directory_path)
        if not directory.exists() or not directory.is_dir():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        return list(directory.glob(pattern))
    
    def load_directory(self, directory_path: str | Path, pattern: str = "*.mat") -> None:
        """Load all MAT files in a directory that match the pattern.
        
        :param directory_path: Path to the directory containing MAT files
        :param pattern: Glob pattern to match files
        :raises FileNotFoundError: If the directory does not exist
        """
        file_paths = self.scan_directory(directory_path, pattern)
        
        for file_path in file_paths:
            self.add_trajectory(file_path)
    
    def add_trajectory(self, file_path: str | Path) -> None:
        """Add a trajectory to the collection from a file.
        
        :param file_path: Path to the MAT file to load
        """
        file_path = Path(file_path)
        trajectory_name = file_path.stem
        
        try:
            trajectory = TrajectoryData(file_path)
            self.trajectories[trajectory_name] = trajectory
            
            # Assign a color
            self.trajectory_colors[trajectory_name] = self._get_color_for_index(
                len(self.trajectories) - 1
            )
        except Exception as e:
            print(f"Failed to load trajectory from {file_path}: {e}")
    
    def _get_color_for_index(self, index: int) -> str:
        """Generate a distinct color for a trajectory based on its index.
        
        :param index: Index of the trajectory in the collection
        :return: Hex color code
        """
        # Generate colors using HSV color space for better distinction
        hue = (index * 0.618033988749895) % 1  # Golden ratio method
        saturation = 0.9
        value = 0.95
        
        # Convert to RGB and then to hex
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        return f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"
    
    def get_trajectory_names(self) -> List[str]:
        """Get the names of all trajectories in the collection.
        
        :return: List of trajectory names
        """
        return list(self.trajectories.keys())
    
    def get_trajectory(self, name: str) -> TrajectoryData:
        """Get a specific trajectory by name.
        
        :param name: Name of the trajectory to retrieve
        :return: TrajectoryData object
        :raises KeyError: If the trajectory name is not found
        """
        if name not in self.trajectories:
            raise KeyError(f"Trajectory '{name}' not found in collection")
        
        return self.trajectories[name]
    
    def get_color(self, name: str) -> str:
        """Get the color assigned to a specific trajectory.
        
        :param name: Name of the trajectory
        :return: Color code string
        :raises KeyError: If the trajectory name is not found
        """
        if name not in self.trajectory_colors:
            raise KeyError(f"Trajectory '{name}' not found in collection")
        
        return self.trajectory_colors[name]
    
    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """Get the bounding box containing all trajectories.
        
        :return: Tuple of (min_lat, max_lat, min_lon, max_lon)
        :raises ValueError: If the collection is empty
        """
        if not self.trajectories:
            raise ValueError("Cannot calculate bounding box for empty collection")
        
        min_lat = float('inf')
        max_lat = float('-inf')
        min_lon = float('inf')
        max_lon = float('-inf')
        
        for trajectory in self.trajectories.values():
            min_lat = min(min_lat, np.min(trajectory.latitude))
            max_lat = max(max_lat, np.max(trajectory.latitude))
            min_lon = min(min_lon, np.min(trajectory.longitude))
            max_lon = max(max_lon, np.max(trajectory.longitude))
        
        return min_lat, max_lat, min_lon, max_lon
    
    def get_center_coordinates(self) -> Tuple[float, float]:
        """Get the center coordinates of all trajectories.
        
        :return: Tuple of (center_latitude, center_longitude)
        :raises ValueError: If the collection is empty
        """
        if not self.trajectories:
            raise ValueError("Cannot calculate center for empty collection")
        
        min_lat, max_lat, min_lon, max_lon = self.get_bounding_box()
        return (min_lat + max_lat) / 2, (min_lon + max_lon) / 2
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert all trajectories to a single DataFrame.
        
        :return: DataFrame containing all trajectory data with trajectory name as an identifier
        :raises ValueError: If the collection is empty
        """
        if not self.trajectories:
            raise ValueError("Cannot create DataFrame from empty collection")
        
        dataframes = []
        
        for name, trajectory in self.trajectories.items():
            df = trajectory.to_dataframe()
            df["TrajectoryName"] = name
            df["Color"] = self.get_color(name)
            dataframes.append(df)
        
        return pd.concat(dataframes, ignore_index=True)

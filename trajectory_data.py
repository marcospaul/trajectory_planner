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
"""Classes for loading and processing trajectory data from MATLAB files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final, Dict, Any, Tuple, List, Optional, NamedTuple

import numpy as np
from scipy.io import loadmat


class VelocityUnits(NamedTuple):
    """Container for velocity unit information."""
    
    vel_north: str
    vel_east: str
    vel_down: str
    speed_3d: str


class AccelerationUnits(NamedTuple):
    """Container for acceleration unit information."""
    
    accel_x: str
    accel_y: str
    accel_z: str


class TrajectoryData:
    """Class to load and process trajectory data from MATLAB files.
    
    This class handles loading target trajectory data from MATLAB files and provides
    methods to access and process the position, velocity, and acceleration data.
    """
    
    # Key names for data extraction
    TARGET_POSITION_KEY: Final = "TargetLatitudeLongitude"
    TARGET_VELOCITY_KEY: Final = "TargetVelocity"
    TARGET_ACCEL_KEY: Final = "TargetAccelVehicle"
    
    def __init__(self, file_path: str | Path) -> None:
        """Initialize the trajectory data loader.
        
        :param file_path: Path to the MATLAB file containing trajectory data
        :raises FileNotFoundError: If the specified file does not exist
        :raises KeyError: If the required data keys are not found in the file
        """
        self.file_path = Path(file_path)
        self._validate_file_exists()
        
        # Initialize position attributes
        self.mat_data: Dict[str, Any] = {}
        self.latitude: np.ndarray = np.array([])
        self.longitude: np.ndarray = np.array([])
        self.time: np.ndarray = np.array([])
        self.time_diffs: np.ndarray = np.array([])
        
        # Initialize velocity attributes
        self.vel_north: np.ndarray = np.array([])
        self.vel_east: np.ndarray = np.array([])
        self.vel_down: np.ndarray = np.array([])
        self.speed_3d: np.ndarray = np.array([])
        self.velocity_time: np.ndarray = np.array([])
        self.velocity_units: Optional[VelocityUnits] = None
        
        # Initialize acceleration attributes
        self.accel_x: np.ndarray = np.array([])
        self.accel_y: np.ndarray = np.array([])
        self.accel_z: np.ndarray = np.array([])
        self.acceleration_time: np.ndarray = np.array([])
        self.acceleration_units: Optional[AccelerationUnits] = None
        
        # Load the data
        self._load_data()
        
    def _validate_file_exists(self) -> None:
        """Check if the specified file exists.
        
        :raises FileNotFoundError: If the file does not exist
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
    
    def _load_data(self) -> None:
        """Load data from MATLAB file and extract trajectory information.
        
        :raises KeyError: If required data keys are missing
        """
        self.mat_data = loadmat(str(self.file_path))
        
        # Load position data
        self._load_position_data()
        
        # Load velocity data if available
        try:
            self._load_velocity_data()
        except KeyError:
            print(f"Warning: Velocity data not found in {self.file_path}. Only position data loaded.")
            
        # Load acceleration data if available
        try:
            self._load_acceleration_data()
        except KeyError:
            print(f"Warning: Acceleration data not found in {self.file_path}. Only position and velocity data loaded.")
    
    def _load_position_data(self) -> None:
        """Load position data from the MATLAB file.
        
        :raises KeyError: If position data is missing
        """
        if self.TARGET_POSITION_KEY not in self.mat_data:
            raise KeyError(f"Required key '{self.TARGET_POSITION_KEY}' not found in MATLAB file")
        
        # Extract the trajectory data structure
        target_struct = self.mat_data[self.TARGET_POSITION_KEY]
        record = target_struct[0, 0]
        
        # Extract latitude, longitude, and time data
        self.latitude = record["TargetPosLat"].flatten()
        self.longitude = record["TargetPosLon"].flatten()
        self.time = record["ctime"].flatten()
        
        # Calculate time differences from start
        if len(self.time) > 0:
            self.time_diffs = self.time - self.time[0]
    
    def _load_velocity_data(self) -> None:
        """Load velocity data from the MATLAB file.
        
        :raises KeyError: If velocity data is missing
        """
        if self.TARGET_VELOCITY_KEY not in self.mat_data:
            raise KeyError(f"Required key '{self.TARGET_VELOCITY_KEY}' not found in MATLAB file")
        
        # Extract the velocity data structure
        vel_struct = self.mat_data[self.TARGET_VELOCITY_KEY]
        vel_record = vel_struct[0, 0]
        
        # Extract velocity components
        self.vel_north = vel_record["TargetVelNorth"].flatten()
        self.vel_east = vel_record["TargetVelEast"].flatten()
        self.vel_down = vel_record["TargetVelDown"].flatten()
        self.speed_3d = vel_record["TargetSpeed3D"].flatten()
        
        # Extract time data for velocity (might be the same as position time)
        self.velocity_time = vel_record["ctime"].flatten()
        
        # Extract units if available
        if "units" in vel_record.dtype.names:
            units_record = vel_record["units"][0, 0]
            self.velocity_units = VelocityUnits(
                vel_north=str(units_record["TargetVelNorth"][0]),
                vel_east=str(units_record["TargetVelEast"][0]),
                vel_down=str(units_record["TargetVelDown"][0]),
                speed_3d=str(units_record["TargetSpeed3D"][0])
            )

    def _load_acceleration_data(self) -> None:
        """Load acceleration data from the MATLAB file.
        
        :raises KeyError: If acceleration data is missing
        """
        if self.TARGET_ACCEL_KEY not in self.mat_data:
            raise KeyError(f"Required key '{self.TARGET_ACCEL_KEY}' not found in MATLAB file")
        
        # Extract the acceleration data structure
        accel_struct = self.mat_data[self.TARGET_ACCEL_KEY]
        accel_record = accel_struct[0, 0]
        
        # Extract acceleration components
        self.accel_x = accel_record["TargetAccelX"].flatten()
        self.accel_y = accel_record["TargetAccelY"].flatten()
        self.accel_z = accel_record["TargetAccelZ"].flatten()
        
        # Extract time data for acceleration
        self.acceleration_time = accel_record["ctime"].flatten()
        
        # Extract units if available
        if "units" in accel_record.dtype.names:
            units_record = accel_record["units"][0, 0]
            self.acceleration_units = AccelerationUnits(
                accel_x=str(units_record["TargetAccelX"][0]),
                accel_y=str(units_record["TargetAccelY"][0]),
                accel_z=str(units_record["TargetAccelZ"][0])
            )
    
    def get_coordinate_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the coordinate data.
        
        :return: Dictionary containing coordinate statistics
        """
        return {
            "num_points": len(self.latitude),
            "lat_range": (float(self.latitude.min()), float(self.latitude.max())),
            "lon_range": (float(self.longitude.min()), float(self.longitude.max())),
            "time_range": float(self.time_diffs[-1]) if len(self.time_diffs) > 0 else 0,
        }
    
    def get_velocity_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the velocity data.
        
        :return: Dictionary containing velocity statistics
        """
        if len(self.speed_3d) == 0:
            return {"available": False}
            
        return {
            "available": True,
            "max_speed": float(self.speed_3d.max()),
            "avg_speed": float(np.mean(self.speed_3d)),
            "max_vel_north": float(np.max(np.abs(self.vel_north))),
            "max_vel_east": float(np.max(np.abs(self.vel_east))),
            "max_vel_down": float(np.max(np.abs(self.vel_down))),
            "units": self.velocity_units._asdict() if self.velocity_units else None
        }
    
    def get_acceleration_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the acceleration data.
        
        :return: Dictionary containing acceleration statistics
        """
        if len(self.accel_x) == 0:
            return {"available": False}
        
        # Calculate total acceleration magnitude
        accel_magnitude = np.sqrt(self.accel_x**2 + self.accel_y**2 + self.accel_z**2)
        
        return {
            "available": True,
            "max_accel_magnitude": float(np.max(accel_magnitude)),
            "avg_accel_magnitude": float(np.mean(accel_magnitude)),
            "max_accel_x": float(np.max(np.abs(self.accel_x))),
            "max_accel_y": float(np.max(np.abs(self.accel_y))),
            "max_accel_z": float(np.max(np.abs(self.accel_z))),
            "units": self.acceleration_units._asdict() if self.acceleration_units else None
        }
    
    def calculate_path_metrics(self) -> Dict[str, float]:
        """Calculate metrics about the trajectory path.
        
        :return: Dictionary containing path metrics
        """
        if len(self.latitude) <= 1:
            return {"total_length": 0.0, "avg_step_size": 0.0}
        
        # Calculate distances between consecutive points
        dist = np.sqrt(np.diff(self.longitude)**2 + np.diff(self.latitude)**2)
        
        return {
            "total_length": float(np.sum(dist)),
            "avg_step_size": float(np.mean(dist))
        }
    
    def get_points_for_visualization(self, max_frames: int = 100) -> np.ndarray:
        """Get indices for visualization frames, limiting to a maximum number.
        
        :param max_frames: Maximum number of frames to generate
        :return: Array of indices to use for visualization frames
        """
        num_frames = min(max_frames, len(self.latitude))
        return np.linspace(0, len(self.latitude)-1, num_frames, dtype=int)
    
    def to_dataframe(self) -> "pd.DataFrame":
        """Convert trajectory data to a pandas DataFrame.
        
        :return: DataFrame containing position, velocity, and acceleration data
        """
        import pandas as pd
        
        # Create basic position DataFrame
        df = pd.DataFrame({
            "Latitude": self.latitude,
            "Longitude": self.longitude,
            "Time": self.time,
            "TimeDiff": self.time_diffs
        })
        
        # Determine the minimum length across all data types
        min_length = len(self.latitude)
        
        # Add velocity data if available
        if len(self.speed_3d) > 0:
            min_length = min(min_length, len(self.speed_3d))
        
        # Add acceleration data if available
        if len(self.accel_x) > 0:
            min_length = min(min_length, len(self.accel_x))
        
        # Trim the DataFrame to the minimum length
        df = df.iloc[:min_length].copy()
        
        # Add velocity data if available
        if len(self.speed_3d) > 0 and len(self.speed_3d) >= min_length:
            df["VelNorth"] = self.vel_north[:min_length]
            df["VelEast"] = self.vel_east[:min_length]
            df["VelDown"] = self.vel_down[:min_length]
            df["Speed3D"] = self.speed_3d[:min_length]
        
        # Add acceleration data if available
        if len(self.accel_x) > 0 and len(self.accel_x) >= min_length:
            df["AccelX"] = self.accel_x[:min_length]
            df["AccelY"] = self.accel_y[:min_length]
            df["AccelZ"] = self.accel_z[:min_length]
            df["AccelMagnitude"] = np.sqrt(
                self.accel_x[:min_length]**2 + 
                self.accel_y[:min_length]**2 + 
                self.accel_z[:min_length]**2
            )
        
        return df

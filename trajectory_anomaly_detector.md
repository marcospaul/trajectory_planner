# Trajectory Anomaly Detector

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Ford%20Proprietary-blue.svg)](LICENSE)

A comprehensive tool for detecting anomalies in vehicle trajectory data using statistical methods and machine learning.

## Overview

The `trajectory_anomaly_detector.py` module provides robust capabilities for quality checking trajectory data (GPS positions, velocities, etc.) to identify potential inconsistencies, errors, or anomalous behavior. It combines traditional statistical approaches with machine learning techniques to detect various types of anomalies in trajectory data.

![Example Anomaly Plot](docs/example_anomaly_plot.png)

## Features

- **Multiple Detection Methods**: Statistical analysis, Isolation Forest, DBSCAN clustering
- **Comprehensive Anomaly Types**: Position jumps, unrealistic velocities/accelerations, GPS jitter, zigzag patterns
- **Quality Scoring**: Automated assessment with severity ratings from 0.0-1.0
- **Visual Reports**: Interactive plots showing detected anomalies on trajectories
- **Batch Processing**: Analyze multiple trajectories and generate comparative reports

## Installation

### Prerequisites

- Python 3.8+
- Required packages:
  - numpy
  - pandas
  - scipy
  - matplotlib
  - scikit-learn
  - folium (for interactive maps)
  - plotly (for animations)

### Setup

```bash
git clone https://github.com/yourusername/trajectory-anomaly-detector.git
cd trajectory-anomaly-detector
pip install -r requirements.txt
```

## Quick Start

```python
from trajectory_data import TrajectoryData
from trajectory_anomaly_detector import TrajectoryQualityAnalyzer

# Load trajectory data
trajectory = TrajectoryData("path/to/trajectory.mat")

# Create analyzer with default parameters
analyzer = TrajectoryQualityAnalyzer()

# Analyze trajectory and get report
quality_report = analyzer.analyze_trajectory(trajectory)

# Print report
print(quality_report.get_report_text())

# Visualize anomalies
anomaly_plot = quality_report.plot_anomalies(trajectory)
anomaly_plot.show()
```

## Anomaly Types

The system detects several types of anomalies that commonly occur in trajectory data:

| Type | Description |
|------|-------------|
| **Position Jumps** | Sudden, unrealistic changes in position |
| **Unrealistic Velocities** | Speeds exceeding maximum expected values |
| **Unrealistic Accelerations** | Acceleration rates beyond physical capabilities |
| **GPS Jitter** | Small, random variations in stationary positions |
| **Zigzag Patterns** | Erratic back-and-forth movements indicating noise |
| **Outliers** | Other unusual patterns detected by ML |

## Usage Examples

### Analyzing Multiple Trajectories

```python
from trajectory_collection import TrajectoryCollection
from trajectory_anomaly_detector import TrajectoryQualityAnalyzer
import matplotlib.pyplot as plt

# Load multiple trajectories
collection = TrajectoryCollection()
collection.load_directory("path/to/trajectory/files")

# Create analyzer
analyzer = TrajectoryQualityAnalyzer()

# Analyze all trajectories
reports = analyzer.analyze_collection(collection)

# Generate summary report
summary = analyzer.generate_summary_report()
print(summary)

# Visualize quality comparison
comparison_plot = analyzer.plot_quality_comparison()
plt.show()
```

### Customizing Detection Parameters

```python
# Create analyzer with custom thresholds
analyzer = TrajectoryQualityAnalyzer(
    detector_type="hybrid",        # Options: "statistical", "ml", or "hybrid"
    max_position_jump_m=10.0,      # Maximum position jump in meters
    max_velocity_mps=50.0,         # Maximum velocity in meters per second (180 km/h)
    max_acceleration_mps2=8.0,     # Maximum acceleration in m/s²
    min_gps_jitter_m=0.2           # Minimum distance to consider as GPS jitter
)
```

### Export Reports and Visualizations

```python
import matplotlib.pyplot as plt
from pathlib import Path

# Create reports directory
reports_dir = Path("./trajectory_reports")
reports_dir.mkdir(exist_ok=True)

# Write summary report
with open(reports_dir / "summary_report.txt", "w") as f:
    f.write(analyzer.generate_summary_report())

# Save quality comparison plot
quality_comparison = analyzer.plot_quality_comparison()
quality_comparison.savefig(reports_dir / "quality_comparison.png", dpi=300)

# Write individual reports and save visualizations
for name, report in reports.items():
    safe_name = "".join(c if c.isalnum() else "_" for c in name)
    
    # Save text report
    with open(reports_dir / f"{safe_name}_report.txt", "w") as f:
        f.write(report.get_report_text())
    
    # Save anomaly visualization if anomalies exist
    if report.anomalies:
        trajectory = collection.get_trajectory(name)
        anomaly_plot = report.plot_anomalies(trajectory)
        anomaly_plot.savefig(
            reports_dir / f"{safe_name}_anomalies.png", dpi=300
        )
        plt.close(anomaly_plot)

print(f"Reports exported to {reports_dir}")
```

## Understanding Quality Reports

### Quality Score

Each trajectory receives a quality score between 0.0 and 1.0:
- **1.0**: Perfect quality, no anomalies
- **0.8-0.99**: Good quality, minor anomalies
- **0.5-0.79**: Moderate quality, significant anomalies
- **< 0.5**: Poor quality, severe anomalies

### Sample Report Output

```
Quality Report for Trajectory: Sample_Trajectory
Overall Quality Score: 0.85 out of 1.00
Status: GOOD

Summary Statistics:
  Total Points: 4501
  Total Distance (m): 1245.7821
  Average Velocity (m/s): 12.4532
  Max Velocity (m/s): 21.8765
  Duration (s): 115.2341

Detected Anomalies:
  1. POSITION_JUMP (Severity: 0.35)
     Sudden position jump of 5.84 meters over 0.22 seconds.
     Location: Indices 1203 to 1204
```

## Advanced Configuration

### Detection Methods

Three detection strategies are available:

1. **Statistical (`detector_type="statistical"`)**: Uses thresholds and statistical tests
2. **Machine Learning (`detector_type="ml"`)**: Uses Isolation Forest and DBSCAN clustering
3. **Hybrid (`detector_type="hybrid"`)**: Combines both approaches (recommended)

### Creating Custom Detectors

You can extend the base `TrajectoryAnomalyDetector` class to create custom detectors:

```python
from trajectory_anomaly_detector import TrajectoryAnomalyDetector, AnomalyType, AnomalyInfo, TrajectoryQualityReport

class CustomDetector(TrajectoryAnomalyDetector):
    def detect_anomalies(self, trajectory) -> TrajectoryQualityReport:
        # Your custom detection logic here
        # ...
        return report
```

## Module Structure

```
trajectory_anomaly_detector.py
├── AnomalyType (Enum) - Types of anomalies
├── AnomalyInfo (Dataclass) - Information about detected anomalies
├── TrajectoryQualityReport - Holds analysis results and reporting
├── TrajectoryAnomalyDetector - Base detector class
│   ├── StatisticalAnomalyDetector
│   ├── MachineLearningAnomalyDetector
│   └── HybridAnomalyDetector
└── TrajectoryQualityAnalyzer - High-level interface for running analyses
```

## Tips for Effective Use

1. **Calibrate Thresholds**: Set appropriate thresholds for your specific vehicle types
2. **Pre-process Data**: Clean obvious errors before analysis for better results
3. **Focus on Severe Anomalies**: When reviewing, prioritize anomalies with severity > 0.7
4. **Context Matters**: Consider the operational context when interpreting results
5. **Visualize Results**: Always look at the visualizations, not just numerical scores


## Acknowledgments

- Built with Python scientific computing libraries (NumPy, Pandas, Scikit-learn)
# FootNet

FootNet is a deep learning emulator of atmospheric transport based on a U-Net++ architecture. It generates source-receptor relationships (footprints) for both surface and column-averaged greenhouse gas measurements at 1 km resolution, operating ~650x faster than traditional Lagrangian models such as STILT and X-STILT. FootNet is trained on 500,000 pseudo-observations across the contiguous United States (CONUS) using meteorological data from HRRR.

## Features

- **Surface footprints** for in-situ and dense surface measurements (e.g., BEACO2N, INFLUX)
- **Column footprints** for column-averaged measurements (e.g., TROPOMI, OCO-2/3)
- **~650x speedup** over full-physics Lagrangian models
- Generalizes to regions and meteorological conditions withheld during training
- Supports both flat and complex terrain
- Footprints can be computed on-the-fly or saved to disk in NetCDF format

## Repository structure

```
FootNet/
├── footnet/
│   ├── ColumnFootNet.py
│   ├── SurfaceFootNet.py
│   ├── getColumnMeteorology.py
│   ├── getSurfaceMeteorology.py
│   ├── ExampleColumnFootNet.ipynb
│   ├── ExampleSurfaceFootNet.ipynb
│   ├── unetpp_model.py
│   └── get_HRRR_met_lite_files.py
├── models/
│   ├── ColumnFootNet_in_sample.pth
│   └── SurfaceFootNet_in_sample.pth
├── inversion/
│   ├── BEACO2N/
│   └── TROPOMI/
├── data/
│   └── HRRR_lon_lat.npz
└── README.md
```

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/nd349/FootNet.git
cd FootNet
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Try the example notebooks

- [`footnet/ExampleSurfaceFootNet.ipynb`](footnet/ExampleSurfaceFootNet.ipynb)
- [`footnet/ExampleColumnFootNet.ipynb`](footnet/ExampleColumnFootNet.ipynb)

## Usage

### Surface footprints

```python
import datetime
import numpy as np
from getSurfaceMeteorology import SurfaceMeteorology
from SurfaceFootNet import SurfaceFootNet

# Load HRRR grid coordinates
hr3lon_full = np.load('data/HRRR_lon_lat.npz')['lon']
hr3lat_full = np.load('data/HRRR_lon_lat.npz')['lat']
hr3lon_full = (hr3lon_full + 180) % 360 - 180  # convert to -180~180

# Define spatial domain (1/120 degree ~ 1 km resolution)
lats = np.arange(30.384, 33.717, 1/120)
lons = np.arange(-96.323, -92.990, 1/120)

# Define receptor
receptor = [datetime.datetime(2021, 10, 20, 16), -94.657, 32.618]
receptors = [receptor]

# Load meteorology
input_met = SurfaceMeteorology(
    [receptor[0]], lons, lats, trimsize=150,
    hr3lat_full=hr3lat_full, hr3lon_full=hr3lon_full,
    HRRR_DIR='PATH', backhours=[0, 6, 12, 18, 24]
)

# Run inference
model = SurfaceFootNet(model_path='models/SurfaceFootNet_in_sample.pth')
foots, ref_indices, ref_timestamps, ref_rlons, ref_rlats = model.run_inference(receptors, input_met)
```

### Column footprints

```python
import datetime
import numpy as np
from getColumnMeteorology import ColumnMeteorology
from ColumnFootNet import ColumnFootNet

# Load HRRR grid coordinates
hr3lon_full = np.load('data/HRRR_lon_lat.npz')['lon']
hr3lat_full = np.load('data/HRRR_lon_lat.npz')['lat']
hr3lon_full = (hr3lon_full + 180) % 360 - 180

# Define spatial domain
lats = np.arange(43.002, 46.335, 1/120)
lons = np.arange(-87.390, -84.058, 1/120)

# Define receptor
receptor = [datetime.datetime(2021, 7, 27, 6), -86.253, 44.314]
receptors = [receptor]

# Load meteorology
input_met = ColumnMeteorology(
    [receptor[0]], lons, lats, trimsize=150,
    hr3lat_full=hr3lat_full, hr3lon_full=hr3lon_full,
    HRRR_DIR='PATH', backhours=[0, 6, 12, 18, 24]
)

# Run inference
model = ColumnFootNet(model_path='models/ColumnFootNet_in_sample.pth')
foots, ref_indices, ref_timestamps, ref_rlons, ref_rlats = model.run_inference(receptors, input_met)
```

## Outputs

Each footprint has dimensions (400, 400) at 1 km spatial resolution, with units of ppm / (µmol m⁻² s⁻¹). Outputs are returned as numpy arrays and can optionally be saved to NetCDF.

## Meteorological inputs

FootNet uses HRRR meteorology regridded to 1 km resolution. Inputs are provided at 0, 6, 12, 18, and 24 hours before the receptor time.

- **Surface FootNet** uses: U10, V10, PBL height, surface pressure (24 channels total)
- **Column FootNet** adds: U850, V850, U500, V500, T850 (49 channels total)

We recommend converting HRRR files to NetCDF "lite" files for efficient data loading using [`footnet/get_HRRR_met_lite_files.py`](footnet/get_HRRR_met_lite_files.py). Other meteorological products (GFS, WRF, ERA-5) are also supported.

## Model weights

Pre-trained weights are available in the [`models/`](models/) directory:

- [`SurfaceFootNet_in_sample.pth`](models/SurfaceFootNet_in_sample.pth)
- [`ColumnFootNet_in_sample.pth`](models/ColumnFootNet_in_sample.pth)

## Training

FootNet was trained using PyTorch's Distributed Data Parallel (DDP) across 10–15 nodes. Training scripts:

- [`training/SurfaceFootNet_multinode_training.py`](training/SurfaceFootNet_multinode_training.py)
- [`training/ColumnFootNet_multinode_training.py`](training/ColumnFootNet_multinode_training.py)

These scripts can also be used as a reference for fine-tuning FootNet on new data.

## References

Dadheech, N. and Turner, A. J.: Simulating out-of-sample atmospheric transport to enable flux inversions, *Atmospheric Chemistry and Physics*, 26, 427–441, https://doi.org/10.5194/acp-26-427-2026

Dadheech, N.\*, He T.\*, and Turner, A. J.: High-resolution greenhouse gas flux inversions using a machine learning surrogate model for atmospheric transport, *Atmospheric Chemistry and Physics*, 25(10), 5159–5174, https://doi.org/10.5194/acp-25-5159-2025

He, T.\*, Dadheech, N.\*, and Turner, A. J.: FootNet v1.0: development of a machine learning emulator of atmospheric transport, *Geoscientific Model Development*, 18(5), 1661–1671, https://doi.org/10.5194/gmd-18-1661-2025

## People

- **Alex Turner** (PI) — turneraj@uw.edu — https://alexjturner.github.io
- **Nikhil Dadheech** (Developer) — nd349@uw.edu — https://nd349.github.io
- **Tailong He** (Developer) — tlhe@mit.edu — https://tailonghe.github.io

## Acknowledgements

This work originated from research supported by NASA (grant nos. 80NSSC22K1557 and 80NSSC21K1808) and the Environmental Defense Fund. It has continued as part of the FETCH4 project, supported by Schmidt Sciences through the VESRI program.

## License

See [LICENSE](LICENSE) for details.

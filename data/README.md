# Data

Speed data is derived from INRIX commercial GPS probe vehicle observations
accessed via RITIS under a data-use agreement and cannot be redistributed.

## Accessing the data

Contact the corresponding author (abae@andrew.cmu.edu) or access INRIX/RITIS
data via your institution's data-sharing agreement.

## Expected directory structure

```
data/
├── tsmo/
│   ├── tsmo_speed_data.parquet      # 5-min speed by link (mph)
│   ├── tsmo_incident_reports.parquet
│   ├── tsmo_network.geojson
│   ├── tsmo_upstream_mapping.json
│   ├── weather.pkl
│   ├── nr_labels_full.parquet       # binary NR labels (T × N)
│   ├── v_rec_full.parquet           # recurrent lower-bound speed
│   └── adj_mx.npy                   # (228, 228) adjacency matrix
└── cranberry/
    └── (same structure, N=78 links)
```

## NR Labels

The NR labels are produced by the ensemble detection framework described in:

> Bae, A.J. (2026). Traffic State Based Labeling of Nonrecurrent
> Disturbances from Speed Data Using Interpretable Ensemble Detection.
> *Transportation Research Part C* (under review).

The label generation script is at `scripts/generate_nr_labels.py`
(requires Mac with the raw data files).

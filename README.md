### Installation

```commandline
pip install segment-geospatial python-decouple rasterio matplotlib
```

### Running
Listing dates with satellite images without cloud
```commandline
python3 app.py --car {CAR} --listdates
```
Segment and Vector
```commandline
python3 app.py --car {CAR} --date {date} --band_type NDVI
```
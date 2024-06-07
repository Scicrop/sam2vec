# Sam2Vec
Sam2Vec is an application that uses Segment Anything Model (https://segment-anything.com/), the first AI foundation model for image segmentation.

## Installation

```commandline
pip install segment-geospatial python-decouple rasterio matplotlib
```

## Running
Listing dates with satellite images without cloud
```commandline
python3 app.py --car {CAR} --listdates
```
Segment and Vector
```commandline
python3 app.py --car {CAR} --date {date} --band_type NDVI
```

## Results

https://github.com/Scicrop/sam2vec/assets/692043/30ae274a-b1cd-42b0-9b12-debf2f7bff6a


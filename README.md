# Wall Segmentation Model

This repository contains the wall semantic segmentation model training and data preparation code.


MLflow is used for experimentation tracking, to launch it, run `mlflow ui --backend-store-uri ./notebooks/mlruns`

Start model training in the [notebooks/wall_segmentation.ipynb](notebooks/wall_segmentation.ipynb) notebook.
Room wall scene dataset filtering is implemented in [src/dataset.py](src/dataset.py).
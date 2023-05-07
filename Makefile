train:
	python src/train.py

download_data:
	bash ./scripts/download_ade20k.sh

start_tensorboard:
	tensorboard --logdir ./notebooks/tb_logs --bind_all

start_mlflow:
	mlflow ui --backend-store-uri ./notebooks/mlruns
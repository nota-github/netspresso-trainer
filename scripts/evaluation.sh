model="resnet18-classification"

python evaluation.py \
  --data config/${model}/data.yaml \
  --augmentation config/${model}/augmentation.yaml \
  --model config/${model}/model.yaml \
  --logging config/${model}/logging.yaml \
  --environment config/${model}/environment.yaml

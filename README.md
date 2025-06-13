# YOLOX

## Install packages

```bash
git clone https://github.com/nota-github/netspresso-trainer.git@v2-yolox-demo
pip install -e .
```

## Get YOLOX ExportedProgram

```bash
python tools/exir_convert.py --config-path demo_configs/model.yaml --output-dir exir/ --num-classes 80 --sample-size 640 640
```

## XNNPACK lowering

```bash
python tools/xnnpack_lowering.py --model-path exir/model.pt2 --output-dir exir/ --sample-size 640 640
```

## Run evaluation

```bash
python evaluation.py \
    --data config/data/local/coco2017.yaml \
    --augmentation config/augmentation/detection.yaml \
    --model config/model/yolox/yolox-s-detection.yaml \
    --logging config/logging.yaml \
    --environment config/environment.yaml
```

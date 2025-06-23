# YOLOX

## Install packages

```bash
git clone --branch v2-yolox-demo https://github.com/nota-github/netspresso-trainer.git
pip install -e .
```

## Get YOLOX ExportedProgram

```bash
python tools/exir_convert.py --config-path demo_configs/yolox_s.yaml --output-dir exir/ --num-classes 80 --sample-size 640 640
```

## XNNPACK lowering

```bash
python tools/xnnpack_lowering.py --model-path exir/yolox_s.pt2 --output-dir exir/ --sample-size 640 640
```

## Run evaluation

```bash
python evaluation.py \
    --data demo_configs/coco2017.yaml \
    --augmentation demo_configs/augmentation.yaml \
    --model demo_configs/yolox_s_pte.yaml \
    --logging demo_configs/logging.yaml \
    --environment demo_configs/environment.yaml
```

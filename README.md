# YOLOX

## Install packages

```bash
git clone --branch v2-yolox-demo https://github.com/nota-github/netspresso-trainer.git
pip install -e .
```

## Get YOLOX ExportedProgram

Export with SiLU separated into Sigmoid + Mul version
https://github.com/nota-github/netspresso-trainer/blob/2e0ffd3abf71b89f23f045b3288b2856c722dc68/src/netspresso_trainer/models/op/custom_act.py#L22-L56

torch silu kernel (cpu)
- https://github.com/pytorch/pytorch/blob/5dd9652389ed7959a842323e4ce063f553710e47/aten/src/ATen/native/cpu/Activation.cpp#L1128-L1158

torch sigmoid kernel (cpu)
- https://github.com/pytorch/pytorch/blob/5dd9652389ed7959a842323e4ce063f553710e47/aten/src/ATen/native/cpu/Activation.cpp#L29-L93

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

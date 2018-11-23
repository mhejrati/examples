#!/usr/bin/env bash

just create job single --name tiny_imagenet_train_gpu_3 --datasets artem-stable/artem-tiny-imagenet-example \
--command "python -m tiny_imagenet_pytorch.train_and_eval --num_epochs 30" \
--project artem-stable/tests-artem --instance-type p3.2xlarge --time-limit 4h \
--docker-image pytorch-0.4.0-gpu-py36-cuda9.2 --setup-command "pip install -r requirements_pytorch.txt"

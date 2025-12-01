#!/bin/bash
torchrun --nproc_per_node=2 scripts/train.py

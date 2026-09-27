#!/bin/bash
echo "Starting Training for X3D-XS..."
python3 multitask_vqa_final.py --variant xs > train_xs.log 2>&1
echo "X3D-XS Training Completed!"

echo "Starting Training for X3D-S..."
python3 multitask_vqa_final.py --variant s > train_s.log 2>&1
echo "X3D-S Training Completed!"

echo "Starting Training for X3D-M..."
python3 multitask_vqa_final.py --variant m > train_m.log 2>&1
echo "X3D-M Training Completed!"

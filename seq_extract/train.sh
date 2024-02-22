#!/bin/bash
nohup  python -u train_phase_1.py     > output_1.log 2>&1 &
nohup  python -u train_phase_2.py     > output_2.log 2>&1 &







#!/bin/bash

# Set log file
LOG_FILE="training_log.txt"

# Run training command and redirect output to log file
nohup python main.py --mode train_and_evaluate --unet_epochs 5000 --simple_rl_episodes 5000 --batch_size 8 --lr 1e-4 --max_steps 10 --eval_samples 20 --eval_interval 5 --num_eval_episodes 3 --save_visualizations > $LOG_FILE 2>&1 &

# Get background process ID
PID=$!

# Print information
echo "Training started in background, process ID: $PID"
echo "Use 'tail -f $LOG_FILE' command to view real-time training logs"
echo "Or use 'ps -p $PID' to check if the process is still running" 
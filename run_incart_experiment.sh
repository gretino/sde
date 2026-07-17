#!/bin/bash

# Default values
GPU=0
ACTION="verify" # or "train"
ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -g|--gpu)
      GPU="$2"
      shift 2
      ;;
    -a|--action)
      ACTION="$2"
      shift 2
      ;;
    *)
      ARGS="$ARGS $1"
      shift
      ;;
  esac
done

export CUDA_VISIBLE_DEVICES=$GPU

if [ "$ACTION" == "train" ]; then
  echo "Running training on GPU $GPU with args: $ARGS"
  conda run -n sde python run_incart_experiment.py $ARGS
elif [ "$ACTION" == "verify" ]; then
  echo "Running verification on GPU $GPU with args: $ARGS"
  conda run -n sde python verify_pipeline.py $ARGS
else
  echo "Unknown action: $ACTION. Use 'train' or 'verify'."
  exit 1
fi

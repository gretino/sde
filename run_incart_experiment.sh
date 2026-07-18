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

# Determine python command
if [ "$CONDA_DEFAULT_ENV" == "sde" ]; then
  PYTHON_CMD="python"
elif [ -n "$CONDA_EXE" ] && [ -x "$CONDA_EXE" ]; then
  PYTHON_CMD="$CONDA_EXE run -n sde python"
else
  PYTHON_CMD="conda run -n sde python"
fi

if [ "$ACTION" == "train" ]; then
  echo "Running training on GPU $GPU with args: $ARGS"
  $PYTHON_CMD run_incart_experiment.py $ARGS
elif [ "$ACTION" == "verify" ]; then
  echo "Running verification on GPU $GPU with args: $ARGS"
  $PYTHON_CMD verify_pipeline.py $ARGS
elif [ "$ACTION" == "verify_single" ] || [ "$ACTION" == "verify-single" ]; then
  echo "Running single-sample verification on GPU $GPU with args: $ARGS"
  $PYTHON_CMD verify_single.py $ARGS
else
  echo "Unknown action: $ACTION. Use 'train', 'verify', or 'verify_single'."
  exit 1
fi

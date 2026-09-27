#!/usr/bin/env bash
# Launch the cross-encoder as a SageMaker training job (GPU, stops itself).
#   ./sm_cross_encoder.sh <job-name> <s3-data-prefix> [instance-type]
# <s3-data-prefix> holds train/, eval/ and optionally score/ parquet folders.
set -euo pipefail
JOB="$1"; DATA="$2"; TYPE="${3:-ml.g6e.xlarge}"
B=s3://sagemaker-us-east-1-677123926505/ml-challenge/ce
ROLE=arn:aws:iam::677123926505:role/service-role/AmazonSageMaker-ExecutionRole-20260925T164130
IMAGE=763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-sagemaker

channels=""
for ch in train eval score model; do
  if aws s3 ls "$DATA/$ch/" >/dev/null 2>&1; then
    channels+="{\"ChannelName\":\"$ch\",\"DataSource\":{\"S3DataSource\":{\"S3DataType\":\"S3Prefix\",\"S3Uri\":\"$DATA/$ch/\",\"S3DataDistributionType\":\"FullyReplicated\"}}},"
  fi
done

aws sagemaker create-training-job --training-job-name "$JOB" --role-arn "$ROLE" \
  --algorithm-specification "TrainingImage=$IMAGE,TrainingInputMode=File" \
  --hyper-parameters "{\"sagemaker_program\":\"\\\"cross_encoder.py\\\"\",\"sagemaker_submit_directory\":\"\\\"$B/code/sourcedir.tar.gz\\\"\",\"model\":\"\\\"${CE_MODEL:-xlm-roberta-base}\\\"\",\"epochs\":\"${CE_EPOCHS:-1}\",\"bs\":\"${CE_BS:-128}\",\"lr\":\"${CE_LR:-3e-5}\"}" \
  --input-data-config "[${channels%,}]" \
  --output-data-config "S3OutputPath=$B/jobs" \
  --resource-config "InstanceType=$TYPE,InstanceCount=1,VolumeSizeInGB=100" \
  --stopping-condition "MaxRuntimeInSeconds=${CE_MAX_SECONDS:-14400}" \
  --query TrainingJobArn --output text

#!/usr/bin/env bash
# Launch the dense-retrieval blocking job (dense_encoder.py) on SageMaker.
#   ./sm_dense.sh <job-name> [instance-type]
set -euo pipefail
JOB="$1"; TYPE="${2:-ml.g6.4xlarge}"
B=s3://sagemaker-us-east-1-677123926505/ml-challenge
aws sagemaker create-training-job --training-job-name "$JOB" \
  --role-arn arn:aws:iam::677123926505:role/service-role/AmazonSageMaker-ExecutionRole-20260925T164130 \
  --algorithm-specification "TrainingImage=763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-sagemaker,TrainingInputMode=File" \
  --hyper-parameters "{\"sagemaker_program\":\"\\\"dense_encoder.py\\\"\",\"sagemaker_submit_directory\":\"\\\"$B/dense/code/sourcedir.tar.gz\\\"\",\"s3-out\":\"\\\"$B/dense/out/$JOB\\\"\"}" \
  --input-data-config "[{\"ChannelName\":\"records\",\"DataSource\":{\"S3DataSource\":{\"S3DataType\":\"S3Prefix\",\"S3Uri\":\"$B/dense/records/\",\"S3DataDistributionType\":\"FullyReplicated\"}}}]" \
  --output-data-config "S3OutputPath=$B/dense/jobs" \
  --resource-config "InstanceType=$TYPE,InstanceCount=1,VolumeSizeInGB=150" \
  --stopping-condition "MaxRuntimeInSeconds=21600" --query TrainingJobArn --output text

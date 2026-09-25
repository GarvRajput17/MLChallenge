#!/usr/bin/env bash
# Wait for the S3 dataset upload to finish, then kick off the pipeline on EC2.
set -uo pipefail
export PATH=/usr/local/bin:$PATH
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
BUCKET=ml-challenge-675830988230
EXPECT=8          # 3 train sources + ground truth + 3 test sources + 1 slack

echo "[$(date +%H:%M:%S)] waiting for upload to finish..."
while pgrep -f "aws s3 sync student_resource/dataset" >/dev/null; do sleep 20; done
echo "[$(date +%H:%M:%S)] sync process exited; verifying objects"

N=$(aws s3 ls "s3://$BUCKET/dataset/" --recursive | grep -c '\.tsv$')
echo "[$(date +%H:%M:%S)] $N tsv objects in S3"
if [ "$N" -lt 7 ]; then
    echo "INCOMPLETE ($N/7 tsv files) -- retrying sync once"
    aws s3 sync student_resource/dataset "s3://$BUCKET/dataset" --only-show-errors
    N=$(aws s3 ls "s3://$BUCKET/dataset/" --recursive | grep -c '\.tsv$')
    echo "after retry: $N"
fi

echo "[$(date +%H:%M:%S)] pulling dataset onto the box"
./work/scripts/ssm.sh run 'mkdir -p /home/ubuntu/mlchallenge/student_resource && \
    aws s3 sync s3://ml-challenge-675830988230/dataset \
        /home/ubuntu/mlchallenge/student_resource/dataset --only-show-errors && \
    du -sh /home/ubuntu/mlchallenge/student_resource/dataset && \
    wc -l /home/ubuntu/mlchallenge/student_resource/dataset/*/*.tsv' 1800

echo "[$(date +%H:%M:%S)] launching prep (normalisation + lexicons)"
./work/scripts/ssm.sh bg prep \
    'python3 -u src/prep_basic.py && python3 -u src/learn_lexicons.py && python3 -u src/normalize.py'
echo "[$(date +%H:%M:%S)] prep launched -- watch with: ./work/scripts/ssm.sh log prep"

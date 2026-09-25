#!/usr/bin/env bash
# Launch the CPU work box, wired for SSH from this machine only.
#   ./aws_launch.sh plan      show exactly what would be created + cost, change nothing
#   ./aws_launch.sh up        create key pair, security group, instance; print EC2_HOST
#   ./aws_launch.sh ip        public IP of the running box
#   ./aws_launch.sh stop      stop  (keeps the disk + data; pay only storage)
#   ./aws_launch.sh start     start it again
#   ./aws_launch.sh kill      TERMINATE (destroys the disk)
set -euo pipefail
export PATH=/usr/local/bin:$PATH

TYPE="${EC2_TYPE:-r7i.2xlarge}"      # 8 vCPU / 64 GB -- the most our 8-vCPU quota allows
DISK_GB="${EC2_DISK:-300}"
NAME="${EC2_NAME:-ml-challenge}"
KEY_NAME="$NAME"
SG_NAME="$NAME-sg"
PUBKEY="${EC2_PUBKEY:-$HOME/.ssh/id_ed25519.pub}"

ami() {
  aws ssm get-parameter --name \
    /aws/service/canonical/ubuntu/server/22.04/stable/current/amd64/hvm/ebs-gp2/ami-id \
    --query Parameter.Value --output text
}
instance_id() {
  aws ec2 describe-instances --filters "Name=tag:Name,Values=$NAME" \
      "Name=instance-state-name,Values=pending,running,stopping,stopped" \
      --query "Reservations[].Instances[0].InstanceId" --output text 2>/dev/null | head -1
}

case "${1:-plan}" in
plan)
    echo "region     : $(aws configure get region)"
    echo "instance   : $TYPE   (8 vCPU / 64 GB -- the 8-vCPU Standard quota ceiling)"
    echo "AMI        : $(ami)  (Ubuntu 22.04 LTS)"
    echo "disk       : ${DISK_GB} GB gp3"
    echo "key pair   : $KEY_NAME  <- imported from $PUBKEY"
    echo "security   : $SG_NAME, inbound TCP 22 from $(curl -s https://checkip.amazonaws.com)/32 only"
    echo
    echo "approximate cost (us-east-1, VERIFY in console -- rates move):"
    echo "  compute  ~\$0.53/hr while RUNNING, \$0 while stopped"
    echo "  disk     ~\$24/month for ${DISK_GB} GB gp3, charged even when stopped"
    echo "  30 hours of actual work  =>  roughly \$16 compute + ~\$1 disk"
    echo
    echo "nothing has been created. run './aws_launch.sh up' to proceed."
    ;;
up)
    MYIP="$(curl -s https://checkip.amazonaws.com)"
    aws ec2 import-key-pair --key-name "$KEY_NAME" \
        --public-key-material "fileb://$PUBKEY" >/dev/null 2>&1 \
        && echo "imported key pair $KEY_NAME" || echo "key pair $KEY_NAME already exists"

    SG_ID=$(aws ec2 describe-security-groups --group-names "$SG_NAME" \
            --query "SecurityGroups[0].GroupId" --output text 2>/dev/null) || SG_ID=""
    if [ -z "$SG_ID" ] || [ "$SG_ID" = "None" ]; then
        VPC=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true \
              --query "Vpcs[0].VpcId" --output text)
        SG_ID=$(aws ec2 create-security-group --group-name "$SG_NAME" \
                --description "ML challenge work box" --vpc-id "$VPC" \
                --query GroupId --output text)
        echo "created security group $SG_ID in $VPC"
    fi
    aws ec2 authorize-security-group-ingress --group-id "$SG_ID" \
        --protocol tcp --port 22 --cidr "$MYIP/32" >/dev/null 2>&1 \
        && echo "opened SSH from $MYIP/32" || echo "SSH rule for $MYIP/32 already present"

    ID=$(aws ec2 run-instances --image-id "$(ami)" --instance-type "$TYPE" \
        --key-name "$KEY_NAME" --security-group-ids "$SG_ID" --count 1 \
        --instance-initiated-shutdown-behavior stop \
        --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$DISK_GB,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME}]" \
        --query "Instances[0].InstanceId" --output text)
    echo "launched $ID -- waiting for it to come up..."
    aws ec2 wait instance-status-ok --instance-ids "$ID"
    IP=$(aws ec2 describe-instances --instance-ids "$ID" \
         --query "Reservations[0].Instances[0].PublicIpAddress" --output text)
    echo
    echo "READY.  export EC2_HOST=ubuntu@$IP"
    ;;
ip)   aws ec2 describe-instances --instance-ids "$(instance_id)" \
          --query "Reservations[0].Instances[0].PublicIpAddress" --output text ;;
stop) aws ec2 stop-instances --instance-ids "$(instance_id)" \
          --query "StoppingInstances[0].CurrentState.Name" --output text ;;
start) aws ec2 start-instances --instance-ids "$(instance_id)" >/dev/null
      aws ec2 wait instance-status-ok --instance-ids "$(instance_id)"
      "$0" ip ;;
kill) echo "This DESTROYS the disk and everything on it."
      read -r -p "type the instance name '$NAME' to confirm: " c
      [ "$c" = "$NAME" ] && aws ec2 terminate-instances --instance-ids "$(instance_id)" \
          --query "TerminatingInstances[0].CurrentState.Name" --output text \
          || echo "aborted" ;;
*) sed -n '2,10p' "$0"; exit 1 ;;
esac

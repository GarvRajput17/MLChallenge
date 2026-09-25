#!/usr/bin/env bash
# Read-only preflight: identity, region, EC2 vCPU quotas, running instances, spend.
# Run this right after `aws configure` -- it answers "can we actually launch?"
set -uo pipefail

line() { printf '\n== %s ==\n' "$1"; }

line "identity"
aws sts get-caller-identity --output table 2>&1 || {
    echo "!! credentials not working. Run:  aws configure"; exit 1; }

REGION=$(aws configure get region 2>/dev/null || echo "")
echo "configured region: ${REGION:-<none set>}"

line "EC2 vCPU quotas (these are what block a launch)"
echo "NOTE: values are vCPU counts, not instance counts, and are PER REGION."
aws service-quotas list-service-quotas --service-code ec2 \
    --query "Quotas[?contains(QuotaName, 'On-Demand') || contains(QuotaName, 'Spot Instance Requests')].[Value,QuotaCode,QuotaName]" \
    --output text 2>/dev/null | sort -rn | head -14 \
    || echo "(needs servicequotas:ListServiceQuotas permission)"

line "what each quota allows"
cat <<'EOT'
  r7i.xlarge    =  4 vCPU / 32 GB     r7i.4xlarge  = 16 vCPU / 128 GB
  r7i.2xlarge   =  8 vCPU / 64 GB     r7i.8xlarge  = 32 vCPU / 256 GB
  g5.2xlarge    =  8 vCPU / 1x A10G   g5.12xlarge  = 48 vCPU / 4x A10G
  A fresh account is often capped at 5 vCPU Standard and 0 G -- both need raising.
EOT

line "running instances (anything here is costing money)"
aws ec2 describe-instances \
    --query "Reservations[].Instances[?State.Name!='terminated'].[InstanceId,InstanceType,State.Name,PublicIpAddress,LaunchTime]" \
    --output table 2>/dev/null || echo "(none / no permission)"

line "key pairs"
aws ec2 describe-key-pairs --query "KeyPairs[].[KeyName,KeyType]" --output table 2>/dev/null \
    || echo "(none yet -- import ~/.ssh/id_ed25519.pub before launching)"

line "month-to-date spend"
aws ce get-cost-and-usage \
    --time-period Start=$(date -v1d +%Y-%m-%d 2>/dev/null || date +%Y-%m-01),End=$(date +%Y-%m-%d) \
    --granularity MONTHLY --metrics UnblendedCost \
    --query "ResultsByTime[].Total.UnblendedCost.[Amount,Unit]" --output text 2>/dev/null \
    || echo "(Cost Explorer not enabled or no ce:GetCostAndUsage permission -- check Billing console)"

echo
echo "Credits are NOT exposed by any API. Check manually:"
echo "  Billing and Cost Management -> Credits -> 'Applicable services' + 'Expiration date'"

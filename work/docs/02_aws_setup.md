# AWS setup: EC2 instead of SageMaker

## 0. Why SageMaker blocked you

**SageMaker quotas and EC2 quotas are completely separate pools.** They are different
services with different Service Quotas entries, and having 0 in one says *nothing* about
the other:

| Service | Quota entry looks like |
|---|---|
| SageMaker | `ml.g5.2xlarge for training job usage`, `ml.g4dn.xlarge for notebook instance usage` |
| EC2 | `Running On-Demand G and VT instances`, `Running On-Demand Standard (A, C, D, H, I, M, R, T, Z) instances` |

Note the `ml.` prefix — those are SageMaker-managed instances, billed and quota'd
separately from the identically-named EC2 hardware. So moving to EC2 is a legitimate
route around a SageMaker quota, **not** a workaround that will hit the same wall
automatically.

Two caveats before investing time:

1. **EC2 has its own quotas**, and they are also often low on a fresh account
   (frequently ~5 vCPUs for the Standard family, and **0 for the G/GPU family**).
   Quotas are **per-region**. Check before launching, request early — approval ranges
   from minutes to a couple of days.
2. The literal phrase *"quota expired"* is unusual — normal quota errors say *exceeded*.
   That wording suggests either a **time-boxed quota grant** attached to the challenge
   account, or an expired *quota-increase request*. Worth capturing the exact error text
   before assuming it is a plain limit problem.

---

## 1. Check this FIRST — do the credits even cover EC2?

⚠️ **This is the one thing that can waste a whole day.** AWS promotional credits are often
restricted to specific services, and credits handed out for an ML challenge are sometimes
**SageMaker-only**. If so, EC2 usage bills to a real payment method.

> Billing and Cost Management → **Credits** → look at the **Applicable services** and
> **Expiration date** columns for the $600 credit.

If it says something like "Amazon SageMaker only", stop and either fight the SageMaker
quota or ask the organisers. If it says "All services" or lists EC2, proceed.

---

## 2. Check and raise EC2 quotas

> Service Quotas console → **AWS services** → **Amazon Elastic Compute Cloud (Amazon EC2)**

Pick your region first (top-right) and keep everything in that one region — EBS volumes,
S3 buckets and instances must co-locate or you pay for cross-region transfer.

Quotas that matter, all **measured in vCPUs, not instance counts**:

| Quota | Needed for | Ask for |
|---|---|---|
| `Running On-Demand Standard (A, C, D, H, I, M, R, T, Z) instances` | the CPU work box (`r7i`, `m7i`, `c7i`) | ≥ 64 |
| `Running On-Demand G and VT instances` | GPU box (`g5`, `g6`) | ≥ 48 |
| `All G and VT Spot Instance Requests` | cheaper GPU experiments | ≥ 48 |

Request all three now, in parallel, before doing anything else — they are free to request
and the GPU one is the long pole. Give a real justification ("ML competition, training a
sentence encoder for entity resolution") — vague requests get auto-rejected more often.

---

## 3. What to launch

Two instances, because paying GPU rates to run pandas is how you burn $600 in a weekend.

### Work box (Phase A — normalisation, lexicons, blocking, features, LightGBM)

- **Type:** `r7i.4xlarge` (16 vCPU / 128 GiB) to start; step up to `r7i.8xlarge`
  (32 vCPU / 256 GiB) if blocking needs it. Memory matters more than cores here.
- **AMI:** Ubuntu Server 22.04 LTS (x86_64) — plain, no GPU drivers needed.
- **Storage:** 500 GB **gp3** root volume. Our intermediates already run ~3 GB locally and
  will grow a lot with candidate features. gp3 lets you raise IOPS/throughput independently
  and is cheaper than gp2.

### GPU box (Phase B — bi-encoder fine-tune, embeddings, cross-encoder)

- **Type:** `g5.2xlarge` (1×A10G 24 GB) for development; `g5.12xlarge` (4×A10G,
  48 vCPU, 192 GiB) for the real training and batch-inference runs.
- **AMI:** **Deep Learning AMI (Ubuntu 22.04)** — comes with CUDA, cuDNN and PyTorch
  preinstalled. Do not hand-install drivers; it burns hours.
- **Spot for experiments, on-demand for the final inference run** that produces the
  submission. A spot reclaim halfway through scoring 10⁸ pairs is a bad afternoon.
- **Start it only for bursts. Stop it the moment the job finishes.**

> **Stop ≠ Terminate.** *Stop* keeps the EBS volume and all your data; you pay only
> storage (a few dollars/month for 500 GB gp3). *Terminate* destroys it. Always stop.

---

## 4. How you actually connect

### 4a. Key pair + security group (one-time)

At launch, EC2 asks for a **key pair**. You already have `~/.ssh/id_ed25519.pub` locally —
import it rather than letting AWS generate a new `.pem`:

> EC2 → Network & Security → **Key Pairs** → *Actions* → **Import key pair** → paste the
> contents of `~/.ssh/id_ed25519.pub`

For the **security group**, allow inbound SSH (TCP 22) from **My IP** only — never
`0.0.0.0/0`. Your home IP changes, so expect to edit this occasionally.

### 4b. Connect

```bash
ssh -i ~/.ssh/id_ed25519 ubuntu@<PUBLIC_IP>
```

Put it in `~/.ssh/config` so every tool picks it up automatically:

```
Host mlbox
    HostName <PUBLIC_IP>
    User ubuntu
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
```

Then it is just `ssh mlbox`, `rsync ... mlbox:...`, and VS Code Remote-SSH sees it too.

**Alternatives if SSH is awkward:**

- **EC2 Instance Connect** — browser terminal from the console, zero key management. Fine
  for a quick look, useless for rsync or long sessions.
- **SSM Session Manager** — connect with **no inbound port open at all**, authenticated
  through IAM. Needs an IAM instance profile with `AmazonSSMManagedInstanceCore` attached.
  This is the security-correct option and worth it if the box will be long-lived.

### 4c. `tmux` is not optional

Your SSH session *will* drop, and an un-tmux'd job dies with it.

```bash
sudo apt install -y tmux
tmux new -s ml          # start
# ctrl-b then d          # detach, safe to disconnect
tmux attach -t ml       # come back later
```

Run **every** long job inside tmux. This is the single most common way people lose hours.

---

## 5. Getting code and data across

**Code** (small, changes constantly) → **git**. Push from the Mac, pull on the box.

**Data** (2.3 GB raw + ~3 GB of derived parquet) → **S3**, not scp.

```bash
# on the Mac, once
brew install awscli
aws configure                       # access key, secret, region

aws s3 mb s3://<your-unique-bucket>
aws s3 sync "student_resource/dataset" s3://<your-unique-bucket>/dataset
```

```bash
# on the EC2 box
aws s3 sync s3://<your-unique-bucket>/dataset ~/mlchallenge/dataset
```

Why S3 rather than `scp`: it is far faster than a home uplink, it survives instance
termination, and **both boxes read the same bucket** — so the GPU box can pick up the
candidate features the CPU box produced without any machine-to-machine copying. Give each
instance an **IAM role** with S3 access rather than pasting access keys onto the box.

S3 transfer *within the same region* is free; cross-region is not. Keep bucket and
instances in one region.

### Working loop

The smoothest setup is to run the agent **on the box**, so it can execute the pipeline
directly instead of you shuttling output back and forth:

```bash
# on the EC2 box
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
npm install -g @anthropic-ai/claude-code
claude          # authenticate once
```

Then: edit and commit locally if you prefer, `git pull` on the box, and let heavy stages
run there inside tmux.

---

## 6. Cost control (do this on day one)

1. **AWS Budgets** → two alerts, at **$400** and **$550**, emailed to you.
2. **Stop instances** whenever you step away. A forgotten `g5.12xlarge` running overnight
   is a meaningful fraction of the credit.
3. Prefer **spot** for anything re-runnable.
4. Check **Cost Explorer** daily for the first few days until the burn rate is predictable.
5. Delete large S3 intermediates you are not reusing; storage is cheap but not free.

---

## 7. Suggested order of operations

1. Verify the credit covers EC2 ← **blocking; do this first**
2. Request the three quota increases ← **long pole; do it in parallel**
3. Launch the CPU work box, import your SSH key, set up tmux + S3 sync
4. Move Phase A of the pipeline onto it and keep building while GPU quota is pending
5. Launch the GPU box only when Phase A has produced a scored baseline

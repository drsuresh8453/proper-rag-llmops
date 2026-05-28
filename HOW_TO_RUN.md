# HOW TO RUN — Sales RAG Assistant
**Suresh D R | AI Product Developer & Technology Mentor**
**Using Git Bash on Windows**

---

## What Is in Each File

```
proper_rag/
├── backend/
│   └── rag_engine.py
│       → The entire RAG brain
│       → chunk_text()        splits document into 400-word overlapping chunks
│       → embed_texts()       calls OpenAI API → converts text to 1536-dim vectors
│       → index_document()    takes chunks → embeds → stores in ChromaDB
│       → retrieve_chunks()   embeds your question → finds similar chunks in ChromaDB
│       → generate_rag_answer() sends retrieved chunks + question to GPT-3.5 → answer
│       → run_rag_pipeline()  calls all above in order — one function to run full RAG
│
├── frontend/
│   └── streamlit_app.py
│       → The browser chat interface
│       → Sidebar: API key input, file uploader, Load Sample Data button, Index button
│       → Tab 1 Chat: chat history, question input, shows answer + retrieved chunks
│       → Tab 2 Document Preview: word count, full text, chunk count after indexing
│       → Tab 3 How RAG Works: step by step explanation with similarity score info
│
├── data/
│   └── sales_report_q1_2024.txt
│       → Sample Q1 2024 sales report — 2000+ words
│       → Contains: monthly revenue Jan/Feb/Mar, regional breakdown,
│         product line performance, top 10 sales reps with numbers,
│         pipeline by stage, customer metrics, competitor losses, Q2 targets
│       → You can also upload your own TXT file from the app sidebar
│
├── requirements.txt
│       → streamlit   — browser UI framework
│       → openai      — embeddings (text-embedding-3-small) + GPT-3.5
│       → chromadb    — local vector database
│       → python-dotenv — environment variable loader
│
├── .github/
│   └── workflows/
│       └── deploy.yml
│           → CI/CD pipeline — GitHub reads this automatically on every push
│           → 11 steps: checkout → install → test → AWS auth → ECR login
│             → Docker build → push to ECR → kubectl deploy → smoke test
│           → Triggered by: git push origin main
│           → Result: new version live on EKS in 12 minutes automatically
│
├── k8s/
│   └── deployment.yml
│       → Kubernetes deployment file — tells EKS how to run the app
│       → Deployment: 2 pods, ECR image, health checks, resource limits
│       → Service: AWS Load Balancer, public URL on port 8501
│
├── Dockerfile
│       → Multi-stage build — builder stage installs libraries,
│         runtime stage copies only what is needed — smaller image
│       → Runs streamlit on port 8501 inside the container
│       → HEALTHCHECK pings /_stcore/health every 30 seconds
│
└── HOW_TO_RUN.md
        → This file
```

---

## Prerequisites — Install These First

### 1. VS Code

Download: https://code.visualstudio.com/download
Click: **Windows** → download the `.exe` installer
Run the installer → click Next → tick these two options:
```
✅ Add "Open with Code" action to Windows Explorer file context menu
✅ Add "Open with Code" action to Windows Explorer directory context menu
✅ Add to PATH
```
Click: Install → Finish

Open VS Code → you will see the welcome screen.

Useful VS Code extensions to install (optional but recommended):
```
Python          → by Microsoft → for Python syntax highlighting
GitLens         → for Git history
Docker          → for Dockerfile syntax highlighting
```
To install: click Extensions icon on left sidebar → search → Install

---

### 2. Python 3.11

**Do NOT install from python.org if you have winget.**
Use winget — it is the easiest way on Windows.

Open Git Bash and run:
```
winget install Python.Python.3.11
```

What you will see:
```
Found Python 3.11 [Python.Python.3.11] Version 3.11.9
Downloading https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
██████████████████████████████  25.0 MB / 25.0 MB
Successfully installed
```

**Important — close Git Bash completely and open a new one after installing.**

Verify:
```
py -3.11 --version
```
Expected: `Python 3.11.9`

**⚠️ If you already have Python 3.13 or 3.14 installed:**
That is fine — keep it. But always create your venv with `py -3.11` specifically:
```
py -3.11 -m venv venv
```
This forces the venv to use Python 3.11 even if a newer version is the default.

**Why Python 3.11 specifically?**
```
Python 3.11 → ✅ all libraries work perfectly
Python 3.12 → ✅ mostly works
Python 3.13 → ⚠️ some libraries fail
Python 3.14 → ❌ numpy, chromadb fail — do not use for AI projects
```

---

### 3. Git Bash

Usually already installed with Git.
Download Git (includes Git Bash): https://git-scm.com/download/win

---

### 4. Docker Desktop

Download: https://www.docker.com/products/docker-desktop/
Install → restart computer → open Docker Desktop
Wait for the whale icon to stop animating — Docker is ready.

Verify in Git Bash:
```
docker --version
```
Expected: `Docker version 24.x.x`

---

### 5. AWS CLI

Download: https://awscli.amazonaws.com/AWSCLIV2.msi
Run the installer → close Git Bash → open new Git Bash

Verify:
```
aws --version
```
Expected: `aws-cli/2.x.x`

---

### 6. OpenAI API Key

Go to: https://platform.openai.com/api-keys
Click: Create new secret key
Copy it — starts with `sk-`
You will enter this in the app sidebar — no file needed.

---

## PHASE 1 — Run Locally Without Docker

### Step 1 — Open Git Bash in the Project Folder

Right click inside the `proper_rag` folder → Git Bash Here

OR open Git Bash and navigate:
```
cd ~/Desktop/proper_rag
```

---

### Step 2 — Create Virtual Environment

```
python -m venv venv
```

What this does: creates an isolated Python environment in a `venv/` folder.
Your global Python is not affected.

---

### Step 3 — Activate Virtual Environment

**Git Bash (always use this):**
```
source venv/Scripts/activate
```

You will see `(venv)` appear at the start of your prompt:
```
(venv)
yourname@DESKTOP MINGW64 ~/Desktop/proper_rag
$
```

This confirms the virtual environment is active.
Run this every time you open a new Git Bash terminal.

**Common mistake:** Do NOT use `venv\Scripts\activate` — that is for Command Prompt.
In Git Bash always use `source venv/Scripts/activate` with forward slash.

---

### Step 4 — Install Libraries

```
pip install streamlit==1.32.0 openai==2.38.0 httpx==0.27.0 chromadb==0.4.22 python-dotenv==1.0.0
```

What gets installed:
```
streamlit==1.32.0    → the browser chat UI
openai==2.38.0       → embeddings + GPT-3.5 answers
httpx==0.27.0        → HTTP client — must match openai version exactly
chromadb==0.4.22     → local vector database
python-dotenv==1.0.0 → load .env file if you use one
```

⚠️ Always install these exact versions together — openai and httpx must be compatible.
If you get `TypeError: Client.__init__() got an unexpected keyword argument 'proxies'`
it means openai and httpx versions are mismatched. Fix:
```
pip install openai==2.38.0 httpx==0.27.0 --force-reinstall
```

This takes 2-3 minutes on first run.

---

### Step 5 — Run the App

```
streamlit run frontend/streamlit_app.py
```

Expected output in Git Bash:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Browser opens automatically at http://localhost:8501

Stop the app: `Ctrl+C` in Git Bash

---

### Step 6 — Use the App

In the browser:

**Sidebar (left side):**
```
1. Enter your OpenAI API key in the key field (sk-...)
2. Click "Load Q1 Sales Report"  ← loads the sample sales data
   OR upload your own TXT file using the file uploader
3. Click "Index Document into Vector DB"
   → splits text into chunks
   → calls OpenAI to embed each chunk (text-embedding-3-small)
   → stores all 1536-dim vectors in ChromaDB
   → takes 5-15 seconds
   → you see: "Indexed 7 chunks into ChromaDB"
```

**Chat Tab:**
```
4. Type your question in the chat box at the bottom
5. Press Enter
6. See:
   → The answer (grounded in retrieved chunks only)
   → Metrics: chunks retrieved, top similarity score, response time
   → Expand "X chunks retrieved" to see exact chunks used + similarity scores
```

---

### Step 7 — Try These Questions

```
What was the total Q1 2024 revenue?
How did March perform compared to January?
Which sales rep had the highest revenue in Q1?
What is the SaaS revenue for the quarter?
Which region had the highest growth?
How many deals were lost to Salesforce?
What is the pipeline value at end of Q1?
What are the Q2 2024 targets?
Who are the top 3 performing sales reps?
Which city contributed the most revenue in March?
What was the customer retention rate?
What are the key initiatives planned for Q2?
```

---

## PHASE 2 — Run With Docker

### Step 8 — Build Docker Image

Make sure Docker Desktop is running first.

```
docker build -t sales-rag-assistant:v1 .
```

What Docker does step by step:
```
Step 1 — Downloads python:3.11-slim base image
Step 2 — Creates /build directory, copies requirements.txt
Step 3 — Runs pip install — installs all libraries inside container
Step 4 — Starts fresh from python:3.11-slim (runtime stage)
Step 5 — Copies only finished libraries from build stage
Step 6 — Copies backend/, frontend/, data/ folders
Step 7 — Records port 8501 and startup command
Step 8 — Saves as sealed image on your laptop
```

First build: 3-5 minutes
Subsequent builds: much faster (Docker caches unchanged layers)

---

### Step 9 — List Docker Images

```
docker images
```

Expected output:
```
REPOSITORY              TAG    IMAGE ID       CREATED         SIZE
sales-rag-assistant     v1     abc123def456   2 minutes ago   1.2GB
python                  3.11-slim  xyz789   3 days ago      130MB
```

Filter by name:
```
docker images sales-rag-assistant
```

---

### Step 10 — Run the Container

```
docker run -d \
  --name sales-rag \
  -p 8501:8501 \
  -e OPENAI_API_KEY=sk-your-key-here \
  sales-rag-assistant:v1
```

What each flag means:
```
-d                       → detached — runs in background, terminal stays free
--name sales-rag         → give the container a name (easier to refer to later)
-p 8501:8501             → map laptop port 8501 to container port 8501
                           left = your laptop, right = inside container
-e OPENAI_API_KEY=sk-... → pass API key into container as environment variable
                           this is how the key gets in without being in the code
sales-rag-assistant:v1   → which image to run
```

Open browser: http://localhost:8501

In the app — you do NOT need to enter the API key in the sidebar.
It is already inside the container from the `-e` flag.
Just load data, index, and start chatting.

---

### Step 11 — Check Container is Running

```
docker ps
```

Expected:
```
CONTAINER ID   IMAGE                    STATUS         PORTS
abc123def456   sales-rag-assistant:v1   Up 30 seconds  0.0.0.0:8501->8501/tcp
```

---

### Step 12 — View Container Logs

```
docker logs sales-rag
```

Follow live logs:
```
docker logs -f sales-rag
```

Press `Ctrl+C` to stop following (container keeps running)

---

### Step 13 — Stop and Remove Container

```
docker stop sales-rag
docker rm sales-rag
```

Or force stop and remove in one command:
```
docker rm -f sales-rag
```

---

## PHASE 3 — Push to AWS ECR

### Step 14 — Create AWS Account

Go to: https://aws.amazon.com → Create an AWS Account

Fill in:
```
Email address  → your email
Account name   → any name (example: LLMOpsLearning)
Password       → strong password
```

Contact Information:
```
Account type   → Personal
Full name      → your name
Phone number   → Indian mobile with +91
Country        → India
Address        → your address
```

Payment card: AWS charges $1 to verify — refunded immediately.

Identity verification: enter OTP sent to your mobile.

Support plan: select Basic — Free.

Check email and click verification link.

---

### Step 15 — Login to AWS Console

Go to: https://console.aws.amazon.com
```
Select: Root user
Email: the email you used to sign up
Password: your password
```

**Set region — important:**
Top right corner → click region name → select **Asia Pacific (Mumbai) ap-south-1**

**Find your Account ID:**
Click your name top right → Account ID is shown (12 digits)
Example: `123456789012`
Save this — you need it for ECR commands.

---

### Step 16 — Create IAM User

AWS Console → search `IAM` → Users → Create user
```
Username: llmops-rag-user
Next → Attach policies directly → search and select:
  ✅ AmazonECR-FullAccess
Next → Create user
```

Get access keys:
```
Click the user → Security credentials tab
→ Create access key
→ Select: Application running outside AWS
→ Download .csv file — SAVE IT SAFELY
  You cannot see the secret key again after closing this screen
```

Your keys look like:
```
Access Key ID:     AKIAXXXXXXXXXXXXXXXX
Secret Access Key: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

### Step 17 — Configure AWS CLI

In Git Bash:
```
aws configure
```

Enter when prompted:
```
AWS Access Key ID:     → paste from csv
AWS Secret Access Key: → paste from csv
Default region name:   → ap-south-1
Default output format: → json
```

Test connection:
```
aws sts get-caller-identity
```

Expected output:
```
{
    "UserId": "AIDAXXXXXXXXXXXXXXXX",
    "Account": "123456789012",
    "Arn": "arn:aws:iam::123456789012:user/llmops-rag-user"
}
```

---

### Step 18 — Create ECR Repository

```
aws ecr create-repository \
  --repository-name sales-rag-assistant \
  --region ap-south-1
```

From the output copy the `repositoryUri`:
```
123456789012.dkr.ecr.ap-south-1.amazonaws.com/sales-rag-assistant
```

Replace `123456789012` with YOUR actual account ID in all commands below.

---

### Step 19 — Authenticate Docker with ECR

Run this before every push (expires every 12 hours):
```
aws ecr get-login-password --region ap-south-1 | \
  docker login --username AWS --password-stdin \
  123456789012.dkr.ecr.ap-south-1.amazonaws.com
```

What each part does:
```
aws ecr get-login-password   → gets a temporary password from AWS (valid 12 hours)
|                            → pipe — sends that password to the next command
docker login                 → logs Docker into ECR registry
--username AWS               → always "AWS" — not your IAM username
--password-stdin             → reads password from the pipe
123456789012.dkr.ecr...      → your ECR registry URL (account ID + region)
```

Expected: `Login Succeeded`

---

### Step 20 — Tag Image for ECR

```
docker tag sales-rag-assistant:v1 \
  123456789012.dkr.ecr.ap-south-1.amazonaws.com/sales-rag-assistant:v1
```

What this does:
Creates a new name for the same image — pointing to your ECR registry.
The image is not duplicated — both tags point to the same data.

Verify both tags exist:
```
docker images sales-rag-assistant
```

---

### Step 21 — Push to ECR

```
docker push \
  123456789012.dkr.ecr.ap-south-1.amazonaws.com/sales-rag-assistant:v1
```

What you see:
```
The push refers to repository [123456789012.dkr.ecr.ap-south-1.amazonaws.com/sales-rag-assistant]
abc123: Pushing [=============>     ] 128MB/453MB
def456: Pushed
v1: digest: sha256:abc123... size: 2345
```

First push: several minutes (all layers upload)
Second push (code change only): much faster (only changed layer uploads)

---

### Step 22 — Verify in ECR

Via CLI:
```
aws ecr list-images \
  --repository-name sales-rag-assistant \
  --region ap-south-1
```

Expected:
```
{
    "imageIds": [
        {
            "imageDigest": "sha256:abc123...",
            "imageTag": "v1"
        }
    ]
}
```

Via AWS Console:
```
AWS Console → search ECR → Elastic Container Registry
→ Repositories → sales-rag-assistant
→ see your image with tag v1, size, and push date
```

---

## Quick Reference — All Commands

```
# ── Virtual Environment ───────────────────────────────────────
python -m venv venv
source venv/Scripts/activate        # Git Bash — always use this
pip install -r requirements.txt

# ── Run App ───────────────────────────────────────────────────
streamlit run frontend/streamlit_app.py

# ── Docker Build ──────────────────────────────────────────────
docker build -t sales-rag-assistant:v1 .

# ── Docker List Images ────────────────────────────────────────
docker images
docker images sales-rag-assistant

# ── Docker Run ────────────────────────────────────────────────
docker run -d \
  --name sales-rag \
  -p 8501:8501 \
  -e OPENAI_API_KEY=sk-your-key \
  sales-rag-assistant:v1

# ── Docker Inspect ────────────────────────────────────────────
docker ps                           # running containers
docker ps -a                        # all containers including stopped
docker logs sales-rag               # view logs
docker logs -f sales-rag            # follow live logs
docker stats                        # live CPU and memory usage

# ── Docker Stop and Remove ────────────────────────────────────
docker stop sales-rag
docker rm sales-rag
docker rm -f sales-rag              # force stop and remove
docker rmi sales-rag-assistant:v1   # delete image

# ── Docker Cleanup ────────────────────────────────────────────
docker system prune                 # remove unused containers and images

# ── AWS CLI ───────────────────────────────────────────────────
aws configure
aws sts get-caller-identity

# ── ECR ───────────────────────────────────────────────────────
aws ecr create-repository \
  --repository-name sales-rag-assistant --region ap-south-1

aws ecr get-login-password --region ap-south-1 | \
  docker login --username AWS --password-stdin \
  YOUR_ACCOUNT_ID.dkr.ecr.ap-south-1.amazonaws.com

docker tag sales-rag-assistant:v1 \
  YOUR_ACCOUNT_ID.dkr.ecr.ap-south-1.amazonaws.com/sales-rag-assistant:v1

docker push \
  YOUR_ACCOUNT_ID.dkr.ecr.ap-south-1.amazonaws.com/sales-rag-assistant:v1

aws ecr list-images \
  --repository-name sales-rag-assistant --region ap-south-1
```

---

## PHASE 4 — Deploy to Kubernetes (EKS) on AWS

### What is Kubernetes — Simple Terms

Kubernetes runs your Docker container at scale — multiple copies, auto-scaling, zero downtime, and auto-restart if anything crashes.

**The simple analogy:**
Running Docker alone = one McDonald's outlet. Fine for testing. When 1000 customers come at once — it crashes.
Kubernetes = a chain of outlets managed automatically. Opens more when busy, closes extras when quiet, replaces any outlet that burns down. You just set the rules.

**Why you need it for the RAG app:**
```
Without Kubernetes:
  1 container → 10 users at once → server overloads → crashes
  Container crashes at 3am → app is down till morning
  New version deployed → app is briefly unavailable

With Kubernetes (EKS):
  3 containers → 500 users handled simultaneously
  Container crashes → replaced in 15 seconds automatically
  New version → rolling update → zero downtime
```

---

### Step 1 — Install eksctl and kubectl

**Install eksctl** (tool to create EKS cluster):
```
winget install eksctl
```

**Install kubectl** (tool to manage Kubernetes):
```
winget install Kubernetes.kubectl
```

Close Git Bash and open a new one. Verify:
```
eksctl version
kubectl version --client
```

---

### Step 2 — Create EKS Cluster

⚠️ **Costs Rs 600-800 per day. Delete after practice.**

```
eksctl create cluster \
  --name rag-cluster \
  --region ap-south-1 \
  --nodegroup-name workers \
  --node-type t3.medium \
  --nodes 2 \
  --nodes-min 1 \
  --nodes-max 5 \
  --managed
```

What each flag means:
```
--name rag-cluster       → name of your cluster
--region ap-south-1      → Mumbai — lowest latency for India
--node-type t3.medium    → each server has 2 CPU, 4GB RAM
--nodes 2                → start with 2 servers
--nodes-min 1            → minimum 1 server always running
--nodes-max 5            → scale up to 5 servers if needed
--managed                → AWS manages the servers for you
```

**Wait 15-20 minutes.** You will see:
```
✅  EKS cluster "rag-cluster" in "ap-south-1" region is ready
```

---

### Step 3 — Connect kubectl to Your Cluster

```
aws eks update-kubeconfig \
  --region ap-south-1 \
  --name rag-cluster
```

Verify nodes are ready:
```
kubectl get nodes
```

Expected:
```
NAME                                        STATUS   AGE
ip-192-168-1-10.ap-south-1.compute.internal Ready    5m
ip-192-168-1-11.ap-south-1.compute.internal Ready    5m
```

Two nodes — Ready ✅

---

### Step 4 — Store OpenAI API Key as Kubernetes Secret

Never put your API key in the YAML file. Store it as a Kubernetes Secret:

```
kubectl create secret generic rag-secrets \
  --from-literal=openai-api-key=your-openai-key-here
```

Verify:
```
kubectl get secrets
```

---

### Step 5 — Update deployment.yml with Your Account ID

Open `k8s/deployment.yml` in VS Code.

Find this line:
```
image: YOUR_ACCOUNT_ID.dkr.ecr.ap-south-1.amazonaws.com/sales-rag-assistant:v1
```

Replace `YOUR_ACCOUNT_ID` with your 12-digit AWS account ID (example: `968603941077`):
```
image: 968603941077.dkr.ecr.ap-south-1.amazonaws.com/sales-rag-assistant:v1
```

Save the file.

---

### Step 6 — Make Sure Docker Image Is in ECR

If not already pushed — push it now:
```
aws ecr get-login-password --region ap-south-1 | \
  docker login --username AWS --password-stdin \
  968603941077.dkr.ecr.ap-south-1.amazonaws.com

docker tag sales-rag-assistant:v1 \
  968603941077.dkr.ecr.ap-south-1.amazonaws.com/sales-rag-assistant:v1

docker push \
  968603941077.dkr.ecr.ap-south-1.amazonaws.com/sales-rag-assistant:v1
```

Verify:
```
aws ecr list-images \
  --repository-name sales-rag-assistant \
  --region ap-south-1
```

---

### Step 7 — Deploy to Kubernetes

```
kubectl apply -f k8s/deployment.yml
```

What Kubernetes does:
```
Reads deployment.yml
Pulls image from ECR → 968603941077.dkr.ecr.../sales-rag-assistant:v1
Starts 2 pods across your 2 nodes
Creates AWS Load Balancer
Assigns public URL
```

Check pods are starting:
```
kubectl get pods
```

Expected (wait 1-2 minutes for Running):
```
NAME                             READY   STATUS    RESTARTS   AGE
sales-rag-app-7d9f8b-abc12       1/1     Running   0          2m
sales-rag-app-7d9f8b-def34       1/1     Running   0          2m
```

---

### Step 8 — Get Public URL

```
kubectl get service sales-rag-service
```

Expected output:
```
NAME               TYPE           CLUSTER-IP     EXTERNAL-IP                              PORT(S)
sales-rag-service  LoadBalancer   10.100.1.100   abc123.ap-south-1.elb.amazonaws.com      8501:30001/TCP
```

Your app is live at:
```
http://abc123.ap-south-1.elb.amazonaws.com:8501
```

Wait 2-3 minutes for the Load Balancer to fully provision before opening.

---

### Step 9 — Check Logs

```
kubectl logs sales-rag-app-7d9f8b-abc12
```

Follow live:
```
kubectl logs -f sales-rag-app-7d9f8b-abc12
```

---

### Step 10 — Deploy a New Version

When you update code and rebuild Docker image:

1. Build and push new image with v2 tag:
```
docker build -t sales-rag-assistant:v2 .
docker tag sales-rag-assistant:v2 \
  968603941077.dkr.ecr.ap-south-1.amazonaws.com/sales-rag-assistant:v2
docker push \
  968603941077.dkr.ecr.ap-south-1.amazonaws.com/sales-rag-assistant:v2
```

2. Update image tag in k8s/deployment.yml:
```yaml
image: 968603941077.dkr.ecr.ap-south-1.amazonaws.com/sales-rag-assistant:v2
```

3. Apply update — zero downtime rolling update:
```
kubectl apply -f k8s/deployment.yml
```

Watch it happen:
```
kubectl rollout status deployment/sales-rag-app
```

---

### Step 11 — Rollback if Something Goes Wrong

```
kubectl rollout undo deployment/sales-rag-app
```

Instantly goes back to previous version.

---

### Step 12 — Delete Cluster When Done (Save Cost)

⚠️ **Always delete when done practising — Rs 600-800/day charge.**

```
eksctl delete cluster \
  --name rag-cluster \
  --region ap-south-1
```

Wait 5-10 minutes for complete deletion.

---

### Kubernetes Quick Reference Commands

```
# ── Pods ──────────────────────────────────────────────────────
kubectl get pods                           # list all pods
kubectl get pods -o wide                   # pods with IP and node info
kubectl describe pod pod-name              # full details
kubectl logs pod-name                      # pod logs
kubectl logs -f pod-name                   # follow live logs
kubectl exec -it pod-name -- bash          # terminal inside pod

# ── Deployment ────────────────────────────────────────────────
kubectl get deployments                    # list deployments
kubectl apply -f k8s/deployment.yml        # create or update
kubectl rollout status deployment/sales-rag-app  # watch update
kubectl rollout undo deployment/sales-rag-app    # rollback
kubectl scale deployment sales-rag-app --replicas=5  # manual scale

# ── Service ───────────────────────────────────────────────────
kubectl get services                       # list services + external URL
kubectl describe service sales-rag-service # full details

# ── Secrets ───────────────────────────────────────────────────
kubectl get secrets                        # list secrets

# ── Nodes ─────────────────────────────────────────────────────
kubectl get nodes                          # list cluster nodes

# ── Cleanup ───────────────────────────────────────────────────
kubectl delete -f k8s/deployment.yml       # delete app from cluster
eksctl delete cluster --name rag-cluster --region ap-south-1
```

---

## PHASE 5 — GitHub Setup and CI/CD Pipeline

### What This Phase Does

After this phase — every time you make a change locally and run `git push`, the entire pipeline runs automatically:

```
You change code locally
        ↓
git push to GitHub
        ↓
GitHub Actions starts automatically (reads .github/workflows/deploy.yml)
        ↓
Tests run → Docker builds → pushes to ECR → deploys to EKS
        ↓
New version live in 12 minutes
Zero manual steps
```

---

### What Is in Each New File

```
proper_rag/
└── .github/
    └── workflows/
        └── deploy.yml    ← GitHub Actions CI/CD pipeline
                            GitHub reads this file automatically on every push
                            Contains 11 steps: test → build → push → deploy
```

---

### Step 1 — Create GitHub Account

Go to: https://github.com
Click: Sign up
Fill in: username, email, password
Verify email.

---

### Step 2 — Create a New Repository

After logging in:
```
Click: + (top right corner) → New repository
Repository name: proper-rag-llmops
Visibility: Private
Click: Create repository
```

---

### Step 3 — Push Your Code to GitHub

In Git Bash inside your `proper_rag` folder:

**First time setup:**
```
git init
git add .
git commit -m "Initial commit — Sales RAG Assistant with Docker and Kubernetes"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/proper-rag-llmops.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

Verify — go to https://github.com/YOUR_USERNAME/proper-rag-llmops
You should see all your files there.

---

### Step 4 — Add GitHub Secrets

GitHub Actions needs your AWS keys and credentials stored as Secrets — never in code.

Go to your repository on GitHub:
```
Settings → Secrets and variables → Actions → New repository secret
```

Add all 5 secrets one by one:

| Secret Name | Value |
|---|---|
| AWS_ACCESS_KEY_ID | Your AWS Access Key ID |
| AWS_SECRET_ACCESS_KEY | Your AWS Secret Access Key |
| AWS_REGION | ap-south-1 |
| ECR_REPOSITORY | sales-rag-assistant |
| EKS_CLUSTER_NAME | rag-cluster |

⚠️ These are stored encrypted in GitHub — never visible in logs or code.

---

### Step 5 — Verify GitHub Actions File Is Present

Check your project has the CI/CD file:
```
proper_rag/
└── .github/
    └── workflows/
        └── deploy.yml   ← must be here
```

If it is there — GitHub will find it automatically on every push. No configuration needed.

---

### Step 6 — Trigger the First Deployment

Make a small change to trigger the pipeline:
```
# Make any small change — example: add a comment in rag_engine.py
echo "# CI/CD test" >> backend/rag_engine.py

git add .
git commit -m "Test CI/CD pipeline — first automated deployment"
git push origin main
```

---

### Step 7 — Watch the Pipeline Run

Go to your GitHub repository:
```
Click: Actions tab (top menu)
```

You will see the pipeline running. Click on it to see live logs for each step:

```
✅ Checkout code
✅ Set up Python 3.11
✅ Install dependencies
✅ Run tests — ALL TESTS PASSED
✅ Configure AWS credentials
✅ Login to Amazon ECR
✅ Build and push Docker image
✅ Update kubeconfig for EKS
✅ Deploy to EKS
✅ Smoke test
✅ Deployment summary
```

If any step fails — click on it to see exactly what went wrong.

---

### Step 8 — Verify New Version Is Live

After pipeline completes:
```
kubectl get pods
```

You should see new pods with a newer AGE — Kubernetes did a rolling update.

```
kubectl get service sales-rag-service
```

Open the External IP URL — your updated app is live.

---

### Step 9 — Make a Real Change and See It Go Live Automatically

Now try a real meaningful change. For example — update the system prompt in `frontend/streamlit_app.py`:

Open `frontend/streamlit_app.py` and find the system prompt section. Change it — add a new instruction. Save the file.

Then:
```
git add frontend/streamlit_app.py
git commit -m "Update system prompt — add bullet point formatting instruction"
git push origin main
```

Go to GitHub → Actions tab. Watch the pipeline run automatically.

In 12 minutes — your updated prompt is live in production on EKS.
No Docker command. No ECR push. No kubectl. Nothing manual.

---

### Step 10 — What Happens Inside the Pipeline (Summary)

```
STEP 1  Checkout code
        GitHub downloads your pushed code onto a fresh Ubuntu VM

STEP 2  Set up Python 3.11
        Installs Python on the VM

STEP 3  Install dependencies
        pip install all libraries with exact pinned versions

STEP 4  Run tests
        Checks imports, chunking, ChromaDB, syntax
        If anything fails → pipeline STOPS. Nothing is deployed.
        You get a notification. Your change never reaches production.

STEP 5  Configure AWS credentials
        Reads AWS keys from GitHub Secrets securely

STEP 6  Login to Amazon ECR
        Authenticates Docker with your ECR registry

STEP 7  Build and push Docker image
        Builds fresh Docker image from your new code
        Tags it with the git commit SHA (unique per deployment)
        Pushes to ECR

STEP 8  Update kubeconfig
        Connects kubectl on the VM to your EKS cluster

STEP 9  Deploy to EKS
        Updates deployment.yml with new image tag
        kubectl apply → rolling update starts
        Waits up to 5 minutes for rollout to complete

STEP 10 Smoke test
        Hits the health endpoint of the live app
        Confirms app is responding

STEP 11 Summary
        Prints commit ID, who pushed, image URI, pod status
```

---

### Step 11 — Rollback if Something Goes Wrong

If the new version has a problem — roll back instantly:

```
git revert HEAD
git push origin main
```

GitHub Actions runs again automatically — deploys the reverted version in 12 minutes.

Or roll back directly with kubectl:
```
kubectl rollout undo deployment/sales-rag-app
```

---

### Git Commands You Will Use Daily

```
# Check what changed locally
git status

# See exactly what lines changed
git diff

# Stage and push changes
git add .
git commit -m "describe what you changed"
git push origin main

# Pull latest from GitHub (if team made changes)
git pull origin main

# See commit history
git log --oneline

# Undo last commit (keep changes locally)
git reset HEAD~1

# Undo last commit and reverse the change
git revert HEAD
git push origin main
```

---

### The Complete Flow — Everything Together

```
LOCAL DEVELOPMENT
  You make changes on your laptop
  Test locally: streamlit run frontend/streamlit_app.py
  Satisfied with changes
          ↓
GIT PUSH
  git add . && git commit -m "message" && git push origin main
          ↓
GITHUB ACTIONS (automatic — you do nothing)
  Tests run
  Docker image built with your new code
  Image pushed to ECR with unique commit ID tag
  kubectl applies new deployment to EKS
  Rolling update — zero downtime
  Smoke test confirms app is healthy
          ↓
PRODUCTION (EKS)
  New pods running your latest code
  Old pods gracefully terminated
  Load balancer distributes traffic to new pods
  Public URL unchanged — users see updated app
          ↓
YOU
  Check GitHub Actions tab — green ticks ✅
  Open app URL — changes are live
  Total time from git push to live: 12 minutes
  Manual steps: ZERO
```

---

## Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `bash: venvScriptsactivate: command not found` | Using wrong activate command in Git Bash | Use `source venv/Scripts/activate` |
| `ModuleNotFoundError` | Virtual env not active | Run `source venv/Scripts/activate` first |
| `streamlit: command not found` | Libraries not installed | Run `pip install -r requirements.txt` |
| `Login Succeeded` not appearing | ECR auth expired | Re-run the `get-login-password` command |
| `docker: command not found` | Docker Desktop not running | Open Docker Desktop and wait for whale icon |
| `Error: No such container` | Container name wrong | Run `docker ps -a` to see actual names |
| `TypeError: Client.__init__() got an unexpected keyword argument 'proxies'` | openai version too old for installed httpx | Run `pip install --upgrade openai httpx` |

---

### Fix for OpenAI + httpx Version Mismatch

If you see this error when running the app:
```
TypeError: Client.__init__() got an unexpected keyword argument 'proxies'
```

This means your openai library version is incompatible with the httpx version installed.

**Fix — run this in Git Bash:**
```
pip install --upgrade openai httpx
```

Then run the app again:
```
streamlit run frontend/streamlit_app.py
```

If the above still fails, pin exact versions:
```
pip install openai==2.38.0 httpx==0.27.0
```

**Why this happens:**
openai 1.12.0 uses an old httpx API that newer httpx versions removed.
Upgrading both together fixes the version mismatch.
Your requirements.txt already has `openai>=2.0.0` to prevent this on fresh installs.

---

**Suresh D R | AI Product Developer & Technology Mentor**

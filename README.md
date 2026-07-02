# Isaac Lab: Docker & Dev Workflow

**References:**
* [Deployment Docker Documentation](https://isaac-sim.github.io/IsaacLab/main/source/deployment/docker.html#deployment-docker)
* [Run Docker Example](https://isaac-sim.github.io/IsaacLab/main/source/deployment/run_docker_example.html)

---

## 1. Host Machine: Start Container

Run these commands from your local machine to authorize display and start the Isaac Sim image:

```bash
cd /home/quinn/IsaacLab
conda activate lab

# Authorize screen sharing
xhost +local:docker

# Start and enter the container
python docker/container.py start
python docker/container.py enter
```

*(To stop later: `python docker/container.py stop` — ensure Git changes are saved!)*

## 2. Inside Container: Environment Setup

Once inside the container, run this block to update dependencies, configure Git, and install your project:

```bash
# Update and install Git LFS
apt update && apt upgrade -y
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | 
apt-get install git-lfs && git lfs install

# Configure Git
git config --global user.email "cooperstein.quinn@gmail.com"
git config --global user.name "Quinn135"

# Clone and install the 'chicken' repository
mkdir -p /workspace/chicken && cd /workspace/chicken
git clone https://github.com/Quinn135/chicken.git .
python -m pip install -e source/chicken
```

## 3. VS Code Integration

1. Open the Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`).
2. Select: **`Dev Containers: Attach to Running Container...`**
3. Run the Python setup task: **`Tasks: Run Task`** -> `setup_python_env`.

## 4. Training & Running Tasks

Navigate to the working directory before running training scripts:

```bash
cd /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/ai3
```

**Training / Playing:**

```bash
# Play/Evaluate
python scripts/skrl/play.py --task=Isaac-Chicken-Robot-v0 --num_envs 4

# Train
python scripts/skrl/train.py --task=Isaac-Chicken-Robot-v0

# Resume Training
python train.py --task=Isaac-Ai3-Direct-v0 --checkpoint /absolute/dir/to/###.pt
```

**Monitoring:**

```bash
tensorboard --logdir logs/skrl/chicken --bind_all
```

## 5. Helpful Utilities

**Copy files from container to host:**

```bash
docker cp isaac-lab-base:/workspace/isaaclab/____ .
```

**Open Isaac Sim UI:**

```bash
isaaclab -s
```

---

## 6. Appendix: Optional & Boilerplate Notes

<details>
<summary>Click to expand manual Docker run command and template basics</summary>

<br>

**Manual Docker Run (Alternative to `container.py`)**

```bash
xhost +local:docker
docker run --name isaac-sim --entrypoint bash -it --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
    -e "PRIVACY_CONSENT=Y" \
    -v $HOME/.Xauthority:/isaac-sim/.Xauthority \
    -e DISPLAY \
    -v ~/docker/isaac-sim/cache/main:/isaac-sim/.cache:rw \
    -v ~/docker/isaac-sim/cache/computecache:/isaac-sim/.nv/ComputeCache:rw \
    -v ~/docker/isaac-sim/logs:/isaac-sim/.nvidia-omniverse/logs:rw \
    -v ~/docker/isaac-sim/config:/isaac-sim/.nvidia-omniverse/config:rw \
    -v ~/docker/isaac-sim/data:/isaac-sim/.local/share/ov/data:rw \
    -v ~/docker/isaac-sim/pkg:/isaac-sim/.local/share/ov/pkg:rw \
    -v ~/Documents/isaac_projects:/isaac-sim/my_projects:rw \
    -u 1234:1234 \
    nvcr.io/nvidia/isaac-sim:5.1.0

# Run inside:
./runapp.sh
```

**Template Pylance Troubleshooting:**

If VS Code runs out of memory, exclude Omniverse packages in `.vscode/settings.json`:

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/chicken"
    ]
}
```

</details>

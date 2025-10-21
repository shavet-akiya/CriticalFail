# CriticalFail
DECO3801
## GPU Acceleration (So Much Faster)

If you have an **NVIDIA GPU** (e.g., RTX 2060, 3060, 4060+), you can enable GPU acceleration for **much faster transcription**:

- **With GPU:** 3-hour audio = ~5 minutes
- **Without GPU (CPU):** 3-hour audio = ~45-60 minutes

### Windows with WSL2

#### 1. Install Ubuntu WSL
```powershell
# In PowerShell (Administrator)
wsl --install -d Ubuntu-22.04
```
Create a username and password when prompted.

#### 2. Verify GPU in WSL
Open Ubuntu terminal (from Start menu) and run:
```bash
nvidia-smi
```
You should see your GPU listed. If not, update your NVIDIA drivers on Windows.

#### 3. Install NVIDIA Container Toolkit
In Ubuntu terminal:
```bash
# Add repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
   && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker Desktop (from Windows)
```

#### 4. Fix Docker Permissions
```bash
sudo usermod -aG docker $USER
```
Close and reopen Ubuntu terminal, then:
```bash
sudo chmod 666 /var/run/docker.sock
```

#### 5. Verify GPU in Docker
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```
You should see your GPU info.

#### 6. Navigate and Launch
```bash
cd /mnt/c/Users/YourUsername/path/to/CriticalFail
docker-compose up --build
```

**Look for this in logs:**
```
🚀 GPU DETECTED AND ENABLED!
   Device: NVIDIA GeForce RTX 3060
   Memory: 12.00 GB
```

---
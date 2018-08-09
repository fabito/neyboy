#!/bin/bash

set -e

echo "Installing docker"    
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
apt-get update
apt-cache policy docker-ce
apt-get install -y docker-ce make 

echo "Checking for CUDA and installing."
# Check for CUDA and try to install.
if ! dpkg-query -W cuda-9-0; then
  # The 16.04 installer works with 16.10.
  curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
  # apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
  wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub | sudo apt-key add -
  dpkg -i ./cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
  apt-get update
  apt-get install cuda -y
fi

# Enable persistence mode
nvidia-smi -pm 1
nvidia-smi --auto-boost-default=DISABLED

echo "Installing nvidia docker"
# Add the package repositories
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu16.04/amd64/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
apt-get update
# Install nvidia-docker2 and reload the Docker daemon configuration
apt-get install -y nvidia-docker2

ln -s /usr/lib/google-cloud-sdk/bin/docker-credential-gcloud /usr/bin/

echo "DefaultTasksMax=infinity" |  sudo tee --append /etc/systemd/system.conf
echo "DefaultLimitNOFILE=10000000" | sudo tee --append /etc/systemd/system.conf
echo "UserTasksMax=infinity" | sudo tee --append /etc/systemd/logind.conf
echo "* soft     nproc          unlimited" | sudo tee --append /etc/security/limits.conf
echo "* hard     nproc          unlimited" | sudo tee --append /etc/security/limits.conf
echo "* soft     nofile         unlimited" | sudo tee --append /etc/security/limits.conf
echo "* hard     nofile         unlimited" | sudo tee --append /etc/security/limits.conf
echo "root soft     nofile         unlimited" | sudo tee --append /etc/security/limits.conf
echo "root hard     nofile         unlimited" | sudo tee --append /etc/security/limits.conf

sudo git clone --single-branch -b gh-pages https://github.com/fabito/neyboy.git /opt/neyboy-gh-pages
sudo docker run -p 8000:80 --name neyboy --restart=always -v /opt/neyboy-gh-pages/:/usr/share/nginx/html:ro -d nginx:alpine
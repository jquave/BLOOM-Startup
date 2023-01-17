#!/usr/bin/env bash
# This assumes pytorch/pytorch docker container
pip install transformers accelerate
sudo apt install vim -y
#git clone https://github.com/jquave/BLOOM-Startup.git
python main.py
#!/usr/bin/env bash
# This assumes pytorch/pytorch docker container
pip install transformers
git clone https://github.com/jquave/BLOOM-Startup.git
python main.py
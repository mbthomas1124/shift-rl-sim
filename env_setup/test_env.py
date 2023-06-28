import argparse
import os
import pprint
import threading
import numpy
import torch
import gym
import gymnasium
import tianshou
import time
import shift

assert shift.__version__[:5] == "2.0.1"
assert torch.__version__[:6] == "1.13.1"
assert gym.__version__[:6] == "0.26.2"
assert gymnasium.__version__[:6] == "0.28.1"
assert tianshou.__version__[:5] == "0.5.1"

print("ENVIRONMENT VERIFIED")

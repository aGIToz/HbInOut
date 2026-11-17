# Portions of this file are adapted from GRAND
# (https://github.com/twitter-research/graph-neural-pde), Apache-2.0.
# Modified by Amitoz Azad, 2026.

from block import ODEBlock
from function_run import ODEFuncRun

def set_block(opt):
  ode_str = opt['block']
  return ODEBlock

def set_function(opt):
  ode_str = opt['function']
  return ODEFuncRun

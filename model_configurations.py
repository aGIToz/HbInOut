from block import ODEBlock
from function_run import ODEFuncRun

def set_block(opt):
  ode_str = opt['block']
  return ODEBlock

def set_function(opt):
  ode_str = opt['function']
  return ODEFuncRun

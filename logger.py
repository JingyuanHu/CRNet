from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import os
import time
import sys
import torch
USE_TENSORBOARD = True
try:
  import tensorboardX
  print('Using tensorboardX')
except:
  USE_TENSORBOARD = False

class Logger(object):
  def __init__(self, opt, save_dir):
    """Create a summary writer logging to log_dir."""
    os.mkdir(os.path.join(save_dir, 'log'))
    log_dir = os.path.join(save_dir, 'log')

    args = dict((name, getattr(opt, name)) for name in dir(opt) if not name.startswith('_'))
    file_name = os.path.join(log_dir, 'opt.txt')
    
    with open(file_name, 'wt') as opt_file:
      opt_file.write('==> torch version: {}\n'.format(torch.__version__))
      opt_file.write('==> cudnn version: {}\n'.format(
        torch.backends.cudnn.version()))
      opt_file.write('==> Cmd:\n')
      opt_file.write(str(sys.argv))
      opt_file.write('\n==> Opt:\n')
      for k, v in sorted(args.items()):
        opt_file.write('  %s: %s\n' % (str(k), str(v)))

    
    if USE_TENSORBOARD:
      self.writer = tensorboardX.SummaryWriter(log_dir=log_dir)

    self.log_dir = log_dir
    self.log = open(log_dir + '/log.txt', 'w')

  def write(self, txt):
    self.log.write(txt)
    if '\n' in txt:
      self.log.flush()

  def close(self):
    self.log.close()

  def scalar_summary(self, tag, value, step):
    """Log a scalar variable."""
    if USE_TENSORBOARD:
      self.writer.add_scalar(tag, value, step)

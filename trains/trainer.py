import time
import torch
from progress.bar import Bar
from utils import utils as utils

class Trainer(object):
    def __init__(self, opt, model, loss, optimizer, device):
        self.opt = opt
        self.model = model
        self.loss_state = ['loss', 'hm_loss', 'wh_loss', 'reg_loss']
        self.loss = loss
        self.optimizer = optimizer
        self.device = device
    
        if len(opt.gpus) > 1:
            self.model = torch.nn.DataParallel(self.model, opt.gpus).to(self.device)
        else:
            self.model = self.model.to(self.device)

        
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)
    
    def run_epoch(self, phase, epoch, data_loader):
        model = self.model
        if phase == 'train':
            model.train()
        else:
            if len(self.opt.gpus) > 1:
                model = self.model.module
            model.eval()
            torch.cuda.empty_cache()

        opt = self.opt
        data_time, batch_time = utils.AverageMeter(), utils.AverageMeter()
        avg_loss_state = {l: utils.AverageMeter() for l in self.loss_state}
        num_iters = len(data_loader)
        bar = Bar(max=num_iters)
        end = time.time()
        for iter_id, batch in enumerate(data_loader):
            data_time.update(time.time() - end)
            batch['input'] = batch['input'].to(device=self.device, non_blocking=True)
            for k in batch:
                if k != 'meta' and k != 'input':
                    for tmp_k in batch[k]:
                        batch[k][tmp_k] = batch[k][tmp_k].to(device=self.device, non_blocking=True)
            output = model(batch['input'])
            loss, loss_state = self.loss(output, batch)
            loss = loss.mean()
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()
            
            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(epoch, iter_id, num_iters, phase=phase,total=bar.elapsed_td, eta=bar.eta_td)

            for l in avg_loss_state:
                avg_loss_state[l].update(loss_state[l].mean().item(), opt.batch_size * len(opt.gpus))
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_state[l].avg)
            Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) |Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
            bar.next()
            del output, loss, loss_state
     
        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_state.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        return ret       

    
    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)

    def val(self, epoch, data_loader):
        return self.run_epoch('val', epoch, data_loader)
    
    def eval(self, data_loader, model_path=None):
        if model_path == None:
            model = self.model
        else:
            checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
            state_dict_ = checkpoint['state_dict']
            state_dict = {}
            for k in state_dict_:
                if k.startswith('module') and not k.startswith('module_list'):
                    state_dict[k[7:]] = state_dict_[k] 
                else:
                    state_dict[k] = state_dict_[k]
            model_state_dict = model.state_dict()

            # check loaded parameters and created model parameters
            for k in state_dict:
                if k in model_state_dict:
                    if state_dict[k].shape != model_state_dict[k].shape:
                        print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(k, model_state_dict[k].shape, state_dict[k].shape))
                    else:
                        state_dict[k] = model_state_dict[k]
                else:
                    print('Drop parameter {}.'.format(k))
            
            for k in model_state_dict:
                if not (k in state_dict):
                    print('No param {}.'.format(k))
                else:
                    state_dict[k] = model_state_dict[k]
            model.load_state_dict(state_dict, strict=False)                
            
        if len(self.opt.gpus) > 1:
            model = model.module
        model.eval()
        torch.cuda.empty_cache()   
        for iter_id, batch in enumerate(data_loader): 
            batch['input'] = batch['input'].to(device=self.device, non_blocking=True)    
            output, _, _ = model(batch)


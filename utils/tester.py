import torch
from .statics import AverageMeter, evaluator
import time
from . import logger
import os
def test(device, model, pretrain, data_loader, print_freq=20):
    if pretrain is not None:
        assert os.path.isfile(pretrain)
        logger.info(f'=> loading checkpoint {pretrain}')
        checkpoint = torch.load(pretrain)
        model.load_state_dict(checkpoint['state_dict'])
        logger.info(f'=> successfully loaded checkpoint {pretrain}\n')
    
    model.eval()
    with torch.no_grad():
        n = len(data_loader)
        iter_data_list = [iter(data_loader[_]) for _ in range(n)]
        iter_nmse_list = [AverageMeter(f'Iter nmse_{_}') for _ in range(n)]
        iter_time = AverageMeter('Iter time')
        time_tmp = time.time()
        for batch_idx in range(len(data_loader[0])):
            for nn in range(n):
                sparse_gt = next(iter_data_list[nn])[0].to(device)
                sparse_pred = model(sparse_gt)
                nmse = evaluator(sparse_pred, sparse_gt)
                iter_nmse_list[nn].update(nmse)  
        
            iter_time.update(time.time() - time_tmp)
            time_tmp = time.time()
            # plot progress
            if (batch_idx + 1) % print_freq == 0:
                logger.info(f'[{batch_idx + 1}/{len(data_loader[0])}] ')
                for nn in range(n):
                    logger.info(f'NMSE_{nn}: {iter_nmse_list[nn].avg:.3e}')
                logger.info(f'time: {iter_time.avg:.3f}')
        
        for nn in range(n):
            logger.info(f'Test NMSE_{nn}: {iter_nmse_list[nn].avg:.3e}')
        return [iter_nmse_list[nn].avg for nn in range(n)]
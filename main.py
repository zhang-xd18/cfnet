import torch
from utils.parser import args
from utils import logger
from utils import init_device, init_model
from dataset import Cost2100DataLoader
from utils.tester import test

def main():
    logger.info('=> PyTorch Version: {}'.format(torch.__version__))
    logger.info('scenario: {}, cr: {} '.format(args.scenarios, args.cr))

    # Environment initialization
    device, _ = init_device(args.seed, args.cpu, args.gpu, args.cpu_affinity)

    # Define model
    model = init_model(args)
    model.to(device)

    # Define scenarios
    scenarios = []
    for s in args.scenarios:
        scenarios.append(s)
        
    n = len(scenarios)
    performance = []
    test_loader_list = []
    
    for i in range(n):
        test_loader = Cost2100DataLoader(
            root=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.workers,
            scenario=scenarios[i],
            device=device)()
        test_loader_list.append(test_loader)
    
    performance = torch.zeros([1,n])
    if args.evaluate:
        performance = test(device=device,
             model=model,
             pretrain=args.pretrained,
             data_loader=test_loader_list)
        logger.info('NMSE performance:')
        for i in range(len(scenarios)):
            logger.info(f'{scenarios[i]}: {performance[i]:.2f}dB')
        
if __name__ == "__main__":
    main()

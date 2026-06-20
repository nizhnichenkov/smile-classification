import torch


def get_device(logger):
    logger.info('=' * 80)
    logger.info('DEVICE')
    logger.info('=' * 80)

    # nvidia
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f'Found {gpu_count} CUDA GPU(s)')

        for i in range(gpu_count):
            logger.info(
                f'GPU {i}: '
                f'{torch.cuda.get_device_name(i)}'
            )

        device = torch.device('cuda')
    
    # apple
    elif (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        logger.info(f'Using Apple Metal (MPS)')
        device = torch.device('mps')
    
    # cpu
    else:
        logger.info(f'No accelerator found. Using CPU.')
        device = torch.device('cpu')
    
    logger.info(
        f'Training device: {device}'
    )
    logger.info('\n')

    return device
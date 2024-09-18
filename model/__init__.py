import logging
logger = logging.getLogger('base')

##----## for har_dp
def create_model_dp(opt):
    from .model import DDPM as M
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    print("create_model_dp")
    return m

##----## for harmony

def create_model_harmony(opt):
    from .model_harmony import DDPM as M
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    print("create_model_harmony")
    return m

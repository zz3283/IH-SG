'''create dataset and dataloader'''
import logging
from re import split
import torch.utils.data


def create_dataloader(dataset, dataset_opt, phase):
    '''create dataloader '''
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['num_workers'],
            pin_memory=True)
    elif phase == 'val' or phase == 'test':
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found.'.format(phase))


def create_dataset_IHD(dataset_opt, phase):
    '''create dataset'''
    # mode = dataset_opt['mode']
    #from data.LRHR_dataset import LRHRDataset as D
    # dataset = D(dataroot=dataset_opt['dataroot'],     # ori
    #             datatype=dataset_opt['datatype'],
    #             l_resolution=dataset_opt['l_resolution'],
    #             r_resolution=dataset_opt['r_resolution'],
    #             split=phase,
    #             data_len=dataset_opt['data_len'],
    #             need_LR=(mode == 'LRHR')
    #             )
    ##----##for IHD
    from data.LRHR_dataset1 import LRHRDataset as D
    dataset = D(dataroot=dataset_opt['dataroot'],
                datatype=dataset_opt['datatype'],
                split=phase,
                data_len=dataset_opt['data_len'],
                )
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset

def create_dataset(dataset_opt, phase):
    '''create dataset'''
    # mode = dataset_opt['mode']
    #from data.LRHR_dataset import LRHRDataset as D
    # dataset = D(dataroot=dataset_opt['dataroot'],     # ori
    #             datatype=dataset_opt['datatype'],
    #             l_resolution=dataset_opt['l_resolution'],
    #             r_resolution=dataset_opt['r_resolution'],
    #             split=phase,
    #             data_len=dataset_opt['data_len'],
    #             need_LR=(mode == 'LRHR')
    #             )
    ##----##for mydata
    from data.mydata import mydata as D
    dataset = D(dataroot=dataset_opt['dataroot'],
                datatype=dataset_opt['datatype'],
                split=phase,
                data_len=dataset_opt['data_len'],
                )
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset

def create_dataset_test(dataset_opt, phase):
    '''create dataset'''

    ##----##for mydata
    from data.testdata import testdata as D
    dataset = D(dataroot=dataset_opt['dataroot'],
                datatype=dataset_opt['datatype'],
                split=phase,
                data_len=dataset_opt['data_len'],
                )
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset

def create_dataset_val(dataset_opt, phase):
    '''create dataset'''

    ##----##for mydata
    from data.valdata import testdata as D
    dataset = D(dataroot=dataset_opt['dataroot'],
                datatype=dataset_opt['datatype'],
                split=phase,
                data_len=dataset_opt['data_len'],
                )
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset

def create_dataset_ref(dataset_opt, phase):
    '''create dataset'''
    # mode = dataset_opt['mode']
    #from data.LRHR_dataset import LRHRDataset as D
    # dataset = D(dataroot=dataset_opt['dataroot'],     # ori
    #             datatype=dataset_opt['datatype'],
    #             l_resolution=dataset_opt['l_resolution'],
    #             r_resolution=dataset_opt['r_resolution'],
    #             split=phase,
    #             data_len=dataset_opt['data_len'],
    #             need_LR=(mode == 'LRHR')
    #             )
    ##----##for IHD
    from data.LRHR_dataset_harmony import LRHRDataset as D
    dataset = D(dataroot=dataset_opt['dataroot'],
                datatype=dataset_opt['datatype'],
                split=phase,
                data_len=dataset_opt['data_len'],
                )
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
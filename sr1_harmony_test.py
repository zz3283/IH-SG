import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/har_test_2.json',  ##config/har_test_mydata.json
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', default='val',type=str, choices=['train', 'val'], ##train  or val
                        help='Run either train(training) or val(generation)', )
    parser.add_argument('-gpu', '--gpu_ids', default='7,6,5,4,3',type=str)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # dataset
    print("load data-------------------------------------------")
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset_IHD(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            # val_set = Data.create_dataset_IHD(dataset_opt, phase)
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    print("create model------------------------------------------")
    diffusion = Model.create_model_harmony(opt)  ##
    # the next will open when train
    if opt['distributed']:
        model_restoration = torch.nn.DataParallel(diffusion)
        model_restoration.cuda()
    logger.info('Initial Model Finished')

    # Train
    print("start train-----------------------------------------")
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            print(current_epoch) ##------------------##
            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data)     # input data
                diffusion.optimize_parameters()     # diffusion
                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    print(message) ##----##
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    avg_psnr = 0.0
                    idx = 0
                    result_path = '{}/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')
                    for _,  val_data in enumerate(val_loader):
                        idx += 1
                        diffusion.feed_data(val_data)
                        diffusion.test(continous=False)
                        visuals = diffusion.get_current_visuals() ##
                        input_img = Metrics.tensor2img(visuals['Input'], min_max=(0,1))  # uint8
                        gt_img = Metrics.tensor2img(visuals['GT'], min_max=(0,1))  # uint8
                        result_img = Metrics.tensor2img(visuals['Result'], min_max=(0,1))  # uint8

                        out = visuals['Input'] * visuals['Mask'] + (1 - visuals['mask']) * visuals['Result']
                        # generation
                        Metrics.save_img(
                            gt_img, '{}/{}_{}_gt.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            input_img, '{}/{}_{}_input.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            result_img, '{}/{}_{}_res.png'.format(result_path, current_step, idx))

                        tb_logger.add_image(
                            'Iter_{}'.format(current_step),
                            np.transpose(np.concatenate(
                                (result_img, input_img, gt_img), axis=1), [2, 0, 1]),
                            idx)
                        avg_psnr += Metrics.calculate_psnr(
                            input_img, gt_img)

                        if wandb_logger:
                            wandb_logger.log_image(
                                f'validation_{idx}', 
                                np.concatenate((result_img, input_img, gt_img), axis=1)
                            )

                    avg_psnr = avg_psnr / idx
                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')
                    # log
                    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                    logger_val = logging.getLogger('val')  # validation logger
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                        current_epoch, current_step, avg_psnr))
                    # tensorboard logger
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)

                    if wandb_logger:
                        wandb_logger.log_metrics({
                            'validation/val_psnr': avg_psnr,
                            'validation/val_step': val_step
                        })
                        val_step += 1

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch-1})

        # save model
        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.')
        avg_psnr = 0.0
        avg_ssim = 0.0
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        diffusion.set_new_noise_schedule(
            opt['model']['beta_schedule']['val'], schedule_phase='val')
        for _,  val_data in enumerate(val_loader):
            idx += 1
            diffusion.feed_data(val_data)
            diffusion.test(continous=True)
            visuals = diffusion.get_current_visuals()

            # for grid image save
            gt_img = Metrics.tensor2img(visuals['GT'], min_max=(0,1))  # uint8
            input_img = Metrics.tensor2img(visuals['Input'], min_max=(0,1))  # uint8
            out = visuals['Input'] * (1 - visuals['Mask']) + visuals['Mask'] * visuals['Result']
            out = Metrics.tensor2img(out, min_max=(0, 1))  # uint8
            input_img_mode = 'grid'
            if input_img_mode == 'single':
                # single img series
                result_img = visuals['Result']  # uint8
                sample_num = result_img.shape[0]
                for iter in range(0, sample_num):
                    Metrics.save_img(
                        Metrics.tensor2img(result_img[iter], min_max=(0,1)), '{}/{}_{}_result_{}.png'.format(result_path, current_step, idx, iter))
            else:
                # grid img
                result_img_grid = Metrics.tensor2img(visuals['Result'], min_max=(0,1))  # uint8
                Metrics.save_img(
                    Metrics.tensor2img(visuals['Result'][-1], min_max=(0,1)), '{}/{}_{}_result.png'.format(result_path, current_step, idx))


            Metrics.save_img(
                gt_img, '{}/{}_{}_gt.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                input_img, '{}/{}_{}_input.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                out, '{}/{}_{}_out.png'.format(result_path, current_step, idx))

            # generation
            eval_psnr = Metrics.calculate_psnr(Metrics.tensor2img(visuals['Result'][-1], min_max=(0,1)), gt_img)
            eval_ssim = Metrics.calculate_ssim(Metrics.tensor2img(visuals['Result'][-1], min_max=(0,1)), gt_img)

            avg_psnr += eval_psnr
            avg_ssim += eval_ssim

            if wandb_logger and opt['log_eval']:
                wandb_logger.log_eval_data(result_img, Metrics.tensor2img(visuals['Result'][-1], min_max=(0,1)), gt_img, eval_psnr, eval_ssim)

        avg_psnr = avg_psnr / idx
        avg_ssim = avg_ssim / idx

        # log
        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
        logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
        logger_val = logging.getLogger('val')  # validation logger
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim：{:.4e}'.format(
            current_epoch, current_step, avg_psnr, avg_ssim))

        if wandb_logger:
            if opt['log_eval']:
                wandb_logger.log_eval_table()
            wandb_logger.log_metrics({
                'PSNR': float(avg_psnr),
                'SSIM': float(avg_ssim)
            })

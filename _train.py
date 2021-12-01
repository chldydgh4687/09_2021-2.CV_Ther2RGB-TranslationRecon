import time
import os
import numpy as np
import torch
from _model import pix2pixHD

import wandb
import _util
from _util import args
from _util import utils
from _util.data_loader import Dataset_Loader
from collections import OrderedDict
from tqdm import tqdm

def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    opt = args.Options().parse()

    #keep training
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, "iter.txt")
    if opt.train_continue_train:
        try:
            start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=",", dtype=int)
        except:
            start_epoch, epoch_iter = 1, 0
        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
    else:
        start_epoch, epoch_iter = 1, 0

    opt.train_print_freq = utils.lcm_freq(opt.train_print_freq, opt.batch_size)

    # Usage Wandb
    wandb.init(project="Ther2RGB")
    wandb.run.name = opt.name

    # Data Loader
    dataset = _util.data_loader.CreateDataset(opt)
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)

    model = pix2pixHD.create_model(opt)
    optimizer_G, optimizer_D = model.optimizer_G, model.optimizer_D

    total_steps = (start_epoch - 1) * dataset_size + epoch_iter
    #print( opt.train_niter + opt.train_niter_decay + 1)
    display_delta = total_steps % opt.train_display_freq
    print_delta = total_steps % opt.train_print_freq
    save_delta = total_steps % opt.train_save_latest_freq

    # model structure load

    for epoch in tqdm(range(start_epoch, opt.train_niter + opt.train_niter_decay + 1)):
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size
        for i, data in tqdm(enumerate(dataset, start=epoch_iter)):
            if total_steps % opt.train_print_freq == print_delta:
                iter_start_time = time.time()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size

            # whether to collect output images
            if opt.train_debug:
                save_fake=True
            else:
                save_fake = total_steps % opt.train_display_freq == display_delta

            #print(data["label"].shape, data["image"].shape)

            ############## Image Processing ##################

            data['label'] = data['label'].to(device)
            data['image'] = data['image'].to(device)
            ############## Forward Pass ######################
            losses, generated = model(data['label'], data['image'], infer=save_fake)

            losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
            loss_dict = dict(zip(model.loss_names, losses))

            if not opt.train_no_gan_loss:
                loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
                loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat', 0) + loss_dict.get('G_VGG',0) + loss_dict.get('G_L1',0)

                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()

                # update discriminator weights
                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()

            if total_steps % opt.train_print_freq == print_delta:
                errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
                t = (time.time() - iter_start_time) / opt.train_print_freq
                if not opt.train_debug:
                    for name in errors.keys():
                        wandb.log({name: errors[name]})

            ### display output images
            if save_fake:
                visuals = OrderedDict([('real_image', utils.tensor2im(data['image'], \
                                                                     normalize=opt.normalize)),
                                       ('synthesized_image', utils.tensor2im(generated, \
                                                                            normalize=opt.normalize)),
                                       ('input_label', utils.tensor2label(data['label'], opt.label_nc))
                                       ])

                if not opt.train_debug:
                    for label, image in visuals.items():
                        wandb.log({label: wandb.Image(image)})

            ### save latest model
            if total_steps % opt.train_save_latest_freq == save_delta:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                model.save('latest')
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

            if epoch_iter >= dataset_size:
                break

        # end of epoch
        iter_end_time = time.time()
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.train_niter + opt.train_niter_decay, time.time() - epoch_start_time))

        ### save model for this epoch
        if epoch % opt.train_save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save('latest')
            model.save(epoch)
            np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')

        ### linearly decay learning rate after certain iterations
        if epoch > opt.train_niter:
            model.update_learning_rate()




if __name__ == "__main__":
    train()
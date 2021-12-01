from _model import Generator
from _model import Discriminator
from _model.ModelHelp import ModelHelper

from _model import VGGpre
import torch
import torch.nn as nn
from torch.autograd import Variable
from _util.utils import ImagePool

def create_model(opt):
    model = Pix2PixHDModel()
    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    return model

class Pix2PixHDModel(ModelHelper):

    def init_loss_filter(self, use_gan_loss, use_gan_feat_loss, use_vgg_loss, use_l1_loss):
        flags = (use_gan_loss, use_gan_feat_loss, use_vgg_loss, use_gan_loss, use_gan_loss, use_l1_loss)

        def loss_filter(g_gan, g_gan_feat, g_vgg, d_real, d_fake, g_l1):
            return [l for (l, f) in zip((g_gan, g_gan_feat, g_vgg, d_real, d_fake, g_l1), flags) if f]

        return loss_filter

    def initialize(self, opt):
        ModelHelper.initialize(self,opt)

        if not opt.isTrain:  # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc

        ##### define networks
        # Generator network
        netG_input_nc = input_nc

        self.netG = Generator.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers,
                                      opt.n_blocks_local, opt.norm, use_noise=opt.use_noise, gpu_ids=self.gpu_ids)

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.train_no_lsgan
            netD_input_nc = input_nc + opt.output_nc
            self.netD = Discriminator.define_D(netD_input_nc, opt.train_ndf, opt.train_n_layers_D, opt.norm, use_sigmoid,
                                          opt.train_num_D, not opt.train_no_ganFeat_loss, gpu_ids=self.gpu_ids)

        ### Encoder network
        if self.opt.verbose:
            print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.train_continue_train or opt.train_load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)

                # set loss functions and optimizers
        if self.isTrain:
            if opt.train_pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.train_pool_size)
            self.old_lr = opt.train_lr

            # define loss functions
            self.loss_filter = self.init_loss_filter(not opt.train_no_gan_loss, not opt.train_no_ganFeat_loss,
                                                     not opt.train_no_vgg_loss,
                                                     not opt.train_no_l1_loss)
            self.criterionGAN = GANLoss(use_lsgan=not opt.train_no_lsgan, tensor=self.Tensor)
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionSmoothL1 = HuberLoss(delta=1. / opt.ab_norm)
            if not opt.train_no_vgg_loss:
                self.criterionVGG = VGGLoss()

            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN', 'G_GAN_Feat', 'G_VGG', 'D_real', 'D_fake', 'G_L1')

            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0:
                import sys
                if sys.version_info >= (3, 0):
                    finetune_list = set()
                else:
                    from sets import Set
                    finetune_list = Set()

                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():
                    if key.startswith('model' + str(opt.n_local_enhancers)):
                        params += [value]
                        finetune_list.add(key.split('.')[0])
                print(
                    '------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                print('The layers that are finetuned are ', sorted(finetune_list))
            else:
                params = list(self.netG.parameters())

            self.optimizer_G = torch.optim.Adam(params, lr=opt.train_lr, betas=(opt.train_beta1, 0.999))

            # optimizer D
            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.train_lr, betas=(opt.train_beta1, 0.999))

    def encode_input(self, label_map, real_image=None, infer=False):
        if self.opt.label_nc == 0:
            input_label = label_map.data
        else:
            # create one-hot vector for label map
            size = label_map.size()
            oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
            input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
            if self.opt.data_type == 16:
                input_label = input_label.half()

        input_label = Variable(input_label, volatile=infer)

        # real images for training
        if real_image is not None:
            real_image = Variable(real_image.data)

        return input_label, real_image

    def discriminate(self, input_label, test_image, use_pool=False):

        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def forward(self, label, image, infer=False):
        # Encode Inputs

        input_label, real_image = self.encode_input(label, image)

        # Fake Generation
        input_concat = input_label

        fake_image = self.netG.forward(input_concat)

        # Fake Detection and Loss
        loss_G_GAN = 0
        loss_D_real = 0
        loss_D_fake = 0

        if not self.opt.train_no_gan_loss:
            # Fake Detection and Loss
            pred_fake_pool = self.discriminate(input_label, fake_image, use_pool=True)
            loss_D_fake = self.criterionGAN(pred_fake_pool, False)

            # Real Detection and Loss
            pred_real = self.discriminate(input_label, real_image)
            loss_D_real = self.criterionGAN(pred_real, True)

            # GAN loss (Fake Passability Loss)
            pred_fake = self.netD.forward(torch.cat((input_label, fake_image), dim=1))
            loss_G_GAN = self.criterionGAN(pred_fake, True)

        loss_G_L1 = 0
        loss_G_L1 = 10 * torch.mean(self.criterionSmoothL1(fake_image.type(torch.cuda.FloatTensor),
                                                           real_image.type(torch.cuda.FloatTensor)))
        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.train_no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.train_n_layers_D + 1)
            D_weights = 1.0 / self.opt.train_num_D
            for i in range(self.opt.train_num_D):
                for j in range(len(pred_fake[i]) - 1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                                       self.criterionFeat(pred_fake[i][j],
                                                          pred_real[i][j].detach()) * self.opt.train_lambda_feat

        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.train_no_vgg_loss:
            loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.train_lambda_feat

        # Only return the fake_B image if necessary to save BW
        return [self.loss_filter(loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake, loss_G_L1),
                None if not infer else fake_image]

    def inference(self, label, image=None):
        # Encode Inputs
        image = Variable(image) if image is not None else None
        input_label, real_image = self.encode_input(label, image, infer=True)

        # Fake Generation
        input_concat = input_label

        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_image = self.netG.forward(input_concat)
        else:
            with torch.no_grad():
                fake_image = self.netG.forward(input_concat)

        return fake_image

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch)
        self.save_network(self.netD, 'D', which_epoch)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())

        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.train_lr, betas=(self.opt.train_beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.train_lr / self.opt.train_niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr


class InferenceModel(Pix2PixHDModel):
    def forward(self, inp):
        label, inst = inp
        return self.inference(label, inst)


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class HuberLoss(nn.Module):
    def __init__(self, delta=.01):
        super(HuberLoss, self).__init__()
        self.delta=delta
    def __call__(self, in0, in1):
        mask = torch.zeros_like(in0)
        mann = torch.abs(in0-in1)
        eucl = .5 * (mann**2)
        mask[...] = mann < self.delta

        # loss = eucl*mask + self.delta*(mann-.5*self.delta)*(1-mask)
        loss = eucl*mask/self.delta + (mann - .5*self.delta)*(1-mask)
        return torch.sum(loss, dim=1, keepdim=True)

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vgg = VGGpre.Vgg19().to(device)
        self.criterion =nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self,x,y):
        if x.shape[1] == 1:
            x = torch.cat([x,x,x], dim=1)
        if y.shape[1] == 1:
            y = torch.cat([y,y,y], dim=1)

        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i],y_vgg[i].detach())

        return loss

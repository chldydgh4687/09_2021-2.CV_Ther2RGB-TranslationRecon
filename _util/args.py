import argparse
import os

class Options():

    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # experiment specifics
        self.parser.add_argument('--name', type=str, default='label2city', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./_checkpoints', help='models are saved here')
        self.parser.add_argument('--model', type=str, default='pix2pixHD', help='which model to use')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        self.parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--data_type', default=32, type=int, choices=[8, 16, 32], help="Supported data type i.e. 8, 16, 32 bit")
        self.parser.add_argument('--verbose', action='store_true', default=False, help='toggles verbose')
        self.parser.add_argument('--fp16', action='store_true', default=False, help='train with AMP')
        self.parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
        self.parser.add_argument('--gray_only', action='store_true')
        self.parser.add_argument('--color_only', action='store_true')
        # input/output sizes
        self.parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        self.parser.add_argument('--load_size', type=int, default=512, help='scale images to this size')
        self.parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
        self.parser.add_argument('--label_nc', type=int, default=0, help='# of input label channels')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        # for setting inputs
        self.parser.add_argument('--dataroot', type=str, default='./datasets/cityscapes/')
        self.parser.add_argument('--resize_or_crop', type=str, default='scale_width', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--ab_norm', type=float, default=110., help='colorization normalization factor')
        self.parser.add_argument('--ab_max', type=float, default=110., help='maximimum ab value')
        self.parser.add_argument('--ab_quant', type=float, default=10., help='quantization factor')
        self.parser.add_argument('--l_norm', type=float, default=100., help='colorization normalization factor')
        self.parser.add_argument('--l_cent', type=float, default=50., help='colorization centering factor')
        self.parser.add_argument('--normalize', action='store_true', help='Normalize input data')
        self.parser.add_argument('--isreverse',action='store_true')
        self.parser.add_argument('--isedge',action='store_true')
        self.parser.add_argument('--isfrequency', action='store_true')
        self.parser.add_argument('--ther_only', action='store_true')
        self.parser.add_argument('--use_noise', action='store_true')
        self.parser.add_argument('--use_colorjitter_contrast', action='store_true')
        self.parser.add_argument('--use_colorjitter_saturation', action='store_true')
        # for displays
        self.parser.add_argument('--display_winsize', type=int, default=512,  help='display window size')
        self.parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')

        # for generator
        self.parser.add_argument('--netG', type=str, default='global', help='selects model to use for netG')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--n_downsample_global', type=int, default=4, help='number of downsampling layers in netG')
        self.parser.add_argument('--n_blocks_global', type=int, default=9, help='number of residual blocks in the global generator network')
        self.parser.add_argument('--n_blocks_local', type=int, default=3, help='number of residual blocks in the local enhancer network')
        self.parser.add_argument('--n_local_enhancers', type=int, default=1, help='number of local enhancers to use')
        self.parser.add_argument('--niter_fix_global', type=int, default=0, help='number of epochs that we only train the outmost local enhancer')

        # for instance-wise features
        self.parser.add_argument('--no_instance', action='store_true', help='if specified, do *not* add instance map as input')
        self.parser.add_argument('--instance_feat', action='store_true', help='if specified, add encoded instance features as input')
        self.parser.add_argument('--label_feat', action='store_true', help='if specified, add encoded label features as input')
        self.parser.add_argument('--feat_num', type=int, default=3, help='vector length for encoded features')
        self.parser.add_argument('--load_features', action='store_true', help='if specified, load precomputed feature maps')
        self.parser.add_argument('--n_downsample_E', type=int, default=4, help='# of downsampling layers in encoder')
        self.parser.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')
        self.parser.add_argument('--n_clusters', type=int, default=10, help='number of clusters for features')

        #Training.................................................................................................................
        # for displays
        self.parser.add_argument('--train_display_freq', type=int, default=100,
                                 help='frequency of showing training results on screen')
        self.parser.add_argument('--train_print_freq', type=int, default=100,
                                 help='frequency of showing training results on console')
        self.parser.add_argument('--train_save_latest_freq', type=int, default=1000,
                                 help='frequency of saving the latest results')
        self.parser.add_argument('--train_save_epoch_freq', type=int, default=10,
                                 help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--train_no_html', action='store_true',
                                 help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--train_debug', action='store_true',
                                 help='only do one epoch and displays at each iteration')

        # for training
        self.parser.add_argument('--train_continue_train', action='store_true',
                                 help='continue training: load the latest model')
        self.parser.add_argument('--train_load_pretrain', type=str, default='',
                                 help='load the pretrained model from the specified location')
        self.parser.add_argument('--train_which_epoch', type=str, default='latest',
                                 help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--train_phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--train_niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--train_niter_decay', type=int, default=100,
                                 help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--train_beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--train_lr', type=float, default=0.0002, help='initial learning rate for adam')

        # for discriminators
        self.parser.add_argument('--train_num_D', type=int, default=2, help='number of discriminators to use')
        self.parser.add_argument('--train_n_layers_D', type=int, default=3,
                                 help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--train_ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--train_lambda_feat', type=float, default=10.0,
                                 help='weight for feature matching loss')
        self.parser.add_argument('--train_no_gan_loss', action='store_true')
        self.parser.add_argument('--train_no_ganFeat_loss', action='store_true',
                                 help='if specified, do *not* use discriminator feature matching loss')
        self.parser.add_argument('--train_no_vgg_loss', action='store_true',
                                 help='if specified, do *not* use VGG feature matching loss')
        self.parser.add_argument('--train_no_lsgan', action='store_true',
                                 help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--train_no_l1_loss', action='store_true', help='do *not* use L1 loss')
        self.parser.add_argument('--train_photometric', action='store_true')
        self.parser.add_argument('--train_pool_size', type=int, default=0,
                                 help='the size of image buffer that stores previously generated images')

        # Test parameter............................................................................................................................
        self.parser.add_argument('--test_ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--test_results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--test_aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--test_phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--test_which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--test_how_many', type=int, default=50, help='how many test images to run')
        self.parser.add_argument('--test_cluster_path', type=str, default='features_clustered_010.npy', help='the path for clustered results of encoded features')
        self.parser.add_argument('--test_use_encoded_image', action='store_true', help='if specified, encode the real image to get the feature map')
        self.parser.add_argument("--test_save_result", action='store_true', help='Save the predicted results')
        self.isTrain = True


    def print_option(self,options,save_path):
        args = vars(options)

        common_list = {}
        training_list = {}
        test_list = {}
        for k, v in sorted(args.items()):
            if "train" in k :
                training_list[k] = v
            elif "test" in k :
                test_list[k] = v
            else:
                common_list[k] = v

        parameter_log = save_path + "/" + options.name + "_opt.txt"
        with open(parameter_log, "wt") as opt_file:
            for k, v in sorted(common_list.items()):
                opt_file.write('%s : %s\n' % (str(k), str(v)))

            opt_file.write("-----------Training_Options-----------\n")
            for k, v in sorted(training_list.items()):
                opt_file.write('%s : %s\n' % (str(k), str(v)))

            opt_file.write("-----------Testing_Options-----------\n")
            for k, v in sorted(test_list.items()):
                opt_file.write('%s : %s\n' % (str(k), str(v)))

        opt_file.close()


    def parse(self):

        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain
        checkpoint_dir = "_checkpoints/"+self.opt.name
        try:
            os.mkdir(checkpoint_dir)
        except:
            pass

        self.print_option(self.opt, checkpoint_dir)

        return self.opt









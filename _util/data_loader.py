import torch.utils.data as data
from PIL import Image
from torchvision import transforms as tf
import os
from _util import utils
import torch

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]

def CreateDataset(opt):
    dataset = None
    dataset = Dataset_Loader()
    dataset.data_loader(opt)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = opt.batch_size,
        shuffle = False,
        num_workers = 2
    )
    return dataloader

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir_):
    images = []
    assert os.path.isdir(dir_), '%s is not a valid directory' % dir_

    for root, _, fnames in sorted(os.walk(dir_)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

class Dataset_Loader():

    def data_loader(self,opt):
        self.opt = opt
        dataroot = "T2R_Dataset"

        self.dir_A = os.path.join(dataroot,opt.train_phase + "_A")
        self.A_paths = sorted(make_dataset(self.dir_A))
        # path ->  path list
        #self.A_paths = [path for path in self.A_paths ]

        if
        self.dir_B = os.path.join(dataroot,opt.train_phase + "_B")
        self.B_paths = sorted(make_dataset(self.dir_B))
        # path -> path list
        self.dataset_size = len(self.A_paths)

        # photometric :
        #self.thm_gamma_low = 0.5
        #self.thm_gamma_high = 1.5

        contrast_param = (0.3, 1)
        self.colorjitter1 = tf.ColorJitter(contrast_param)

    def __getitem__ (self, index):
        A_path = self.A_paths[index]
        A = utils.gen_ther_color_pil(A_path)

        params = utils.get_params(self.opt, A.size)
        transform_A = utils.get_transform(self.opt, params)



        # scale_width
        A_tensor = transform_A(A)
        # 랜덤 대비 변화
        A_tensor = self.colorjitter1(A_tensor)

        B_path = self.B_paths[index]
        B = Image.open(B_path)

        params = utils.get_params(self.opt, B.size)
        transform_B = utils.get_transform(self.opt, params)
        B_tensor = transform_B(B)

        input_dict = {"label": A_tensor, "image": B_tensor, "path" : A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batch_size * self.opt.batch_size





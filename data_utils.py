from os import listdir
from os.path import join

from PIL import Image

from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, functional
from scipy.misc import imresize

import os
import numpy as np
import torch
import pickle as p

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


# def train_hr_transform(crop_size):
#     return Compose([
#         RandomCrop(crop_size),
#         ToTensor(),
#     ])


# def train_lr_transform(crop_size, upscale_factor):
#     return Compose([
#         ToPILImage(),
#         Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC), #88//4 = 22
#         ToTensor()
#     ])

def train_hr_transform(crop_size):
    return Compose([
        # RandomCrop(crop_size),
        Resize((crop_size,crop_size)),
        ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        # ToPILImage(), #들어올때부터 PIL이라 변형안해도 됨
       Resize((crop_size,crop_size)),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])

def load_img_path(folder):
    name = list(sorted(os.listdir(os.path.join(folder))))
    pth= [ os.path.join(folder, name) for name in name if name[-3:]=='jpg']
    return pth

def load_tar_path(folder):
    name = list(sorted(os.listdir(os.path.join(folder))))
    pth= [ os.path.join(folder, name) for name in name ]
    return pth ,name


class RainHazeImageDataset(Dataset):
    def __init__(self, root_dir, mode, aug=False, transform=None):
        """
        At __init__ state, we read in all the image paths of the entire dataset instead of image data
        :param root_dir: directory of files containing the paths to all rain images
        :param mode: 'train', 'val', or 'test'
        :param aug: Whether augment the input image
        :param transform:
        """
        self.root_dir = root_dir
        self.mode = mode
        self.aug = aug
        self.transform = transform
        self.path = os.path.join(self.root_dir, (mode + '_s_rain.txt'))
        self.in_list = generate_new_seq(os.path.join(self.root_dir, (mode + '_in.txt')))
        self.real_list = generate_new_seq(os.path.join(self.root_dir, (mode + '_real.txt')))
        self.streak_list = generate_new_seq(os.path.join(self.root_dir, (mode + '_streak.txt')))
        self.trans_list = generate_new_seq(os.path.join(self.root_dir, (mode + '_trans.txt')))
        self.clean_list = generate_new_seq(os.path.join(self.root_dir, (mode + '_clean.txt')))
        self.atm_list = generate_new_seq(os.path.join(self.root_dir, (mode + '_atm.txt')))
        self.no_realrain = len(self.real_list)

        self.target_folder='E:\\Image_Captioning_data\\default\\train'
        self.tar_pth, self.tar_name = load_tar_path(self.target_folder)


    def __len__(self):
        return len(self.in_list)

    def __getitem__(self, idx):
        if self.mode == 'train':
            noise_trigger = True
        else:
            noise_trigger = False

        real_rain_index = np.random.randint(self.no_realrain)
        rain = read_image(self.in_list[idx], noise_trigger)
        im_gt = read_image(self.clean_list[idx]) # clean image = input - sparse - middle - dense
        st_gt = read_image(self.streak_list[idx]) # sparse streak
        trans_gt = read_image(self.trans_list[idx])  # middle streak
        atm_gt = read_image(self.atm_list[idx])   # dense streak
        realrain = read_image(self.real_list[real_rain_index])
        
        # render haze
        # if np.min(trans_gt) == 0:
        #     print(self.trans_list[idx])

        input_list = [rain, st_gt, trans_gt, atm_gt, im_gt, realrain]
        if self.aug:
            input_list = augment(input_list)
        else:
            input_list = ImageResize(input_list, size=256)
            #input_list = RandomCrop(input_list, size=128)
            
        
        if self.transform:
            input_list = self.transform(input_list)

        filepath=os.path.join(self.tar_pth[idx], self.tar_name[idx])
        with open(filepath, 'rb') as file:  
            target = p.load(file)
            target = torch.squeeze(target, 0)
            target = target.cpu().detach().numpy()

        input_list.append(target)

        return input_list

class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        # ex) data/DIV2K_train_HR\\00001.jpg
        self.hr_image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)] #train file root , image load

        #++
        self.lr_image_filenames = [join('E:\\Image_Captioning_data\\haze\\train2014', i) for i in listdir('E:\\Image_Captioning_data\\haze\\train2014') if is_image_file(i)] #lr image load

        #crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

        self.h, self.w = None, None

    # def __getitem__(self, index):
    #     hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
    #     lr_image = self.lr_transform(hr_image)
    #     return lr_image, hr_image

    #dahee 수정
    def __getitem__(self, index):
        hr_image = Image.open(self.hr_image_filenames[index]) #hr 이미지 저장
        lr_image = Image.open(self.lr_image_filenames[index]) #lr 이미지 저장

        # #랜덤크롭
        # i, j, h, w = RandomCrop.get_params(hr_image, output_size=(88, 88)) #88,88 값 직접 바꿔주기 
        # hr_image = functional.crop(hr_image, i, j, h, w) # 같은위치에서 자르기
        # lr_image = functional.crop(lr_image, i, j, h, w)

        self.w, self.h = hr_image.size
        hr_image = self.hr_transform(hr_image)  #32로 리사이징
        lr_image = self.lr_transform(lr_image)
        
        return lr_image, hr_image

    def __len__(self):
        return len(self.hr_image_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.hr_image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)] #hr img
        #++
        self.lr_image_filenames = [join('data/DIV2K_valid_LR', i) for i in listdir('data/DIV2K_valid_LR') if is_image_file(i)] #lr img

    def __getitem__(self, index):
        hr_image = Image.open(self.hr_image_filenames[index]) #hr 저장
        lr_image = Image.open(self.lr_image_filenames[index]) #lr 저장 32,32
        w, h = hr_image.size

        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor) #128
        # lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC) # 줄이려고 받아옴 scale
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        # hr_image = CenterCrop(crop_size)(hr_image) #안잘라도되니까 필요 없음
        # lr_image = lr_scale(hr_image)  #리사이징 부분 22

        hr_restore_img = hr_scale(lr_image) #32에서 128로 늘린사진
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.hr_image_filenames)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
        self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)] #이미지 다 불러와짐

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1] #18001.png
        lr_image = Image.open(self.lr_filenames[index]) #32,32로 불러옴
        w, h = lr_image.size #32,32
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = Resize((32,32), interpolation=Image.BICUBIC) #hr을 32x32로 리사이징
        hr_image = hr_scale(hr_image)
        # hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        # hr_restore_img = hr_scale(lr_image) #4배로 늘려줌
        return image_name, ToTensor()(lr_image), ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_filenames)


def generate_new_seq(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    file_list = []
    for line in lines:
        file_list.append(line.strip())
    f.close()
    return file_list#[1:10000]

def read_image(image_path, noise=False):
    """
    function: read image function
    :param image_path: input image path
    :param noise: whether apply noise on image
    :return: image in numpy array, range [0,1]
    """
    img_file = Image.open(image_path)
    img_data = np.array(img_file, dtype=np.float32)

    if len(img_data.shape) < 3:
        img_data = np.dstack((img_data, img_data, img_data))
    (h, w, c) = img_data.shape

    if h < 300 or w < 300:
        if h > w:
            img_data = imresize(img_data, [int((300*h)/w*1.5),300])
        if h < w:
            img_data = imresize(img_data, [300,int((300*w)/h*1.5)])
    #print(2,h,w)

    if noise:
        (h,w,c) = img_data.shape
        noise = np.random.normal(0,1,[h,w])
        noise = np.dstack((noise, noise, noise))
        img_data = img_data + noise
    img_data = img_data.astype(np.float32)/255.0
    img_data[img_data > 1.0] = 1.0
    img_data[img_data < 0] = 0.0
    return img_data.astype(np.float32)

def ImageResize(input_list, size=256):
    from skimage.transform import resize
    output_list = []
    for item in input_list:
        item = resize(item, [size, size])
        output_list.append(item.astype(np.float32))
    return output_list

def RandomCrop(input_list, size=224):
    output_list = []
    num_of_length = len(input_list)
    h, w, c = input_list[0].shape
    try:
        row = np.random.randint(h-size)
        col = np.random.randint(w-size)
    except:
        print("random low value leq high value")
        print(h, w, c)
    for i in range(num_of_length-1):
        item = input_list[i]
        item = item[row:row+size, col:col+size, :]
        output_list.append(item)
    h,w,c = input_list[-1].shape
    row = np.random.randint(h-size)
    col = np.random.randint(w-size)
    item = input_list[-1][row:row+size, col:col+size, :]
    output_list.append(item)
    assert(len(input_list)==len(output_list))
    return output_list

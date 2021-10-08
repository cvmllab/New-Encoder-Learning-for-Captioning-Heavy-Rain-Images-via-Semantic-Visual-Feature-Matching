import argparse
import os
from math import log10

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from data_utils import  RainHazeImageDataset
from loss import GeneratorLoss
from model import AtJ, Encoder #change.
from torchvision import transforms

import gc

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--num_epochs', default=300, type=int, help='train epoch number')
#parser.add_argument('--trained_Atj', default='C:\\pung\\derain_imagecaption\\dehaze_patch_encoder\\epochs_patch_Ats\\netG_epoch_198.pth', type=str, help='pretrained weight')
parser.add_argument('--trained_Atj', default='C:\\pung\\derain_imagecaption\\dehaze_patch_encoder\\epochs_patch_Ats_encoder_target_8000\\netG_epoch_144.pth', type=str, help='pretrained weight')
#parser.add_argument('--encoder', default='C:\\pung\\image_captioning\\BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar', type=str)
parser.add_argument('--encoder', default='C:\\pung\\derain_imagecaption\\dehaze_patch_encoder\\epochs_patch_Ats_encoder_target_8000\\encoder_144.pth', type=str)

def tensor_to_image(tensor):
    if type(tensor) in [torch.autograd.Variable]:
        img = tensor.data[0].cpu().detach().numpy()
    else:
        img = tensor[0].cpu().detach().numpy()
    img = img.transpose((1,2,0))
    #try:
    img = np.clip(img, 0, 255)
    if img.shape[-1] == 1:
        img = np.dstack((img, img, img))
    # except:
    #     #print("invalid value catch")
    #     Image.fromarray(img).save('catch.jpg')
    return img


def write_image_stage(path, input_list, output_list):
    # tensor_zero = torch.zeros(self.batch_size, 3, self.image_size, self.image_size)
    st_out = output_list[0].cpu()
    trans_out = output_list[1].cpu()
    atm_out = output_list[2].cpu()
    J_out = output_list[3].cpu()

    im_in = input_list[0]

    recons = (im_in - (1 - trans_out.cpu()) * atm_out.cpu()) / (trans_out.cpu() + 0.0001) - st_out.cpu()
    #recons = (recons - torch.min(recons)/ torch.max(recons) - torch.min(recons))
    input_row = torch.cat(input_list[0:-2], dim=3)
    output_row = torch.cat(
        (recons, st_out.cpu(), trans_out.cpu(), atm_out.cpu(), J_out.cpu()), dim=3)
    painter = torch.cat((input_row, output_row), dim=2)
    img = tensor_to_image(painter)
    img = np.clip(img * 255, 0, 255)
    painter_image = Image.fromarray(img.astype(np.uint8))
    painter_image.save(path)


class ToTensor(object):
    """ Conver ndarray to Tensors"""
    def __call__(self, image_list):
        # input image_list is: H x W x C
        # torch image_list is: C x H x W
        tensor_list = []
        for image in image_list:
            image = image.transpose((2, 0, 1))
            tensor_list.append(image)
        return tensor_list

        
if __name__ == '__main__':
    opt = parser.parse_args()
    
    NUM_EPOCHS = opt.num_epochs
    fine_tune_encoder = True


    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                                 
    train_set = RainHazeImageDataset('E:\\Image_Captioning_data\\path_8000\\train', 'train',
                                           aug=False,
                                           transform=transforms.Compose([ToTensor()]))
    print(len(train_set))

    train_loader = DataLoader(dataset=train_set, num_workers=2, batch_size=4, shuffle=True) #배치사이즈 조절 가능 
    
    netG = AtJ()
    if opt.trained_Atj != None:
        trained_Atj = torch.load(opt.trained_Atj)
        netG.load_state_dict(trained_Atj)
    

    if opt.encoder == None:
        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=1e-4) if fine_tune_encoder else None
    else:
        encoder_ckt = torch.load(opt.encoder)
        encoder = encoder_ckt['encoder']
        # encoder_optimizer = encoder_ckt['encoder_optimizer']
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=1e-4) if fine_tune_encoder else None
        encoder.fine_tune(fine_tune_encoder)


    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    
    generator_criterion = GeneratorLoss()

    if torch.cuda.is_available():
        netG.cuda()
        encoder.cuda()
        generator_criterion.cuda()

    
    optimizerG = optim.Adam(netG.parameters())
    
    results = { 'Ats_loss' : [], 'Feature_loss': [] }
    
    iter = 1

    for epoch in range(144, NUM_EPOCHS + 1):
        count = 0
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'Ats_loss' : 0, 'Feature_loss': 0}
    
        netG.train()
        for input_list in train_bar:
            batch_size = input_list[0].size(0)
            running_results['batch_sizes'] += batch_size
    
            image_in_var = Variable(input_list[0]).cuda()

            J, A, t, s = netG(image_in_var) 

            F = encoder(J)

            netG.zero_grad()
            optimizerG.zero_grad()
            encoder_optimizer.zero_grad()

            total_loss, Ats_loss, Feature_loss = generator_criterion(J, A, t, s, F, input_list)
            #Feature_loss = L1_loss(F, pre_encoder(clean_gt_var))

            #Ats_loss.backward()
            #Feature_loss.backward()
            total_loss.backward()

            optimizerG.step()
            encoder_optimizer.step()

            running_results['Ats_loss'] += Ats_loss.item() * batch_size
            running_results['Feature_loss'] += Feature_loss.item() * batch_size
    
            train_bar.set_description(desc='[%d/%d]  Ats_loss: %.8f, Feature_loss: %.8f' % (
                epoch, NUM_EPOCHS,
                running_results['Ats_loss'] / running_results['batch_sizes'],
                running_results['Feature_loss'] / running_results['batch_sizes']))
            

            if count % 50 == 0:
                output_list = [ s, t, A, J ]
                save_dir = 'E:\Image_Captioning_data\heavyrain_imagecaption_result\\out_patch_Ats_encoder_target_8000'
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                save_dir = save_dir + '\\' + str(epoch)
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                write_image_stage(save_dir+'\output' + str(count) + '.jpg', input_list, output_list)

            results['Ats_loss'].append(running_results['Ats_loss'] / running_results['batch_sizes'])
            results['Feature_loss'].append(running_results['Feature_loss'] / running_results['batch_sizes'])

            if iter % 3 == 0 and iter != 0:
                out_path = 'statistics_patch_Ats_encoder_target_8000/'
                data_frame = pd.DataFrame(
                    data={'Ats_loss': results['Ats_loss'], 'Feature_loss': results['Feature_loss']}, index=range(1, iter+1))
                data_frame.to_csv(out_path + 'srf' + '_train_results.csv', index_label='Epoch')
            count += 1
            iter += 1
       
        # save model parameters
        if epoch % 3 == 0 and epoch != 0:
            torch.save(netG.state_dict(), 'epochs_patch_Ats_encoder_target_8000/netG_epoch_%d.pth' % (epoch))
            state = {'encoder' : encoder,
            'encoder_optimizer' : encoder_optimizer }
            torch.save(state, 'epochs_patch_Ats_encoder_target_8000/encoder_%d.pth' % (epoch))
      
        gc.collect() #가비지컬렉터
    

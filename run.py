import os
import numpy as np
import cv2
import sys
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributions
from utils import get_dataset_list, CustomFaceDataset, cycle, get_val_hter, TPC_loss
from vgg_face_dag import vgg_face_dag, spoof_model
import pdb
import torchvision.utils as tutil
import argparse
import torchvision
from parameters import *
import time
import scipy.io
import pdb
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('-datasetPath', required=False, default= '/project01/cvrl/datasets/CYBORG_centered_face/',type=str)
parser.add_argument('-heatmaps', required=False, default= '/project01/cvrl/datasets/CYBORG_centered_face/train/heatmaps/',type=str)
parser.add_argument('-csvPath', required=False, default='/project01/cvrl/datasets/CYBORG_centered_face/csvs/original.csv',type=str)
parser.add_argument('-network', required=False, default= 'densenet',type=str)
parser.add_argument('-nClasses', required=False, default= 2,type=int)
parser.add_argument('-outputPath',required=False, default= './models/')
args = parser.parse_args()

os.makedirs(args.outputPath,exist_ok=True)

np.set_printoptions(formatter='10.3f')
torch.set_printoptions(sci_mode=False, threshold=5000)

#My custom vgg16 that can get the appropriate features and cams
class VGG(torch.nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        # Make the vgg16 into a binary classifier
        self.model = torchvision.models.vgg16(pretrained=True)
        self.num_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = torch.nn.Linear(self.num_features, 2)

        # Apply the hooks (defined below)
        self.target_layer = self.model.features[-1]
        self.target_layer.register_forward_hook(self.forward_hook)
        self.model.classifier[1].register_forward_hook(self.feature_hook)

        # Storage for calculating the gradcams and features
        self.activations = None
        self.gradients = None
        self.features = None

    def forward_hook(self, module, input, output):
        self.activations = output

    def feature_hook(self, module, input, output):
        self.features = output

    def logits(self, batchdata):
        return self.model(batchdata)

    def cams(self, batchdata, classes = None):
        if classes == None:
            classes = torch.zeros(batchdata.shape[0],dtype=int)
        logits = self.model(batchdata)

        self.model.zero_grad()
        self.gradients = torch.autograd.grad(logits[:,classes].sum(), self.activations,retain_graph=True)[0]
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)
        maxes = cam.view([-1,49]).max(1,keepdim=True)[0]
        cams = (cam.view([-1,49]) / maxes).view([-1,7,7])

        return cams
        
    def forward(self, batchdata, classes = None):
        # Return Features
        self.model(batchdata)
        return self.features

# for Fixing the seed
# torch.manual_seed(0)
#
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
#
# np.random.seed(0)

from dataset_Loader_cam import datasetLoader
map_size = 7
im_size = 224
class_assgn = {'Real':0}
dataseta = datasetLoader(args.csvPath,args.datasetPath,train_test='train',c2i=class_assgn,map_location=args.heatmaps,map_size=map_size,im_size=im_size,network=args.network)
dl = torch.utils.data.DataLoader(dataseta, batch_size=run_parameters['train_batch_size'], shuffle=True, num_workers=0, pin_memory=True)
dataset = datasetLoader(args.csvPath,args.datasetPath, train_test='test', c2i=dataseta.class_to_id,map_location=args.heatmaps,map_size=map_size,im_size=im_size,network=args.network)
test = torch.utils.data.DataLoader(dataset, batch_size=run_parameters['train_batch_size'], shuffle=True, num_workers=0, pin_memory=True)
dataloader = {'train': dl, 'test': test}


# load model
face_model = VGG()

# set train mode
face_model.train()

spoof_classifier = spoof_model(run_parameters['dimension'])
spoof_classifier.train()

if run_parameters['multi_gpu']:
    face_model = nn.DataParallel(face_model)
    spoof_classifier = nn.DataParallel(spoof_classifier)


if torch.cuda.is_available():
    face_model.cuda()
    spoof_classifier.cuda()

vgg_optim = optim.Adam(face_model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)
spoof_optim = optim.Adam(spoof_classifier.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-5)

criterion = nn.CrossEntropyLoss()
criterion.cuda()
criterion_hmap = nn.MSELoss(reduction='mean')
criterion_hmap.cuda()

best_test = 0
f = 0
for ep in range(run_parameters['epoch']):
    for phase in ['train','test']:
        batch_accuracy = 0        
    
        for  batch_idx, (data, cls, imageName, hmap) in enumerate(dataloader[phase]):
            data = data.cuda()

            if phase=='test':
                cls = cls.cuda()
                face_model.eval()
                spoof_classifier.eval()
                features = face_model(data)
                logits = spoof_classifier(features)
                predictions = torch.max(logits,dim=1)[1]
                correct = torch.sum((predictions == cls).int())
                batch_accuracy += correct/data.shape[0]
            else:
            
                hmap = hmap.cuda()
    
    
                face_model.train()
                spoof_classifier.train()
                vgg_optim.zero_grad()
                spoof_optim.zero_grad()
                
                features = face_model(data)
    
                cams = face_model.cams(data)
                salience_comp = criterion_hmap(cams,hmap)
        
                # spoof work
                if run_parameters['white_noise']:
                    # push from origin
                    sampler = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(run_parameters['dimension']).cuda(),  \
                                                                                         run_parameters['std_dev'] * torch.eye(run_parameters['dimension']).cuda())
                else:
                    # push from shifted mean cluster
                    if batch_idx == 0:
                        old_mean = torch.zeros(run_parameters['dimension']).cuda()
                    else:
                        old_mean = mean_vector
                    mean_vector = torch.mean(features, axis=0)
                    # like running avg
                    new_mean = run_parameters['alpha'] * old_mean + (1 - run_parameters['alpha']) * mean_vector
        
                    if f == 1:
                        save_name = os.path.join('models', args.exp_name, 'mean_vector.npy')
                        scipy.io.savemat(save_name, {'mean':new_mean})
                        f = 0
                    sampler = torch.distributions.multivariate_normal.MultivariateNormal(new_mean,  \
                                                                                         run_parameters['std_dev'] * torch.eye(run_parameters['dimension']).cuda())
                # sample from pseudo-negative gaussian
                noise = sampler.sample((features.shape[0],))
                noise = noise.cuda()
                spoof_input = torch.cat([features, noise], dim=0)
        
                spoof_output = spoof_classifier(spoof_input)
        
                spoof_label = torch.cat([torch.zeros(features.shape[0]), torch.ones(noise.shape[0])], dim=0)
                spoof_label = spoof_label.cuda()
                spoof_label = spoof_label.long()
                    
                # Spoof Loss for classifier
                spoof_loss = criterion(spoof_output, spoof_label)
        
    
                loss = run_parameters['lambda1'] * salience_comp + \
                       run_parameters['lambda2'] * spoof_loss
        
                loss.backward()
        
                vgg_optim.step()
                spoof_optim.step()
                if batch_idx % 10 == 0:
                    print("Epoch: {0}, Salience Loss: {1}, Spoof Loss: {2}, Total Loss: {3}".format(ep,salience_comp.item(), spoof_loss.item(), loss.item()))
    states = {
        'epoch': ep + 1,
        'face_state_dict': face_model.state_dict(),
        'face_optimizer': vgg_optim.state_dict(),
        'spoof_state_dict': spoof_classifier.state_dict(),
        'spoof_optimizer': spoof_optim.state_dict()
    }
    if phase == 'train':
        torch.save(states, os.path.join(args.outputPath,f'current_model.pth'))
    if phase == 'test':
        accuracy = batch_accuracy/(batch_idx+1)
        if accuracy > best_test:
            torch.save(states, os.path.join(args.outputPath, f'best_model.pth'))

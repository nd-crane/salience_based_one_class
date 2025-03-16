import sys
import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import glob
import os
import csv
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
sys.path.append("../")
import random
import pdb
import torchvision
from vgg_face_dag import vgg_face_dag, spoof_model2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    device = torch.device('cpu')
    parser.add_argument('-imageFolder', default='/project01/cvrl/datasets/CYBORG_centered_face/',type=str)
    parser.add_argument('-modelPath',  default='./models/current_model.pth',type=str)
    parser.add_argument('-csv',  default="/project01/cvrl/datasets/CYBORG_centered_face/csvs/original.csv",type=str)
    parser.add_argument('-output_dir',  default="./models/",type=str)
    parser.add_argument('-output_filename', default="original_densenet_attempt1_centered_1.csv",type=str)
    parser.add_argument('-fakeToken', default='Spoof', type=str)
    parser.add_argument('-nClasses',default=2,type=int)
    parser.add_argument('-imageSize',default=224,type=int)
    args = parser.parse_args()

    os.makedirs(args.output_dir,exist_ok=True)

    # file exist?
    #if os.path.exists(os.path.join(args.output_dir,args.output_filename)):
    #    sys.exit("File exists: " + args.output_filename)  

    # seed random
    random.seed(1234)

    # Load weights of single binary DesNet121 model
    states = torch.load(args.modelPath,map_location=torch.device('cpu'))
    face_model = torchvision.models.densenet121(pretrained=True)
    num_ftrs = face_model.classifier.in_features
    face_model.classifier = nn.Linear(num_ftrs, args.nClasses)
    face_model.load_state_dict(states['face_state_dict'])
    face_model = face_model.to(device)
    face_model.eval()

    spoof_classifier = spoof_model2(1024)
    spoof_classifier.load_state_dict(states['spoof_state_dict'])
    spoof_classifier = spoof_classifier.to(device)
    spoof_classifier.eval()

    # Transformation specified for the pre-processing
    transform = transforms.Compose([
                transforms.Resize([args.imageSize, args.imageSize]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    imagesScores=[]
    sigmoid = nn.Sigmoid()
    
    imageCSV = open(args.csv,"r")
    for entry in tqdm(imageCSV):
        tokens = entry.split(",")
        if tokens[0] != 'test':
            continue

        label = tokens[1]

        upd_name = tokens[-1].replace("\n","")
        imgFile = args.imageFolder + upd_name

        # Read the image
        image = Image.open(imgFile).convert('RGB')
        # Image transformation
        tranformImage = transform(image)
        image.close()


        ## START COMMENTING HERE

        tranformImage = tranformImage[0:3,:,:].unsqueeze(0)
        tranformImage = tranformImage.to(device)

        # Output from single binary CNN model
        with torch.no_grad():
            features = face_model.features(tranformImage)
            input_features = nn.AdaptiveAvgPool2d((1, 1))(features).view([-1,1024])
            output = spoof_classifier(input_features)
            
            PAScore = sigmoid(output).detach().cpu().numpy()[:, 1]
            SMScore = nn.Softmax(dim=1)(output).detach().cpu().numpy()[:, 1]
            imagesScores.append([f'{label}--{imgFile}', PAScore[0], SMScore[0]])

    # Writing the scores in the csv file
    print('ATTEMPTING TO WRITE')
    with open(os.path.join(args.output_dir,args.output_filename),'w',newline='') as fout:
        writer = csv.writer(fout)
        writer.writerows(imagesScores)
        print('WRITE SUCCESSFUL')


import sys
import os

#from Utilities.SaveAnimation import Video

from druida import Stack
from druida import setup
from druida.DataManager import datamanager
from druida.tools import utils

import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optimizer

from torchsummary import summary
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torcheval.metrics import BinaryAccuracy
from torchvision.utils import save_image

from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Lambda
import torchvision.utils as vutils
from torch.autograd import Variable

import glob
from tqdm.notebook import tqdm
import argparse
import json
from PIL import Image
import cv2 
# Arguments
parser = argparse.ArgumentParser()
# boxImagesPath="\\data\\francisco_pizarro\\jorge-cardenas\\data\\MetasufacesData\\Images Jorge Cardenas 512\\"
# DataPath="\\data\\francisco_pizarro\\jorge-cardenas\\data\\MetasufacesData\\Exports\\output\\"
# simulationData="\\data\\francisco_pizarro\\jorge-cardenas\\data\\MetasufacesData\\DBfiles\\"

boxImagesPath="../../../data/MetasufacesData/Images-512-Bands/"
#boxImagesPath="../../../data/MetasufacesData/Images-512-Suband/"

DataPath="../../../data/MetasufacesData/Exports/output/"
simulationData="../../../data/MetasufacesData/DBfiles/"
validationImages="../../../data/MetasufacesData/testImages/"


Substrates={"Rogers RT/duroid 5880 (tm)":0, "other":1}
Materials={"copper":0,"pec":1}
Surfacetypes={"Reflective":0,"Transmissive":1}
TargetGeometries={"circ":0,"box":1, "cross":2}
Bands={"30-40":0,"40-50":1, "50-60":2,"60-70":3,"70-80":4, "80-90":5}



def arguments():

    parser.add_argument("run_name",type=str)
    parser.add_argument("epochs",type=int)
    parser.add_argument("batch_size",type=int)
    parser.add_argument("workers",type=int)
    parser.add_argument("gpu_number",type=int)
    parser.add_argument("device",type=str)
    parser.add_argument("learning_rate",type=float)
    parser.add_argument("condition_len",type=float) #This defines the length of our conditioning vector
    parser.add_argument("metricType",type=float) #This defines the length of our conditioning vector
    parser.add_argument("latent",type=int) #This defines the length of our conditioning vector
    parser.add_argument("spectra_length",type=int) #This defines the length of our conditioning vector

    parser.run_name = "GAN Training"
    parser.epochs = 2
    parser.batch_size = 1
    parser.workers=1
    parser.gpu_number=1
    parser.image_size = 64
    parser.dataset_path = os.path.normpath('/content/drive/MyDrive/Training_Data/Training_lite/')
    parser.device = "cpu"
    parser.learning_rate =5e-5
    parser.condition_len = 16 #Incliuding 3 top frequencies
    parser.metricType='AbsorbanceTM' #this is to be modified when training for different metrics.
    parser.latent=200 #this is to be modified when training for different metrics.
    parser.spectra_length=100 #this is to be modified when training for different metrics.

    categories=["box", "circle", "cross"]


#From the DCGAN paper, the authors specify that all model weights shall be randomly initialized
#from a Normal distribution with mean=0, stdev=0.02.
#The weights_init function takes an initialized model as input and reinitializes all convolutional,
#convolutional-transpose, and batch normalization layers to meet this criteria.

# Data pre-processing
def join_simulationData():
    df = pd.DataFrame()
    for file in glob.glob(simulationData+"*.csv"): 
        df2 = pd.read_csv(file)
        df = pd.concat([df, df2], ignore_index=True)
    
    df.to_csv('out.csv',index=False)
    


def prepare_data(names, device,df,classes,classes_types):
    bands_batch=[]
    array1 = []
    array2 = []
    noise = torch.Tensor()

    substrate_encoder=encoders(Substrates)
    materials_encoder=encoders(Materials)
    surfaceType_encoder=encoders(Surfacetypes)
    TargetGeometries_encoder=encoders(TargetGeometries)
    bands_encoder=encoders(Bands)

    for idx,name in enumerate(names):

        series=name.split('_')[-2]#
        band_name=name.split('_')[-1].split('.')[0]#
        batch=name.split('_')[4]


        for file_name in glob.glob(DataPath+batch+'/files/'+'/'+parser.metricType+'*'+series+'.csv'): 
            #loading the absorption data
            train = pd.read_csv(file_name)

            # # the band is divided in chunks 
            if Bands[str(band_name)]==0:
                
                train=train.loc[1:100]

            elif Bands[str(band_name)]==1:
                
                train=train.loc[101:200]

            elif Bands[str(band_name)]==2:
                
                train=train.loc[201:300]

            elif Bands[str(band_name)]==3:
                
                train=train.loc[301:400]

            elif Bands[str(band_name)]==4:
                
                train=train.loc[401:500]

            elif Bands[str(band_name)]==5:

                train=train.loc[501:600]
            
            values=np.array(train.values.T)
            values=np.around(values, decimals=2, out=None)

            all_values=torch.from_numpy(values[1])
            all_frequencies=torch.from_numpy(values[0])
            _, indices  = torch.topk(all_values, 3)

            top_frequencies = all_frequencies[indices]

            conditional_data = set_conditioning_one_hot(df,
                                                name,
                                                classes[idx],
                                                classes_types[idx],
                                                Bands[str(band_name)],
                                                top_frequencies,
                                                substrate_encoder,
                                                materials_encoder,
                                                surfaceType_encoder,
                                                TargetGeometries_encoder,
                                                bands_encoder)


            bands_batch.append(band_name)

            #loading data to tensors for discriminator
            tensorA = torch.from_numpy(values[1])
            array2.append(tensorA.to(device)) #concat por batches

            latent_tensor=torch.rand(parser.latent)
            tensor1 = torch.cat((conditional_data.to(device),tensorA.to(device),latent_tensor.to(device),)) #concat side
            #un vector que inclue
            #datos desde el dataset y otros datos aleatorios latentes.

            """No lo veo tan claro pero es necesario para pasar los datos al ConvTranspose2d"""
            tensor2 = tensor1.unsqueeze(1).unsqueeze(1).unsqueeze(1).to(device)
            tensor3 = tensor2.permute(1,0,2,3)
            noise = torch.cat((noise.to(device),tensor3.to(device)),0)

            array1.append(tensor1.to(device))
        
    print(tensor1.shape)

    return array1, array2, noise,tensor1, values

def set_conditioning_one_hot(df,name,target,categories,band_name,top_freqs,substrate_encoder,materials_encoder,surfaceType_encoder,TargetGeometries_encoder,bands_encoder):
    series=name.split('_')[-2]
    batch=name.split('_')[4]
    iteration=series.split('-')[-1]
    row=df[(df['sim_id']==batch) & (df['iteration']==int(iteration))  ]
        #print(batch)
        #print(iteration)

    target_val=target
    category=categories
    band=band_name

    """"
    surface type: reflective, transmissive
    layers: conductor and conductor material / Substrate information
    """
    surfacetype=row["type"].values[0]
        
    layers=row["layers"].values[0]
    layers= layers.replace("'", '"')
    layer=json.loads(layers)
        
        
    if (target_val==2): #is cross. Because an added variable to the desing 
        
        sustratoHeight= json.loads(row["paramValues"].values[0])
        sustratoHeight= sustratoHeight[-2]
    else:
    
        sustratoHeight= json.loads(row["paramValues"].values[0])
        sustratoHeight= sustratoHeight[-1]
        
    materialsustrato=torch.Tensor(substrate_encoder.transform(np.array(Substrates[layer['substrate']['material']]).reshape(-1, 1)).toarray()).squeeze(0)
    materialconductor=torch.Tensor(materials_encoder.transform(np.array(Materials[layer['conductor']['material']]).reshape(-1, 1)).toarray()).squeeze(0)
    surface=torch.Tensor(surfaceType_encoder.transform(np.array(Surfacetypes[surfacetype]).reshape(-1, 1)).toarray()).squeeze(0)
    band=torch.Tensor(bands_encoder.transform(np.array(band).reshape(-1, 1)).toarray()).squeeze(0)
  

    """[ 1.0000,  0.0000,  1.0000,  0.0000,  1.0000,  0.0000,  0.2520,  0.0000,
         0.0000,  1.0000,  0.0000,  0.0000,  0.0000, 56.8000, 56.7000, 56.6000]
         surface,materialconductor,materialsustrato,torch.Tensor([sustratoHeight]),band,top_freqs
         """

    values_array = torch.cat((surface,materialconductor,materialsustrato,torch.Tensor([sustratoHeight]),band,top_freqs),0) #concat side
    """ Values array solo pouede llenarse con n+umero y no con textos"""
    # values_array = torch.Tensor(values_array)
    return values_array


def set_conditioning(df,name,target,categories,band_name,top_freqs):
    series=name.split('_')[-2]
    batch=name.split('_')[4]
    iteration=series.split('-')[-1]
    row=df[(df['sim_id']==batch) & (df['iteration']==int(iteration))  ]
        #print(batch)
        #print(iteration)

    target_val=target
    category=categories
    geometry=TargetGeometries[category]
    band=band_name

    """"
    surface type: reflective, transmissive
    layers: conductor and conductor material / Substrate information
    """
    surfacetype=row["type"].values[0]
    surfacetype=Surfacetypes[surfacetype]
        
    layers=row["layers"].values[0]
    layers= layers.replace("'", '"')
    layer=json.loads(layers)
        
    materialconductor=Materials[layer['conductor']['material']]
    materialsustrato=Substrates[layer['substrate']['material']]
        
        
    if (target_val==2): #is cross. Because an added variable to the desing 
        
        sustratoHeight= json.loads(row["paramValues"].values[0])
        sustratoHeight= sustratoHeight[-2]
    else:
    
        sustratoHeight= json.loads(row["paramValues"].values[0])
        sustratoHeight= sustratoHeight[-1]
        


    values_array=torch.Tensor([geometry,surfacetype,materialconductor,materialsustrato,sustratoHeight,band])
    
    values_array = torch.cat((values_array,top_freqs),0) #concat side

    """ Values array solo pouede llenarse con n+umero y no con textos"""
    # values_array = torch.Tensor(values_array)
    return values_array



def test(netG,device):
    
    df = pd.read_csv("out.csv")
    
    vdataloader = utils.get_data_with_labels(parser.image_size, parser.image_size,1,
                                            validationImages,parser.batch_size, 
                                            drop_last=True,
                                            filter="30-40")
    for i, data in enumerate(vdataloader, 0):
        # Genera el batch del espectro, vectores latentes, and propiedades
        # Estamos Agregando al vector unas componentes condicionales
        # y otras de ruido en la parte latente  .

        inputs, classes, names, classes_types = data
        #sending to CUDA
        inputs = inputs.to(device)
        classes = classes.to(device)
        _, _, noise,tensor1,real_values = prepare_data(names, device,df,classes,classes_types)


        testTensor = noise.type(torch.float).to(device)


        ##Eval

        fake = netG(testTensor).detach().cpu()

        t_np = torch.squeeze(noise,(2,3)).cpu().numpy() #convert to Numpy array
        df_ = pd.DataFrame(t_np) #convert to a dataframe
        df_real = pd.DataFrame(real_values)
        df_.to_csv("validation_tests/validate",index=True) #save to file
        df_real.to_csv("validation_tests/realValues"+str(classes_types)+"_"+str(names),index=True) #save to file

        plt.plot(real_values[0], real_values[1])
         # Saving figure by changing parameter values
        plt.savefig("validation_tests/realValues"+str(classes_types)+"_"+str(names)+".png")

        
        # upscaling
        save_image(fake, "validation_tests/"+str(classes_types)+"_"+str(names)+'.png')

        image = cv2.imread("validation_tests/"+str(classes_types)+"_"+str(names)+'.png')
        r = 512.0 / image.shape[1]
        dim = (512, int(image.shape[0] * r))
        # perform the actual resizing of the image using cv2 resize
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite("validation_tests/"+str(classes_types)+"_"+str(names)+'.png', resized) 

        #plot Data

        break

def testwithLabels(netG,device):
    
    #geometry,surfacetype,materialconductor,materialsustrato,sustratoHeight,band
    
    labels=np.array([ 1.0000,  0.0000,  1.0000,  0.0000,  1.0000,  0.0000,  0.7870,  0.0000, 0.0000,  1.0000,  0.0000,  0.0000,  0.0000, 53.6000, 52.1000, 53.5000,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.5,0.8,0.9,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

    """
         surface,materialconductor,materialsustrato,torch.Tensor([sustratoHeight]),band,top_freqs
    """
    """
    "Reflective","Transmissive"
    "copper","pec"
    Rogers RT/duroid 5880 (tm)","other"
    height
    "30-40","40-50", "50-60","60-70","70-80", "80-90"
    top 3 freqs in such band
    """
    
    latent_tensor=torch.rand(parser.latent)
    labels_tensor=torch.from_numpy(labels)
    tensor1 = torch.cat((labels_tensor.to(device),latent_tensor.to(device),)) #concat side
          
    testTensor = tensor1.type(torch.float).to(device)

    tensor2 = testTensor.unsqueeze(1).unsqueeze(1).unsqueeze(1).to(device)
    tensor3 = tensor2.permute(1,0,2,3)
    

    fake = netG(tensor3).detach().cpu()


    t_np = torch.squeeze(tensor3,(2,3)).cpu().numpy() #convert to Numpy array
    df_ = pd.DataFrame(t_np) #convert to a dataframe

    df_.to_csv("testfile_labels",index=True) #save to file

    # let's resize our image to be 150 pixels wide, but in order to
    # prevent our resized image from being skewed/distorted, we must
    # first calculate the ratio of the *new* width to the *old* width 
    save_image(fake, str("withLabels")+"_"+str("64")+'.png')

    image = cv2.imread(str("withLabels")+"_"+str("64")+'.png')

    r = 512.0 / image.shape[1]
    dim = (512, int(image.shape[0] * r))
    # perform the actual resizing of the image using cv2 resize
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite("Test_512.png", resized) 


def encoders(dictionary):
    index = []
    for x,y in dictionary.items():
        index.append([y])

    index = np.asarray(index)
    enc = OneHotEncoder()
    enc.fit(index)
    return enc


    


def main():

    os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    arguments()
    join_simulationData()  

    trainer = Stack.Trainer(parser)

    input_size=parser.spectra_length+parser.condition_len+parser.latent
    generator_mapping_size=parser.image_size
    output_channels=3

    netG = Stack.Generator(trainer.gpu_number, input_size, generator_mapping_size, output_channels)
    netG.load_state_dict(torch.load('NETGModelTM_abs__GAN_Bands_30Abr_50e-5_50epc_64_16conds_onehot_.pth'))
    netG.eval()
    netG.cuda()

    discriminator_mapping_size=64
#depth of feature maps propagated through the discriminator
    channels=3
    netD = Stack.Discriminator(parser.condition_len+parser.spectra_length,trainer.gpu_number, parser.image_size, discriminator_mapping_size, channels)
    netD.load_state_dict(torch.load('NETDModelTM_abs__GAN_Bands_30Abr_50e-5_50epc_64_16conds_onehot_.pth'))
    netD.eval()
    netD.cuda()
    
    print(netD)
    print(netG)

    criterion = nn.BCELoss()
    # Setup Adam optimizers for both G and D

    test(netG,device)
    #testwithLabels(netG,device)

if __name__ == "__main__":
    main()
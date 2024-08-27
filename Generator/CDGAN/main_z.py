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
#Bands={"30-40":[1,0,0,0,0,0],"40-50":[0,1,0,0,0,0], 
#       "50-60":[0,0,1,0,0,0],"60-70":[0,0,0,1,0,0],"70-80":[0,0,0,0,1,0], 
#       "80-90":[1,0,0,0,0,1]}


def arguments():

    parser.add_argument("run_name",type=str)
    parser.add_argument("epochs",type=int)
    parser.add_argument("batch_size",type=int)
    parser.add_argument("workers",type=int)
    parser.add_argument("gpu_number",type=int)
    parser.add_argument("output_channels",type=int)
    parser.add_argument("discriminator_channels",type=int)
    parser.add_argument("device",type=str)
    parser.add_argument("learning_rate",type=float)
    parser.add_argument("condition_len",type=float) #This defines the length of our conditioning vector
    parser.add_argument("metricType",type=float) #This defines the length of our conditioning vector
    parser.add_argument("latent",type=int) #This defines the length of our conditioning vector
    parser.add_argument("spectra_length",type=int) #This defines the length of our conditioning vector
    parser.add_argument("output_folder",type=str)

    parser.run_name = "GAN Training"
    parser.epochs = 200
    parser.batch_size = 56
    parser.workers=1
    parser.gpu_number=0
    parser.output_channels=3
    parser.discriminator_channels=3
    parser.image_size = 64
    parser.dataset_path = os.path.normpath('/content/drive/MyDrive/Training_Data/Training_lite/')
    parser.device = "cpu"
    parser.learning_rate =1e-4
    parser.condition_len = 7 #Incliuding 3 top frequencies con one-hot-encoding
    parser.metricType='AbsorbanceTM' #this is to be modified when training for different metrics.
    parser.latent=107 #this is o be modified when training for different metrics.
    parser.spectra_length=100 #this is to be modified when training for different metrics.
    parser.output_folder="output_zprod_107_3/"

    categories=["box", "circle", "cross"]


#From the DCGAN paper, the authors specify that all smodel weights shall be randomly initialized
#from a Normal distribution with mean=0, stdev=0.02.
#The weights_init function takes an initialized model as input and reinitializes all convolutional,
#convolutional-transpose, and batch normalization layers to meet this criteria.

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Data pre-processing
def join_simulationData():
    df = pd.DataFrame()
    for file in glob.glob(simulationData+"*.csv"): 
        df2 = pd.read_csv(file)
        df = pd.concat([df, df2], ignore_index=True)
    
    df.to_csv('out.csv',index=False)
    

def train(opt_D,opt_G, schedulerD,schedulerG,criterion,netD,netG,device,PATH ,substrate_encoder, materials_encoder,surfaceType_encoder,TargetGeometries_encoder,bands_encoder):

    # Lists to keep track of progress
    img_list,G_losses,D_losses,real_scores,fake_scores,iter_array= [],[],[],[],[],[]

    iters = 0

    # convenciones sobre algo real o fake
    # This is required for the discriminator training
    real_label = random.uniform(0.9,1.0)
    fake_label = 0

    # Load training data set
    df = pd.read_csv("out.csv")
    

    dataloader = utils.get_data_with_labels(parser.image_size,parser.image_size,1, 
                                            boxImagesPath,parser.batch_size,
                                            drop_last=True,
                                            filter="30-40")#filter disabled
    
    vdataloader = utils.get_data_with_labels(parser.image_size, parser.image_size,1,
                                            validationImages,parser.batch_size, 
                                            drop_last=False,
                                            filter="30-40")

    for epoch in range(parser.epochs):
        
        
        # For each batch in the dataloader
        netG.train()

        for i, data in enumerate(dataloader, 0):
            # Genera el batch del espectro, vectores latentes, and propiedades
            # Estamos Agregando al vector unas componentes condicionales
            # y otras de ruido en la parte latente  .

            inputs, classes, names, classes_types = data
            
            #sending to CUDA
            inputs = inputs.to(device) #images 
            classes = classes.to(device) #classes
            
            """Prepare Data"""
            _, labels, noise,_ = prepare_data(names, device,df,classes,classes_types,
                                                             substrate_encoder,
                                                             materials_encoder,
                                                             surfaceType_encoder,
                                                             TargetGeometries_encoder,
                                                             bands_encoder)

            noise = noise.type(torch.float).to(device) #Generator input espectro+ruido
            conditions = torch.stack(labels).type(torch.float).to(device) #Discrminator Conditioning Espectro
            
            #noise = torch.nn.functional.normalize(noise, p=2.0, dim=1, eps=1e-5, out=None)

            label = torch.full((parser.batch_size,), real_label,dtype=torch.float, device=device)
            #label_real = torch.full((parser.batch_size,), real_label,dtype=torch.float, device=device)

            # Train discriminator
            loss_d,  D_x, D_G_z1, fakes = train_discriminator(netD,netG,criterion,inputs, opt_D, conditions,noise, label, parser.batch_size,real_label,fake_label)

            # Train generator
            loss_g, D_G_z2  = train_generator(opt_G,netG, netD,parser.batch_size,criterion,fakes,conditions, label,real_label)

            # Record losses & scores
            G_losses.append(loss_g)
            D_losses.append(loss_d)
            real_scores.append(D_x)
            fake_scores.append(D_G_z1)
            iter_array.append(iters)

            # Log losses & scores (last batch)
            if i % 50 == 0:
               print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, D(G(z)): {:.4f} / {:.4f}".format(
                    epoch+1, parser.epochs, loss_g, loss_d, D_x, D_G_z1,D_G_z2))
            

            # Validation by generating images
            # taking data from validation DS to generate an equivalent image
            if (iters % 1000 == 0) or ((epoch == parser.epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():


                    testTensor = torch.Tensor().to(device)
                        
                    _,data_val = list(enumerate(vdataloader))[0]
                    _, classes_val, names_val, classes_types_val = data_val

                    _, _, noise_val,_ = prepare_data(names_val, device,df,classes_val,classes_types_val,
                                                                                 substrate_encoder,
                                                                                 materials_encoder,
                                                                                 surfaceType_encoder,
                                                                                 TargetGeometries_encoder,
                                                                                 bands_encoder)
                    
                    testTensor = noise_val.type(torch.float).to(device)
                    #testTensor = torch.nn.functional.normalize(testTensor, p=2.0, dim=1, eps=1e-5, out=None)

                    fake = netG(testTensor).detach().cpu()

                    """Saving Data"""

                    if not os.path.exists(parser.output_folder):
                        os.makedirs(parser.output_folder)


                    save_image(fake, parser.output_folder+str(epoch)+"_"+str(iters)+'.png')

                img_list.append(vutils.make_grid(fake,nrow=10, padding=2, normalize=True))

            iters += 1

        if epoch % 10 == 0:
            ##Guarda el modelo en el directorio cada 50 epocas
            if not os.path.exists(parser.output_folder+'/model'):
                os.makedirs(parser.output_folder+'/model')

            torch.save(netG, parser.output_folder+'/model' + 'netG' + str(epoch) + '.pt')
            torch.save(netD, parser.output_folder+'/model' + 'netD' + str(epoch) + '.pt')
    
        schedulerD.step()
        schedulerG.step()
    
    return G_losses,D_losses,iter_array,real_scores,fake_scores
            

def prepare_data(names, device,df,classes,classes_types,substrate_encoder, materials_encoder,surfaceType_encoder,TargetGeometries_encoder,bands_encoder):
    bands_batch,array1,array2=[],[],[]

    noise = torch.Tensor()

    for idx,name in enumerate(names):

        series=name.split('_')[-2]#
        band_name=name.split('_')[-1].split('.')[0]#
        batch=name.split('_')[4]
        version_batch=1
        if batch=="v2":
            version_batch=2
            batch=name.split('_')[5]

       
        for file_name in glob.glob(DataPath+batch+'/files/'+'/'+parser.metricType+'*'+series+'.csv'): 
            #loading the absorption data
            train = pd.read_csv(file_name)

            # # the band is divided in chunks 
            if Bands[str(band_name)]==0:
                
                train=train.loc[1:100]

            elif Bands[str(band_name)]==1:
                
                train=train.loc[101:200]

            elif Bands[str(band_name)]==2:
                if version_batch==1:
                    train=train.loc[201:300]
                else:
                    train=train.loc[1:100]
            elif Bands[str(band_name)]==3:
                if version_batch==1:
                    train=train.loc[301:400]
                else:
                    train=train.loc[101:200]

            elif Bands[str(band_name)]==4:
                if version_batch==1: 
                    train=train.loc[401:500]
                else:
                    train=train.loc[201:300]

            elif Bands[str(band_name)]==5:

                train=train.loc[501:600]
            
            #getting the spectra
            values=np.array(train.values.T)
            values=np.around(values, decimals=3, out=None)

            #getting the freqency points under test
            all_values=torch.from_numpy(values[1])
            all_frequencies=torch.from_numpy(values[0])

            # top 3 values
            #_, indices  = torch.topk(all_values, 3)
            #top_frequencies = all_frequencies[indices]

            """12 conditiosn"""
            # conditional_data = set_conditioning_one_hot(df,
            #                                     name,
            #                                     classes[idx],
            #                                     classes_types[idx],
            #                                     Bands[str(band_name)],
            #                                     top_frequencies,
            #                                     substrate_encoder,
            #                                     materials_encoder,
            #                                     surfaceType_encoder,
            #                                     TargetGeometries_encoder,
            #                                     bands_encoder)


            """6 conditions no one-hot-encoding"""
            conditional_data = set_conditioning(df,name,classes[idx],
                                                classes_types[idx],
                                                Bands[str(band_name)],
                                                None)


            bands_batch.append(band_name)


            #loading data to tensors for discriminator
            tensorA = torch.from_numpy(values[1]) #Just have spectra profile
            labels = torch.cat((conditional_data.to(device),tensorA.to(device))) #concat side
            array2.append(labels) # to create stack of tensors
                                            
            latent_tensor=torch.rand(parser.latent)

            """multiply noise and labels to get a single vector"""
            tensor1=torch.mul(labels.to(device),latent_tensor.to(device) )
            """concat noise and labels adjacent"""
            #tensor1 = torch.cat((conditional_data.to(device),tensorA.to(device),latent_tensor.to(device),)) #concat side

            #preparing noise in the right format to be sent for generation
            tensor2 = tensor1.unsqueeze(1).unsqueeze(1).unsqueeze(1).to(device)
            tensor3 = tensor2.permute(1,0,2,3)
            noise = torch.cat((noise.to(device),tensor3.to(device)),0)

            array1.append(tensor1.to(device))

    return array1, array2, noise,bands_batch

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

    values_array = torch.cat((surface,materialconductor,materialsustrato,torch.Tensor([sustratoHeight])),0) #concat side
    """ Values array solo pouede llenarse con n+umero y no con textos"""
    # values_array = torch.Tensor(values_array)
    return values_array

def set_conditioning(df,name,target,categories,band_name,top_freqs):

    #splitting file names to get some parameters
    series=name.split('_')[-2]
    batch=name.split('_')[4]
    if batch=="v2":
        batch=name.split('_')[5]
    
    iteration=series.split('-')[-1]
    row=df[(df['sim_id']==batch) & (df['iteration']==int(iteration))  ]

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
        substrateWidth = json.loads(row["paramValues"].values[0])[-1] # from the simulation crosses have this additional free param
    else:
    
        sustratoHeight= json.loads(row["paramValues"].values[0])
        sustratoHeight= sustratoHeight[-1]
        substrateWidth = 5 # 5 mm size
        

    values_array=torch.Tensor([geometry,surfacetype,materialconductor,materialsustrato,sustratoHeight,substrateWidth,band ])
    #values_array = torch.cat((values_array,torch.Tensor(band)),0)
    """if wanting to add top frequencies to the conditions"""
    #values_array = torch.cat((values_array,top_freqs),0) #concat side

    """ Values array solo pouede llenarse con n+umero y no con textos"""
    # values_array = torch.Tensor(values_array)
    return values_array


def train_discriminator(modelD,modelG,criterion,real_images, opt_d,conditions, generator_noise, label,batch_size,real_label,fake_label):

    # Clear discriminator gradients
    #opt_d.zero_grad()
    modelD.zero_grad()
     # Forward pass del batch real a través de NetD
     #noise just creates added channels conditioning the real image

    label.fill_(real_label)
    #if random.uniform(0.0,1)<0.1:
    #    label.fill_(fake_label)

    output = modelD.forward(real_images,conditions,batch_size).view(-1)

    # Calcula la perdida de all-real batch
    errD_real = criterion(output, label)
    # Calcula el gradients para NetD en backward pass
    errD_real.backward()
    D_x = output.mean().item()


    ## Entrenamiento con all-fake batch
    # Genera un batch de imagenes falsas con NetG
    fake = modelG(generator_noise)
    
    label.fill_(fake_label)
    if random.uniform(0.0,1)<0.1:
        label.fill_(real_label)

    # Clasifica todos los batch falsos con NetD
    output2 = modelD.forward(fake.detach(),conditions, batch_size).view(-1)

    # Calcula la perdida de NetD durante el btach de imagenes falsas
    errD_fake = criterion(output2, label)
    # Calcula el gradiente para este batch
    errD_fake.backward()

    D_G_z1 = output2.mean().item()
    # Se suman los gradientes de los batch all-real y all-fake

    errD = errD_real + errD_fake
    # Se actualiza NetD con la optimizacion
    opt_d.step()

    return errD.item(), D_x, D_G_z1, fake

def train_generator(opt_g,net_g, net_d,batch_size,criterion,fakes,noise2, label,real_label):
    # Clear generator gradients
    net_g.zero_grad()
    # Generate fake images
    label.fill_(real_label)
    output = net_d.forward(fakes,noise2, batch_size).view(-1)

    # Calcula la perdida de NetG basandose en este output
    errG = criterion(output, label)
    # Calcula los gradientes de NetG
    errG.backward()
    D_G_z2 = output.mean().item()
    # Actualiza NetG
    opt_g.step()

    return errG.item(), D_G_z2

def encoders(dictionary):
    index = []
    for x,y in dictionary.items():
        index.append([y])

    index = np.asarray(index)
    enc = OneHotEncoder()
    enc.fit(index)
    return enc

def main():

    # Get available devices
    os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # prepare settings and dataset
    arguments()
    join_simulationData()
    

    #one hot encoders in case needed
    # substrate_encoder=encoders(Substrates)
    # materials_encoder=encoders(Materials)
    # surfaceType_encoder=encoders(Surfacetypes)
    # TargetGeometries_encoder=encoders(TargetGeometries)
    # bands_encoder=encoders(Bands)

    # Trainer object (look something to reasses)
    trainer = Stack.Trainer(parser)

    # Sizes for discrimnator and generator
    """Z product"""
    input_size=parser.spectra_length+parser.condition_len
    
    """this for Z concat"""
    #input_size=parser.spectra_length+parser.condition_len+parser.latent

    #
    generator_mapping_size=64

    # Create models
    netG = Stack.Generator(trainer.gpu_number, 
                           input_size,
                            generator_mapping_size, 
                            parser.output_channels,
                            leakyRelu_flag=False)

    netG.apply(weights_init)
    netG.cuda()


    #depth of feature maps propagated through the discriminator

    discriminator_mapping_size=32

    #netD = Stack.Discriminator(parser.condition_len+parser.spectra_length,trainer.gpu_number, parser.image_size, discriminator_mapping_size, parser.discriminator_channels)
    netD = Stack.Discriminator(parser.condition_len+parser.spectra_length,trainer.gpu_number, parser.image_size, discriminator_mapping_size, parser.discriminator_channels)

    netD.cuda()
    netD.apply(weights_init)
    
    print(netD)
    print(netG)

    #Binary cross entropy for Discriminator
    criterion = nn.BCELoss()

    # Setup Adam optimizers for both G and D
    opt_D = optimizer.Adam(netD.parameters(), lr=trainer.learning_rate, betas=(0.7, 0.999))
    #opt_D = optimizer.SGD(netD.parameters(), lr=trainer.learning_rate, momentum=0.7)
    opt_G = optimizer.Adam(netG.parameters(), lr=trainer.learning_rate, betas=(0.5, 0.999))
    schedulerD = torch.optim.lr_scheduler.ExponentialLR(opt_D, gamma=1.00004)
    schedulerG = torch.optim.lr_scheduler.ExponentialLR(opt_G, gamma=1.00004)


    #naming the output file
    date="_GAN_Bands_26Ag_200epc_64_7conds_prod_Adam"

    G_losses,D_losses,iter_array,_,_=train(opt_D,opt_G,schedulerD,schedulerG,
                                                            criterion,
                                                            netD,netG,
                                                            device,
                                                            date,
                                                            None,
                                                            None,
                                                            None,
                                                            None,
                                                            None )

    torch.save(netD.state_dict(), 'NETDModelTM_abs_'+date+'.pth')
    torch.save(netG.state_dict(), 'NETGModelTM_abs_'+date+'.pth')

    try:
        np.savetxt(parser.output_folder+'loss_Train_TM_NETG_'+date+'.out', G_losses, delimiter=',')
    except:
        np.savetxt(parser.output_folder+'loss_Train_TM_NETG_'+date+'.out', [], delimiter=',')

    try:
        np.savetxt(parser.output_folder+'acc_Train_TM_NETD_'+date+'.out', D_losses, delimiter=',')
    except:
        np.savetxt(parser.output_folder+'acc_Train_TM_NETD_'+date+'.out', [], delimiter=',')
    
    try:
        np.savetxt(parser.output_folder+'loss_Valid_TM_iterArray'+date+'.out', iter_array, delimiter=',')
    except:
        np.savetxt(parser.output_folder+'loss_Valid_TM_iterArray'+date+'.out', [], delimiter=',')


if __name__ == "__main__":
    main()
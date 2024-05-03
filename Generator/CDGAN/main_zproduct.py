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
    parser.epochs = 100
    parser.batch_size = 50
    parser.workers=1
    parser.gpu_number=1
    parser.image_size = 64
    parser.dataset_path = os.path.normpath('/content/drive/MyDrive/Training_Data/Training_lite/')
    parser.device = "cpu"
    parser.learning_rate =5e-5
    parser.condition_len = 6 #Incliuding 3 top frequencies con one-hot-encoding
    parser.metricType='AbsorbanceTM' #this is to be modified when training for different metrics.
    parser.latent=200 #this is to be modified when training for different metrics.
    parser.spectra_length=100 #this is to be modified when training for different metrics.

    categories=["box", "circle", "cross"]


#From the DCGAN paper, the authors specify that all model weights shall be randomly initialized
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
    img_list = []
    G_losses = []
    D_losses = []
    real_scores = []
    fake_scores = []
    iter_array = []
    array2 = []
    iters = 0

    #convenciones sobre algo real o fake
    real_label = random.uniform(0.9,1.0)
    fake_label = 0
# For each epoch
    df = pd.read_csv("out.csv")
    

    dataloader = utils.get_data_with_labels(parser.image_size,parser.image_size,1, 
                                            boxImagesPath,parser.batch_size,
                                            drop_last=True,
                                            filter="30-40")#filter disabled
    
    vdataloader = utils.get_data_with_labels(parser.image_size, parser.image_size,1,
                                            validationImages,parser.batch_size, 
                                            drop_last=True,
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
            inputs = inputs.to(device)
            classes = classes.to(device)
            
            """Prepare Data"""
            array1, labels, noise,bands_batch = prepare_data(names, device,df,classes,classes_types,
                                                             substrate_encoder,
                                                             materials_encoder,
                                                             surfaceType_encoder,
                                                             TargetGeometries_encoder,
                                                             bands_encoder)

            noise = noise.type(torch.float).to(device) #Generator input espectro+ruido
            conditions = torch.stack(labels).type(torch.float).to(device) #Discrminator Conditioning Espectro
            #print(noise2)
            label = torch.full((parser.batch_size,), real_label,dtype=torch.float, device=device)
            label_real = torch.full((parser.batch_size,), real_label,dtype=torch.float, device=device)

            # Train discriminator
            loss_d,  D_x, D_G_z1, fakes = train_discriminator(netD,netG,criterion,inputs, opt_D, conditions,noise, label, parser.batch_size,fake_label)

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
            if (iters % 1000 == 0) or ((epoch == parser.epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():

                    testTensor = torch.Tensor().to(device)
                        
                    _,data_val = list(enumerate(vdataloader))[0]
                    inputs_val, classes_val, names_val, classes_types_val = data_val

                    array1, array2_val, noise_val,bands_batch_val = prepare_data(names_val, device,df,classes_val,classes_types_val,
                                                                                 substrate_encoder,
                                                                                 materials_encoder,
                                                                                 surfaceType_encoder,
                                                                                 TargetGeometries_encoder,
                                                                                 bands_encoder)
                                        
                    testTensor = noise_val.type(torch.float).to(device)

                    fake = netG(testTensor).detach().cpu()

                    save_image(fake, "output/"+str(epoch)+"_"+str(iters)+'.png')

                img_list.append(vutils.make_grid(fake,nrow=10, padding=2, normalize=True))

            iters += 1

        if epoch % 1 == 0:
            ##Guarda el modelo en el directorio cada 50 epocas
            if not os.path.exists('CGAN_Model'):
                os.makedirs('CGAN_Model')

            torch.save(netG, 'CGAN_Model/model' + 'netG' + str(epoch) + '.pt')
            torch.save(netD, 'CGAN_Model/model' + 'netD' + str(epoch) + '.pt')
        
    
    return G_losses,D_losses,iter_array,real_scores,fake_scores
            

def prepare_data(names, device,df,classes,classes_types,substrate_encoder, materials_encoder,surfaceType_encoder,TargetGeometries_encoder,bands_encoder):
    bands_batch=[]
    array1 = []
    array2 = []
    noise = torch.Tensor()

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
                                                top_frequencies,)


            bands_batch.append(band_name)


            #loading data to tensors for discriminator
            tensorA = torch.from_numpy(values[1]) #Just have spectra profile
            labels = torch.cat((conditional_data.to(device),tensorA.to(device))) #concat side
            array2.append(labels) # to create stack of tensors
                                            

            latent_tensor=torch.rand(parser.latent)
            
            """multiply noise and labels to get a single vector"""
            #tensor1=torch.mul(labels.to(device),latent_tensor.to(device) )
    
            """concat noise and labels adjacent"""
            tensor1 = torch.cat((conditional_data.to(device),tensorA.to(device),latent_tensor.to(device),)) #concat side

            #un vector que inclue
            #datos desde el dataset y otros datos aleatorios latentes.
            """No lo veo tan claro pero es necesario para pasar los datos al ConvTranspose2d"""
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
    
    """if wanting to add top frequencies to the conditions"""
    #values_array = torch.cat((values_array,top_freqs),0) #concat side

    """ Values array solo pouede llenarse con n+umero y no con textos"""
    # values_array = torch.Tensor(values_array)
    return values_array


def train_discriminator(modelD,modelG,criterion,real_images, opt_d,conditions, generator_noise, label,batch_size,fake_label):

    # Clear discriminator gradients
    #opt_d.zero_grad()
    modelD.zero_grad()
     # Forward pass del batch real a través de NetD
     #noise just creates added channels conditioning the real image

    
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

    os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    arguments()
    join_simulationData()
    

    #one hot encoders
    substrate_encoder=encoders(Substrates)
    materials_encoder=encoders(Materials)
    surfaceType_encoder=encoders(Surfacetypes)
    TargetGeometries_encoder=encoders(TargetGeometries)
    bands_encoder=encoders(Bands)

    # Trainer object (look something to reasses)
    trainer = Stack.Trainer(parser)

    # Sizes for discrimnator and generator
    input_size=parser.spectra_length+parser.condition_len+parser.latent
    generator_mapping_size=parser.image_size
    output_channels=3


    # Create models
    netG = Stack.Generator(trainer.gpu_number, input_size, generator_mapping_size, output_channels)
    netG.apply(weights_init)
    netG.cuda()

    discriminator_mapping_size=64

#depth of feature maps propagated through the discriminator
    
    channels=3
    netD = Stack.Discriminator(parser.condition_len+parser.spectra_length,trainer.gpu_number, parser.image_size, discriminator_mapping_size, channels)
    netD.cuda()
    netD.apply(weights_init)
    
    print(netD)
    print(netG)

    #Binary cross entropy for Discriminator
    criterion = nn.BCELoss()

    # Setup Adam optimizers for both G and D
    opt_D = optimizer.Adam(netD.parameters(), lr=trainer.learning_rate, betas=(0.5, 0.999))
    opt_G = optimizer.Adam(netG.parameters(), lr=trainer.learning_rate, betas=(0.5, 0.999))
    schedulerD = torch.optim.lr_scheduler.ExponentialLR(opt_D, gamma=0.95)
    schedulerG = torch.optim.lr_scheduler.ExponentialLR(opt_G, gamma=0.95)


    date="_GAN_Bands_2May_100epc_64_6conds_zcat"

    G_losses,D_losses,iter_array,real_scores,fake_scores=train(opt_D,opt_G,
                                                            schedulerD,schedulerG,
                                                            criterion,
                                                            netD,netG,
                                                            device,
                                                            date,
                                                            substrate_encoder,
                                                            materials_encoder,
                                                            surfaceType_encoder,
                                                            TargetGeometries_encoder,
                                                            bands_encoder )

    torch.save(netD.state_dict(), 'NETDModelTM_abs_'+date+'.pth')
    torch.save(netG.state_dict(), 'NETGModelTM_abs_'+date+'.pth')

    try:
        np.savetxt('output/loss_Train_TM_NETG_'+date+'.out', G_losses, delimiter=',')
    except:
        np.savetxt('output/loss_Train_TM_NETG_'+date+'.out', [], delimiter=',')

    try:
        np.savetxt('output/acc_Train_TM_NETD_'+date+'.out', D_losses, delimiter=',')
    except:
        np.savetxt('output/acc_Train_TM_NETD_'+date+'.out', [], delimiter=',')
    
    try:
        np.savetxt('output/loss_Valid_TM_iterArray'+date+'.out', iter_array, delimiter=',')
    except:
        np.savetxt('output/loss_Valid_TM_iterArray'+date+'.out', [], delimiter=',')
    
    # try:
    #     np.savetxt('output/acc_val_'+date+'.out', acc_val, delimiter=',')
    # except:
    #     np.savetxt('output/acc_val_'+date+'.out', [], delimiter=',')

    # try:
    #     np.savetxt('output/score_train_'+date+'.out', score_train, delimiter=',')
    # except:
    #     np.savetxt('output/score_train_'+date+'.out', [], delimiter=',')

if __name__ == "__main__":
    main()
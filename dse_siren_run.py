
import librosa
import pandas as pd
import os
import struct
import json
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime
from sklearn import metrics 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
#import tflite_runtime.interpreter as tflite
import absl.logging

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# TODO SOBRE DATASET AUDIO
#https://github.com/karolpiczak/ESC-50
#https://labs.freesound.org/datasets/
#https://zenodo.org/record/2552860#.Y21fVHaZOUk
#https://github.com/DCASE-REPO/dcase2018_baseline/tree/master/task2
#https://zenodo.org/record/3665275#.Y21fNnaZOUk
#https://arxiv.org/pdf/2010.00475.pdf

#USADA
#https://zenodo.org/record/3519845#.Y3QXLXaZOUl

# Se obtienen las diferentes rutas de los datos, tanto audios como metadata y path para obtener las muestras



folder_path_models = 'temp_models'
filenameOutput = 'output/dse_siren_times.json'
soundTestPath = 'audio_test/siren_test.wav'

initialCommandSCP = "scp -r -J rtangarife@200.24.23.71:22666 rtangarife@gita-srv-002:Mfcc_hyperparameters_search/output/models/siren_models"


"""Extracting features using Librosa"""

#Se define funcion para extrar las caracteristicas con la librería librosa, obetiendo los coeficientes ceptrales de frecuencia de Mel
#Se realiza un pading en el tamaño para que concuerden los tamaños de las caracteristicas de entrada al modelo.

def extract_features(file_name, Nmfcc, Nfft, NhopL, NwinL):
    samplerate = 22050
    longitudMaxAudio = 4
    max_pad_len = int(samplerate*longitudMaxAudio/NhopL) + int(samplerate*longitudMaxAudio/NhopL*0.07)  #Calculo longitud de salida de mfcc con 7% de tolerancia para longitud de audios

    try:
      audio, sample_rate = librosa.load(file_name, res_type='soxr_hq') 
      mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=Nmfcc, n_fft=Nfft, hop_length=NhopL, win_length=NwinL)
      pad_width = max_pad_len - mfccs.shape[1]
      mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

    except Exception as e:
      print("Error encountered while parsing file: ", file_name)
      return None 
    #print(mfccs.shape)
    return mfccs
#Se realiza la extracción de caracteristicas, teniendo en cuenta la clase, si el sonido es de la carpeta agregada de la clase explosions va y busca este sonido en la carpeta requerida


"""#DSE SIREN"""

import subprocess

def my_grep(pattern, file):
  string_file = file.decode("utf-8")
  for line in string_file.split('\n'):
      if pattern in line:
          return line

bashCommand = "lscpu"
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
specCPU, error = process.communicate()
specCore = my_grep('Core(s) per socket:',specCPU)
print("SpecsCore:", specCore)

specCPU = my_grep('Model name:', specCPU)
print("SpecsCPU:", specCPU)



#Model and History container
predicTimes = []
TFLitePredicTimes = []
preprocsTimes = []
outAccs = []
outAccsTFLite = []

NExp = 2401              #1              #Identificación con número de experimento
samplerate = 22050
longitudMaxAudio = 4
valuesNmfcc = [30]   #[3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45]  #Valores de parametro a variar para el numero de coeficientes MFCC
valuesNfft = [256, 512, 1024, 2048, 4096]     #[256, 512, 1024, 2048, 4096] #Valores de parametro a variar para la longitud de la FFT
valuesWinL = [256, 512, 1024, 2048, 4096]      #[256, 512, 1024, 2048, 4096]  #Valores de parametro a variar para el tamaño de ventana, este debe ser menor o igual a NFFT, la función hace padding con 0
valuesHopL = [0.25, 0.5, 0.75, 1.0]            #[0.25, 0.5, 0.75, 1.0]      #Valores de parametro a variar para el overlaping opuesto de hop_length
valuesKernelSize = [2, 3, 5, 7]                 #[2, 3, 5, 7] #Valores de parametro de tamaño de kernel a variar dentro del modelo


for Nmfcc in valuesNmfcc:                     #Loop para variar valores del parametro n_mfcc => Numero de coeficientes MFCC
  for Nfft in valuesNfft:                         #Loop para variar valores del parametro n_fft => Tamaño del la FFT
    for iterableNwinL in valuesWinL:              #Loop para variar valores del parametro Win_Length => Longitud de la ventana de muestreo
      if iterableNwinL<=Nfft:
        NwinL = iterableNwinL
      else:
        continue
      for iterableNhopL in valuesHopL:            #Loop para variar valores del parametro Hop_Length => 1/Overlaping
        #if (Nfft==4096 and NwinL==2048 and iterableNhopL<1.0): #or (Nfft==2048 and NwinL==512 and iterableNhopL==0.5):
          #continue
        NhopL = int(iterableNhopL*NwinL)
        num_rows = Nmfcc
        num_columns = int(samplerate*longitudMaxAudio/NhopL) + int(samplerate*longitudMaxAudio/NhopL*0.07) #Calculo longitud de salida de mfcc con 7% de tolerancia para longitud de audios
        num_channels = 1

        
        for k_size in valuesKernelSize:           #Loop para variar valores del parametro kernel size => Tamaño del kernel de capas convolucionales
          #if (Nfft==4096 and NwinL==2048 and iterableNhopL==1.0 and k_size<=5):
            #continue

          predicTimes = []
          TFLitePredicTimes = []
          preprocsTimes = []
          outAccs = []
          outAccsTFLite = []
          for i in range(5):                        #Loop para promediar el experimento realizandolo 5 veces con los mismos valores de parametros
            
            

            print(f'EXERIMENT NUMBER = {NExp}')
            print(f'N_MFCC= {Nmfcc}, Nfft= {Nfft}, NwinL= {NwinL}, NhopL= {NhopL}')

            #OBTAIN MODELS FROM DATA SERVER
            #TOTAL TF MODEL
            modelNamePath = f"/Saved_Siren_NExp{NExp}_Rep{i+1}"
            totalCommand = initialCommandSCP+modelNamePath+f" {folder_path_models}"+modelNamePath
            print(totalCommand)

            bashCommand = totalCommand
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            outModelSCP, error = process.communicate()
            #print(outModelSCP)

            
            #TF LITE MODEL
            modelNamePath = f"/Saved_Siren_NExp{NExp}_Rep{i+1}_lite.tflite"
            totalCommand = initialCommandSCP+modelNamePath+f" {folder_path_models}"+modelNamePath
            #print(totalCommand)

            bashCommand = totalCommand
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            outModelSCP, error = process.communicate()
            #print(outModelSCP)

            startP = datetime.now()
            featuresMfcc = extract_features(soundTestPath, Nmfcc, Nfft, NhopL, NwinL)
            preprocsTime = datetime.now() - startP
            
            print('\nPreprocessing Finished For:\n\n')
            print(f'N_MFCC= {Nmfcc}, Nfft= {Nfft}, NwinL= {NwinL}, NhopL= {NhopL}, k_size= {k_size}, NRep= {i+1}')
            print('\n\nPreprocessing Duration For Sample: \n\n', preprocsTime)
            preprocsTimes.append(preprocsTime)

            featuresMfcc = featuresMfcc.reshape(1, num_rows, num_columns, num_channels)
            input_data = featuresMfcc
            print("INPUT DATA SHAPE", input_data.shape)

            #Load Model Experiment to predict 
            modelPath =  folder_path_models+f"/Saved_Siren_NExp{NExp}_Rep{i+1}"
            model = tf.keras.models.load_model(modelPath)

            #Predict and take time
            startPredicTime = datetime.now()
            probOut = model.predict(input_data)[0]
            durationPredicTime = datetime.now() - startPredicTime
            predicTimes.append(durationPredicTime)
            indexMax = np.argmax(probOut)
            outAccs.append(probOut)
            print(probOut)
            print(indexMax)
            maxProb = probOut[indexMax]
            if(maxProb>0.70):
                if(indexMax==0):
                    classP = 'siren'
                else:
                    classP = 'non_siren'
            else:
                classP = 'None'
            print('Class predicted :',classP,'\n\n')

            print(f"Prediction completed in time for sample: ", durationPredicTime)

            #LOAD TF LITE MODEL AND GET PREDICTION
            modelPathTFLite =  folder_path_models+f"/Saved_Siren_NExp{NExp}_Rep{i+1}_lite.tflite"
            #interpreter = tflite.Interpreter(model_path=modelPathTFLite)
            interpreter = tf.lite.Interpreter(model_path=modelPathTFLite)
            interpreter.allocate_tensors()  # Needed before execution!
            output = interpreter.get_output_details()[0]  # Model has single output.
            input = interpreter.get_input_details()[0]  # Model has single input.

            
            startPredicTime = datetime.now()
            # Invoke the model on the input data
            interpreter.set_tensor(input['index'], input_data)
            interpreter.invoke()
            # Get the result
            output_data = interpreter.get_tensor(output['index'])[0]
            durationPredicTime = datetime.now() - startPredicTime
            TFLitePredicTimes.append(durationPredicTime)
            print("OUTPUT DATA SHAPE TF LITE", output_data.shape)
            print(output_data)
            indexMax = np.argmax(output_data)
            print(indexMax)
            maxProb = output_data[indexMax]
            if(maxProb>0.70):
                if(indexMax==0):
                    classP = 'siren'
                else:
                    classP = 'non_siren'
            else:
                classP = 'None'
            print('Class predicted :',classP,'\n\n')
            outAccsTFLite.append(output_data)
            print(f"Prediction completed for TF LITE MODEL in time for sample: ", durationPredicTime)


            #REMOVE MODELS USED IN THE EXPERIMENT RM -RF
            #DELETE TOTAL TF MODEL
            modelNamePath = f"/Saved_Siren_NExp{NExp}_Rep{i+1}"
            totalCommand = "rm -rf"+f" {folder_path_models}"+modelNamePath
            print(totalCommand)

            bashCommand = totalCommand
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            outModelDel, error = process.communicate()
            print(outModelDel)

            
            #DELETE TF LITE MODEL
            modelNamePath = f"/Saved_Siren_NExp{NExp}_Rep{i+1}_lite.tflite"
            totalCommand = "rm -rf"+f" {folder_path_models}"+modelNamePath
            print(totalCommand)

            bashCommand = totalCommand
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            outModelDel, error = process.communicate()
            print(outModelDel)

          Experiment = {
            'NExp': NExp,
            'N_MFCC': Nmfcc,
            'Nfft': Nfft,
            'NwinL': NwinL,
            'NhopL': NhopL,
            'Ksize': k_size,
            'preprocsTimes': preprocsTimes,
            'predicTimes': predicTimes,
            'TFLitePredicTimes': TFLitePredicTimes,
            'outAccs': outAccs,
            'outAccsTFLite': outAccsTFLite,
            'specCPU': specCPU,
            'specCore': specCore
          }
          
          print(Experiment)
          json_object = json.dumps(Experiment, ensure_ascii = False, indent=2, sort_keys=True, default=str)
          print(json_object)

          # 1. Read json file
          with open(filenameOutput, "r") as f:
              data = json.load(f)
              f.close() 
          # 2. Update json object
          data["Experiments"].append(Experiment)
          # 3. Write json file
          with open(filenameOutput, "w") as f:
              json.dump(data, f, ensure_ascii = False, indent=2, sort_keys=True, default=str)
              f.close()

          NExp += 1 #Add one to experiments counter



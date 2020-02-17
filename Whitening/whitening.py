from pytsa.tsa import *
from pytsa.tsa import SeqView_double_t as SV
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from wdfml.config.parameters import Parameters
from wdfml.processes.whitening import  *
from wdfml.processes.filtering import *
import pandas as pd

def wdf_whitening(filejson,gpsStart,LVfile,ARfile):
    #Load json file
    par = Parameters()
    try:
        par.load(filejson)
    except IOError:
        logging.error("Cannot find resource file " + filejson)
        quit()
    
    #Get some info    
    strInfo = FrameIChannel(par.file, par.channel, 1.0, par.gps)
    Info = SV()
    strInfo.GetData(Info)
    par.sampling = int(1.0 / Info.GetSampling())
    par.resampling = int(par.sampling / par.ResamplingFactor)
    print("sampling frequency= %s, resampled frequency= %s" %(par.sampling, par.resampling))
    
    #Load AR,LV parameters
    print(ARfile)
    print(LVfile)
    Learn = SV()
    Learn_DS = SV()
    whiten=Whitening(par.ARorder)
    whiten.ParametersLoad(ARfile, LVfile)
    par.sigma = whiten.GetSigma()
    print('Estimated sigma= %s' % par.sigma)
    par.lenStart  = 10
    par.len = 1
    
    #Whitening
    # Parameter for sequence of data
    # read data
    print('Starting whitening process')
    par.Noutdata = int(par.len * par.resampling)
    ds = downsamplig(par)
    gpsstart = gpsStart - 2*par.len
    streaming = FrameIChannel(par.file, par.channel,par.lenStart, gpsstart)
    data = SV()
    data_ds = SV()
    dataw = SV()
    streaming.GetData(data)
    ds.Process(data, data_ds)
    whiten.Process(data_ds, dataw)
    ds.Process(data, data_ds)
    whiten.Process(data_ds,dataw)
    ds.Process(data, data_ds)
    whiten.Process(data_ds,dataw)
    
    #Transform into numpy array
    data_w=[]
    time = []
    for j in range(dataw.GetSize()):
        data_w.append(dataw.GetY(0,j))
        time.append(dataw.GetX(j))
    return np.array(time),np.array(data_w)

def wdf_whitening_segment(filejson,gps_list,LVfile,ARfile,sampling,resampling,
                          length, lenStart=10):
    #Load json file
    par = Parameters()
    try:
        par.load(filejson)
    except IOError:
        logging.error("Cannot find resource file " + filejson)
        quit()
    
    #Get some info    
    #strInfo = FrameIChannel(par.file, par.channel, 1.0, par.gps)
    #Info = SV()
    #strInfo.GetData(Info)
    #par.sampling = int(1.0 / Info.GetSampling())
    #par.resampling = int(par.sampling / par.ResamplingFactor)
    par.sampling = sampling
    par.resampling = resampling
    print("sampling frequency= %s, resampled frequency= %s" %(par.sampling, par.resampling))
    
    #Load AR,LV parameters
    print(ARfile)
    print(LVfile)
    Learn = SV()
    Learn_DS = SV()
    whiten=Whitening(par.ARorder)
    whiten.ParametersLoad(ARfile, LVfile)
    par.sigma = whiten.GetSigma()
    print('Estimated sigma= %s' % par.sigma)
    par.lenStart  = lenStart
    par.len = length
    output = []
    #Whitening
    # Parameter for sequence of data
    # read data
    print('Starting whitening process')
    print ('Number of triggers:%i'%len(gps_list))
    for i,gpsStart in enumerate(gps_list):
        print (i,gpsStart)
        par.Noutdata = int(par.len * par.resampling)
        ds = downsamplig(par)
        gpsstart = gpsStart - 2*par.len
        streaming = FrameIChannel(par.file, par.channel,par.lenStart, gpsstart)
        data = SV()
        data_ds = SV()
        dataw = SV()
        streaming.GetData(data)
        ds.Process(data, data_ds)
        whiten.Process(data_ds, dataw)
        ds.Process(data, data_ds)
        whiten.Process(data_ds,dataw)
        ds.Process(data, data_ds)
        whiten.Process(data_ds,dataw)

        #Transform into numpy array
        data_w=[]
        time = []
        for j in range(dataw.GetSize()):
            data_w.append(dataw.GetY(0,j))
            time.append(dataw.GetX(j))
        output.append({'time': np.array(time),'strain': np.array(data_w)})
    return  output
import time
import os
from pytsa.tsa import *
from pytsa.tsa import SeqView_double_t as SV
from wdfml.config.parameters import * # Includes: Parameters, wdfParameters
from wdfml.processes.wdfWorker import *
import logging
import coloredlogs
coloredlogs.install(isatty=True)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.info("read parameters from JSON file")
    par = Parameters()
    filejson = "channel.json"
    try:
        par.load(filejson)
    except IOError:
        logging.error("Cannot find resource file " + filejson)
        quit()
    strInfo = FrameIChannel(par.file, par.channel, 1.0, par.gps)
    Info = SV()
    strInfo.GetData(Info)
    par.sampling = int(1.0 / Info.GetSampling())
    par.resampling = int(par.sampling / par.ResamplingFactor)
    logging.info("sampling frequency= %s, resampled frequency= %s" %(par.sampling, par.resampling))
    del Info, strInfo
    for segment in par.segment:
        wdf=wdfWorker(par,fullPrint=1)
        #wdf.segmentProcess(par.segment, WaveletThreshold.cuoco)
        wdf.segmentProcess(segment)


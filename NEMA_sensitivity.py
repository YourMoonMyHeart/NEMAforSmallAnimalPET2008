import numpy as np
from matplotlib import pyplot as plt
plt.ion()

def GetSensitivity(bins,views,slices,crystalPitch,michPathList,michBackgroundPath,scanTimeList,activity):
    '''get the sensitivity of measurement i'''
    michB = ReadMichFromRowData(bins,views,slices,michBackgroundPath)
    ssrbSinoB = SSRB(michB)
    S=[]
    S_A=[]
    del michB
    for michPath,scanTime in zip(michPathList,scanTimeList):
        mich = ReadMichFromRowData(bins,views,slices,michPath)
        ssrbSino = SSRB(mich)
        del mich
        sino,sinoB = SetGreaterThan1cmFromPeakPixelZeroWithBackground(ssrbSino,2,ssrbSinoB)
        R_i = np.sum(sino)/scanTime # R for count rate, and i for measurements i
        R_Bi = np.sum(sinoB)/scanTime # B for background
        S_i = (R_i-R_Bi)/activity
        S_Ai = S_i/0.9060 # A for absolute
        S.append(S_i)
        S_A.append(S_Ai)
    return S,S_A

def SetGreaterThan1cmFromPeakPixelZeroWithBackground(ssrbSino,crystalPitch,ssrbBg):
    slices,views,bins = ssrbSino.shape
    distance = 10 # 1 cm
    mark=int(np.ceil(distance/crystalPitch*2)) # approximate the detector as a panel
    maxBin = np.argmax(np.max(ssrbSino,axis=0),axis=1) # find the max bin of each view
    sino = np.zeros([slices,views,bins])
    sinoB = np.zeros([slices,views,bins])
    for i in range(views):
        sino[:,i,maxBin[i]-mark:maxBin[i]+mark+1] = ssrbSino[:,i,maxBin[i]-mark:maxBin[i]+mark+1]
    for i in range(views):
        sinoB[:,i,maxBin[i]-mark:maxBin[i]+mark+1] = ssrbBg[:,i,maxBin[i]-mark:maxBin[i]+mark+1]
    return sino,sinoB

def SetGreaterThan1cmFromPeakPixelZero(ssrbSino,crystalPitch):
    slices,views,bins = ssrbSino.shape
    distance = 10 # 1 cm
    mark=int(np.ceil(distance/crystalPitch*2)) # approximate the detector as a panel
    maxBin = np.argmax(np.max(ssrbSino,axis=0),axis=1) # find the max bin of each view
    sino = np.zeros([slices,views,bins])
    for i in range(views):
        sino[:,i,maxBin[i]-mark:maxBin[i]+mark+1] = ssrbSino[:,i,maxBin[i]-mark:maxBin[i]+mark+1]
    return sino

def SSRB(mich): # do the single-slice rebinning
    slices,views,bins = mich.shape
    ringNum = int(slices**0.5)
    ssrbSino = np.zeros([2*ringNum-1,views,bins])
    for i in range(2*ringNum-1):
        for ring1 in range(ringNum):
            ring2 = i-ring1 # i = ring1+ring2
            if((ring2>=0) & (ring2<ringNum)): # note:'&' has the same priority as '>' and '<'
                ssrbSino[i,:,:] += mich[ring1*ringNum+ring2,:,:]
    return ssrbSino

def ReadMichFromRowData(bins,views,slices,michPath,dType='float32'):
    img = np.fromfile(michPath,dtype=dType)
    img = img.reshape([slices,views,bins])
    return img

if __name__ == '__main__':
    #===========================================================================
    #                                   setting
    #===========================================================================
    bins,views,slices = (311,156,104*104)
    crystalPitch = 2
    activity = 17*37000
    # you need to notify the time of each scan
    scanTimeList = [3*60 for i in range(39)]
    # you need to notify the axial position of each scan
    axialPos = [i*5 for i in range(-19,20,1)]
    # you need to write the path of each michelogram
    michBackgroundPath = "D://NEMA201903//sensitivity//RadialBackground//PET//mich.dat"
    fileList = []
    for pos in axialPos:
        path = "D:/NEMA201903/sensitivity/Radial"+str(pos)+'/PET/mich.dat'
        fileList.append(path)
        print(path)
    #===========================================================================
    #                               calculation
    #===========================================================================
    S,S_A = GetSensitivity(bins,views,slices,crystalPitch,fileList,michBackgroundPath,scanTimeList,activity)
    print("S:",end='\t')
    print(S)
    print("Sabsolute",end='\t')
    print(S_A)
    #plt.plot(axialPos,S,'r.-')
    plt.plot(axialPos,S_A,'b.-')
    plt.title(' Absolute Sensitivity',fontsize=18)
    plt.xlabel('axial',fontsize=12)
    plt.ylabel('counts',fontsize=12)

import numpy as np
from matplotlib import pyplot as plt
import math
import sys
plt.ion()

def GetNECR(bins,views,slices,michPathList,scanTimeList,backgroundPath,bgScanTime,phantomRadius,sourceLength,scannerRadius,crystalSize,SFInLowActivity=0,randPathList=0):
    if not SFInLowActivity is 0:
        print('calculate random events use scatter fraction')
    if not randPathList is 0:
        print('calculate random events use random estimation')
    michB = ReadMichFromRowData(bins,views,slices,backgroundPath)
    ssrbSinoB = SSRB(michB)
    del michB
    sinoB = SetFartherThan8mmFromThePhantomEdgesPixelZero(ssrbSinoB,phantomRadius,scannerRadius)
    Rint = np.sum(np.sum(sinoB,axis=1),axis=1)/bgScanTime
    acquisitions = len(michPathList)
    ssrbSlices = sinoB.shape[0]
    Rtot = np.zeros([acquisitions,ssrbSlices])
    Rt = np.zeros([acquisitions,ssrbSlices])
    Rr = np.zeros([acquisitions,ssrbSlices])
    Rs = np.zeros([acquisitions,ssrbSlices])
    SF = np.zeros([acquisitions])
    Rnec = np.zeros([acquisitions,ssrbSlices])
    j=0
    for michPath,time in zip(michPathList,scanTimeList):
        mich = ReadMichFromRowData(bins,views,slices,michPath)
        ssrbSino = SSRB(mich)
        sino = SetFartherThan8mmFromThePhantomEdgesPixelZero(ssrbSino,phantomRadius,scannerRadius)
        alignedSino = AlignGreatestValuePixelofSSRBSinogram(sino)
        countsPlot = np.sum(alignedSino,axis=1)                     # sum in view
        Ctot = np.sum(countsPlot,axis=1)                            # total count in each slice i
        Crs = GetScatterCounts(countsPlot,crystalSize)              # scatter & random counts for each slice i of acquisition j
        Rtot[j,:] = Ctot/time                                       # total counts rates
        Rt[j,:] = (Ctot-Crs)/time                                   # true counts rates
        if not SFInLowActivity is 0:#================use scatter fraction in low dose to estimate random events
            Rr[j,:] = Rtot[j,:]-(Rt[j,:]/(1-SFInLowActivity))       # random counts rates
            Rr[j,Rr[j,:]<0]=0 # set negtive to zero
        elif not randPathList is 0:#=================use system random estimate
            michR = ReadMichFromRowData(bins,views,slices,randPathList[j])
            ssrbSinoR = SSRB(michR)
            del michR
            sinoR = SetFartherThan8mmFromThePhantomEdgesPixelZero(ssrbSinoR,phantomRadius,scannerRadius)
            Rr[j,:] = np.sum(np.sum(sinoR,axis=1),axis=1)/bgScanTime
        else:
            print('GetNECR: Input Error, can not calculate random events!')
            sys.exit()
        Rs[j,:] = Rtot[j,:] - Rt[j,:] - Rr[j,:] - Rint      # scatter counts rates
        Rnec[j,:] = Rt[j,:]*Rt[j,:]/Rtot[j,:]               # NECR
        SF[j] = np.sum(Rs[j,:])/np.sum(Rt[j,:] + Rs[j,:])   # scatter fraction
        j += 1
        print("%d%%"%int(j/acquisitions*100))
    Rtot = np.sum(Rtot,axis=1) # sum all the slice
    Rt = np.sum(Rt,axis=1)
    Rr = np.sum(Rr,axis=1)
    Rs = np.sum(Rs,axis=1)
    Rnec = np.sum(Rnec,axis=1)
    return Rtot,Rt,Rr,Rs,SF,Rnec

def GetScatterFraction(bins,views,slices,michPathList,scanTimeList,phantomRadius,scannerRadius,crystalSize):
    ssrbSlices = 2*int(slices**0.5)-1
    acquisitions = len(michPathList)
    Ctot = np.zeros([acquisitions,ssrbSlices])
    Crs = np.zeros([acquisitions,ssrbSlices])
    j=0
    for michPath in michPathList:
        mich = ReadMichFromRowData(bins,views,slices,michPath)
        ssrbSino = SSRB(mich)
        sino = SetFartherThan8mmFromThePhantomEdgesPixelZero(ssrbSino,phantomRadius,scannerRadius)
        alignedSino = AlignGreatestValuePixelofSSRBSinogram(sino)
        countsPlot = np.sum(alignedSino,axis=1)             # sum in view
        Ctot[j,:] = np.sum(countsPlot,axis=1)               # total count in each slice i
        Crs[j,:] = GetScatterCounts(countsPlot,crystalSize) # scatter & random counts for each slice i of acquisition j
        j += 1
    SF_i = np.sum(Crs,axis=0)/np.sum(Ctot,axis=0)
    SF = np.sum(Crs)/np.sum(Ctot)
    return SF_i,SF

def GetScatterCounts(sumSino,crystalSize):
    slices,bins = sumSino.shape
    centerBin = int(bins/2)
    fOffsetBin = 7/crystalSize*2
    nOffsetBin = int(fOffsetBin)
    Cr = GetYpositionByLinearInterpolation(sumSino[:,centerBin+nOffsetBin],centerBin+nOffsetBin,sumSino[:,centerBin+nOffsetBin+1],centerBin+nOffsetBin+1,centerBin+fOffsetBin)
    Cl = GetYpositionByLinearInterpolation(sumSino[:,centerBin-nOffsetBin],centerBin-nOffsetBin,sumSino[:,centerBin-nOffsetBin-1],centerBin-nOffsetBin-1,centerBin-fOffsetBin)
    # calculate the scatter counts
    Csr = (Cr + Cl)/2*2*fOffsetBin
    Csr += np.sum(sumSino[:,:centerBin-nOffsetBin-1],axis=1) + np.sum(sumSino[:,centerBin+nOffsetBin+1:],axis=1)
    return Csr

def GetYpositionByLinearInterpolation(y1,x1,y2,x2,xk):
    return (y2-y1)/(x2-x1)*(xk-x1)+y1

def AlignGreatestValuePixelofSSRBSinogram(ssrbSino):
    slices,views,bins = ssrbSino.shape
    alignedSino = np.zeros([slices,views,bins])
    centerBin = int(bins/2)
    for sl in range(slices):
        for vi in range(views):
            maxBin = np.argmax(ssrbSino[sl,vi,:])
            shiftBinNum = centerBin-maxBin
            copyBinLowNum = min(centerBin-0,maxBin)
            copyBinHighNum = min(bins-centerBin,bins-maxBin)
            alignedSino[sl,vi,centerBin-copyBinLowNum:centerBin+copyBinHighNum] = ssrbSino[sl,vi,maxBin-copyBinLowNum:maxBin+copyBinHighNum]
    return alignedSino


def SetFartherThan8mmFromThePhantomEdgesPixelZero(ssrbSino,phantomRadius,scannerRadius):
    slices,views,bins = ssrbSino.shape
    sino = np.zeros([slices,views,bins])
    # approximate the detector as a ring
    theta = np.arcsin((phantomRadius+8)/scannerRadius)
    mark = int(np.ceil(theta/np.pi*views))
    binCenter = int(bins/2)
    for i in range(views):
        sino[:,i,binCenter-mark:binCenter+mark+1] = ssrbSino[:,i,binCenter-mark:binCenter+mark+1]
    return sino

def SSRB(mich):
    slices,views,bins = mich.shape
    ringNum = int(slices**0.5)
    ssrbSino = np.zeros([2*ringNum-1,views,bins])
    for i in range(2*ringNum-1):
        for ring1 in range(ringNum):
            ring2 = i-ring1                         # i = ring1+ring2
            if((ring2>=0) & (ring2<ringNum)):       # note:'&' has the same priority as '>' and '<'
                ssrbSino[i,:,:] += mich[ring1*ringNum+ring2,:,:]
    return ssrbSino

def ReadMichFromRowData(bins,views,slices,michPath,dType='float32'):
    img = np.fromfile(michPath,dtype=dType)
    img = img.reshape([slices,views,bins])
    return img

def GetAverageActivityInEachScan(beginActivity,beginTime,scanBeginTimeList,scanTimeList,halfLife):
    avgActivityList = []
    for scanBeginTime,scanTime in zip(scanBeginTimeList,scanTimeList):
        Tb = scanBeginTime-beginTime            # the time from measurement to scan begin
        Ts = scanTime                           # the time of scan
        Thalf = halfLife                        # change a name
        avgActivity = beginActivity*Thalf/(math.log(2)*Ts)*0.5**(Tb/Thalf)*(1-0.5**(Ts/Thalf))
        avgActivityList.append(avgActivity)
    return avgActivityList

if __name__ == '__main__':
    #===========================================================================
    #                                   setting
    #===========================================================================
    bins,views,slices = 311,156,104*104
    phantomRadius = 50
    sourceLength = 140-10
    scannerRadius = 106
    crystalSize = 2
    measuredActivity = 4000*37000 # the measurement activity before scan
    #=========== time =============
    measuredTime = 1                                    # the time when source activity was measured
    halfLifeTime = 108*60                               # the half life of the source
    scanTimeList = [10*60 for i in range(10)]           # the scan time of each acquisition
    scanBeginTimeList = [10*i for i in range(10)]       # the time of each qcuisition start
    bgScanTime = 8*60*60                                # the scan time for background
    #=========== file path ===============
    michPathList=[]
    for i in range(10):
        path = "D:/NEMA201903NECR/NECR_RatSize/"+str(i+1+80)+"/PET/mich.dat"
        michPathList.append(path)
        print(path)
    backgroundPath = r'D:\NEMA201903NECR\NECR_RatSize\NECR_RatSize_Background\PET\mich.dat'
    michPathListInLowDose = michPathList[7:9]
    scanTimeListInLowDose = scanTimeList[7:9]
    #===========================================================================
    #                               calculation
    #===========================================================================
    #=========== the average activity in acquisition ============
    avgActivityList = GetAverageActivityInEachScan(measuredActivity,measuredTime,scanBeginTimeList,scanTimeList,halfLifeTime)
    #=========== scatter fraction ===============
    SF_i,SF = GetScatterFraction(bins,views,slices,michPathListInLowDose,scanTimeListInLowDose,phantomRadius,scannerRadius,crystalSize)
    print('system scatter fraction: ',end='')
    print(SF)
    #=========== NECR ===========
    Rtot,Rt,Rr,Rs,SF,Rnec = GetNECR(bins,views,slices,michPathList,scanTimeList,backgroundPath,bgScanTime,phantomRadius,sourceLength,scannerRadius,crystalSize,SFInLowActivity=SF_i)
    #=========== plot scatter fraction =============
    plt.figure('scatter fraction in each slice')
    plt.title('scatter fraction in each slice',fontsize=18)
    plt.xlabel('slice number',fontsize=12)
    plt.plot(SF_i)
    #=========== plot NECR =============
    plt.figure('NECR')
    plt.title('NECR',fontsize=18)
    plt.xlabel('activity concentration',fontsize=12)
    plt.ylabel('counts',fontsize=12)
    plt.plot(avgActivityList,Rtot,'.-',color='blue',label='total')
    plt.plot(avgActivityList,Rt,'.-',color='red',label='true')
    plt.plot(avgActivityList,Rr,'.-',color='green',label='random')
    plt.plot(avgActivityList,Rs,'.-',color='yellow',label='scatter')
    plt.plot(avgActivityList,Rnec,'.-',color='cyan',label='necr')
    plt.legend(loc='upper left')
    #=========== plot scatter fraction in each acquisition ============
    plt.figure('scatter fraction')
    plt.title('scatter fraction in each acquisition',fontsize=18)
    plt.xlabel('activity concentration',fontsize=12)
    plt.plot(SF)









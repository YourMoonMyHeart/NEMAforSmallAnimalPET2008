import numpy as np
from matplotlib import pyplot as plt
try:
    import pydicom
except ImportError:
    print('You do not have pydicom in your computer! Please install if you need to read the dicom file.')
import sys

def GetNEMASpatialResolution(img,pixelSizeX=1,pixelSizeY=1,pixelSizeZ=1,isPlot=True):
    try:
        z,y,x = img.shape
    except ValueError:
        print('GetNEMASpatialResolution: Image input error! you should input a 3D numpy array.')
        sys.exit() # exit the interpreter
    xline = img.sum(axis=0).sum(axis=0) #sum z, sum y. for radial spatial resolution
    yline = img.sum(axis=0).sum(axis=1) #sum z, sum x. for Tangential spatial resolution
    zline = img.sum(axis=1).sum(axis=1) #sum y, sum x. for axial spatial resolution
    radialFWHM,radialFWTM = GetNEMASpatialResolutionOneDirection(xline,pixelSizeX,'radial',isPlot)
    tangentialFWHM,tangentialFWTM = GetNEMASpatialResolutionOneDirection(yline,pixelSizeY,'rangential',isPlot)
    axialFWHM,axialFWTM = GetNEMASpatialResolutionOneDirection(zline,pixelSizeZ,'axial',isPlot)
    return (radialFWHM,tangentialFWHM,axialFWHM),(radialFWTM,tangentialFWTM,axialFWTM)

def GetNEMASpatialResolutionOneDirection(line,pixelSize,discription='',isPlot=True):
    try:
        l, = line.shape
    except ValueError:
        print('GetNEMASpatialResolutionOneDirection: Line input error! you should input a 1D numpy array.')
        sys.exit() #exit the interpreter
    maxid = line.argmax()
    #maxvalue = GetParabolicMaxValue(line[maxid-1],line[maxid],line[maxid+1])
    a,b,c = GetParabolicParaBy3Point(maxid*pixelSize,line[maxid],(maxid-1)*pixelSize,line[maxid-1],(maxid+1)*pixelSize,line[maxid+1])
    maxvalue = (4*a*c-b*b)/4/a
    halfMaxValue = maxvalue/2
    tenthMaxValue = maxvalue/10
    #FWHM
    idHalfHigh=idHalfLow=maxid
    while((line[idHalfHigh]>halfMaxValue)  & (idHalfHigh<l-1)):
        idHalfHigh = idHalfHigh+1
    while((line[idHalfLow]>halfMaxValue) & (idHalfLow>0)):
        idHalfLow = idHalfLow-1
    xHalfHigh = GetXpositionByLinearInterpolation(line[idHalfHigh],idHalfHigh,line[idHalfHigh-1],idHalfHigh-1,halfMaxValue)
    xHalfLow = GetXpositionByLinearInterpolation(line[idHalfLow],idHalfLow,line[idHalfLow+1],idHalfLow+1,halfMaxValue)
    FWHM = (xHalfHigh - xHalfLow)*pixelSize
    #FWTM
    idTenthHigh=idTenthLow=maxid
    while((line[idTenthHigh]>tenthMaxValue) & (idTenthHigh<l-1)):
        idTenthHigh = idTenthHigh+1
    while((line[idTenthLow]>tenthMaxValue) & (idTenthLow>0)):
        idTenthLow = idTenthLow-1
    xTenthHigh = GetXpositionByLinearInterpolation(line[idTenthHigh],idTenthHigh,line[idTenthHigh-1],idTenthHigh-1,tenthMaxValue)
    xTenthLow = GetXpositionByLinearInterpolation(line[idTenthLow],idTenthLow,line[idTenthLow+1],idTenthLow+1,tenthMaxValue)
    FWTM = (xTenthHigh - xTenthLow)*pixelSize
    #plot
    if isPlot:
        plt.ion()
        x = np.arange(l)*pixelSize
        plt.figure()
        plt.title(discription,fontsize=18)
        # response function
        plt.plot(x,line)
        #max value
        xParabolicCenter = -b/(2*a)
        xParabolic = np.arange(xParabolicCenter-pixelSize,xParabolicCenter+pixelSize + pixelSize/50,pixelSize/50)
        yParabolic = a*xParabolic*xParabolic + b*xParabolic + c
        plt.plot(xParabolic,yParabolic,'r')
        plt.plot(xParabolicCenter,maxvalue,'r.-')
        # FWHM
        plt.plot([xHalfLow*pixelSize,xHalfHigh*pixelSize],[halfMaxValue,halfMaxValue],'r.-')
        plt.text(xHalfHigh*pixelSize,halfMaxValue,'%.3f'% FWHM,color='green',fontsize=15)
        # FWTM
        plt.plot([xTenthLow*pixelSize,xTenthHigh*pixelSize],[tenthMaxValue,tenthMaxValue],'r.-')
        plt.text(xTenthHigh*pixelSize,tenthMaxValue,'%.3f'% FWTM,color='green',fontsize=15)
        plt.xlim((idTenthLow-10)*pixelSize,(idTenthHigh+10)*pixelSize) # x axis limit
        plt.xlabel('Pixel Location',fontsize=12)
    return FWHM,FWTM

def GetParabolicParaBy3Point(x1,y1,x2,y2,x3,y3):
    if((x2-x1)*(y3-y2)-(x3-x2)*(y2-y1)==0):
        print('GetParabolicParaBy3Point: 3 point are in the same line.')
        sys.exit()
    a = 1/(x3*x3-(x1+x2)*x3+x1*x2)*y3 + 1/(x2*x2-(x1+x3)*x2+x1*x3)*y2 + 1/(x1*x1-(x2+x3)*x1+x2*x3)*y1
    b = -(x1+x2)/(x3*x3-(x1+x2)*x3+x1*x2)*y3 - (x1+x3)/(x2*x2-(x1+x3)*x2+x1*x3)*y2 - (x2+x3)/(x1*x1-(x2+x3)*x1+x2*x3)*y1
    c = x1*x2/(x3*x3-(x1+x2)*x3+x1*x2)*y3 + x1*x3/(x2*x2-(x1+x3)*x2+x1*x3)*y2 + x2*x3/(x1*x1-(x2+x3)*x1+x2*x3)*y1
    return a,b,c
    
def GetParabolicYBy3Point(x1,y1,x2,y2,x3,y3,x):
    return (x-x1)*(x-x2)/(x3-x1)/(x3-x2)*y3 + (x-x1)*(x-x3)/(x2-x1)/(x2-x3)*y2 + (x-x2)*(x-x3)/(x1-x2)/(x1-x3)*y1
    
def GetParabolicMaxValue(y1,y2,y3):
    '''Input 3 value of y direction with the same distance in x direction. Out put is the max value of the parabolic fix by this 3 point'''
    return y2-(y1-y3)**2/8/(y1+y3-2*y2)

def GetXpositionByLinearInterpolation(y1,x1,y2,x2,yk):
    '''Get the x coordinate value by interpolation with (x1,y1) and (x2,y2) in the value of yk'''
    return (yk-y1)*(x2-x1)/(y2-y1)+x1
    
def ReadImage3DFromDicom(dicomPathList):
    ds = pydicom.dcmread(dicomPathList[0],force=True)
    rows = ds.Rows
    columns = ds.Columns
    slices = ds.Slices
    img = np.zeros([slices,columns,rows],dtype='float32')
    for i in range(slices):
        ds = pydicom.dcmread(dicomPathList[i],force=True)
        imgSlice = ds.pixel_array
        img[i,:,:] = imgSlice
    return img,ds.PixelSpace,ds.Thickness

def ReadImage3DFromSliceRowData(rows,columns,slices,imgPathList,dType='int16'):
    z,y,x = slices,columns,rows
    img = np.zeros([z,y,x],dtype=dType)
    for i in range(z):
        imgSlice = np.fromfile(imgPathList[i],dtype=dType)
        imgSlice = imgSlice.reshape([y,x])
        img[i,:,:] = imgSlice
    return img

def ReadImage3DFromRowData(rows,columns,slices,imgPath,dType='float32'):
    img = np.fromfile(imgPath,dtype=dType)
    img = img.reshape([slices,columns,rows])
    return img

if __name__ == '__main__':
    print('start:')
    #===========================================================================
    #                                   setting
    #===========================================================================
    imgFloder = r'G:\Raycan\NEMA201903\Spatial Resolution\2.Axial0_Radial5\3.Img_CudaPSF'
    imgPathList=[]
    for i in range(400):
        imgPath =imgFloder +  '\\I'+str(i+1)
        imgPathList.append(imgPath)
    img = ReadImage3DFromSliceRowData(320,320,400,imgPathList,'int16')
    #===========================================================================
    #                               calculation
    #===========================================================================
    FWHM,FWTM = GetNEMASpatialResolution(img,0.5,0.5,0.5,isPlot=True)
    print('the FWHM in radial, tangential and axial is:',end='\t')
    print(FWHM)
    print('the FWTM in radial, tangential and axial is:',end='\t')
    print(FWTM)

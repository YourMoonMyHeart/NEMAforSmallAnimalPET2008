import numpy as np
import matplotlib.pyplot as plt
import math
#plt.ion()

# Get Uniformity Parameters
def getUniformity(Img, ZRange, XYRange, ThicknessZ, PixelSizeXY):
    z, y, x = Img.shape
    VOI = np.zeros([int(ZRange), int(XYRange), int(XYRange)], dtype = 'float32')
    VOI = Img[int(z/2-ZRange):int(z/2), int(y/2-XYRange/2+5):int(y/2+XYRange/2+5), int(x/2-XYRange/2-3):int(x/2+XYRange/2-3)]
    '''
    plt.figure('VOI[0]')
    plt.imshow(VOI[0, :, :])
    plt.figure('VOI[19]')
    plt.imshow(VOI[19, :, :])
    plt.show()
    '''
    PixelList = []
    for i in range(int(ZRange)):
        for j in range(int(XYRange)):
            for k in range(int(XYRange)):
                if(np.sqrt(((j-XYRange/2)**2 + (k-XYRange/2)**2)) < XYRange/2):
                    PixelList.append(VOI[i, j, k])

    MaxValue = np.max(PixelList)
    MinValue = np.min(PixelList)
    MeanValue = np.mean(PixelList)
    UniformitySTD = np.std(PixelList, ddof = 1) / MeanValue 
    AvgActivityConcentration = MeanValue  / (PixelSizeXY ** 2 * ThicknessZ)
    return AvgActivityConcentration, MaxValue, MinValue, MeanValue, UniformitySTD
    
# Get Recovery Coefficients Parameters
def getRC(Img, ZRange, XYRange, ThicknessZ, PixelSizeXY, AvgActivityConcentration, UniformitySTD):
    z, y, x = Img.shape
    VOI = np.zeros([int(ZRange), int(XYRange), int(XYRange)], dtype = 'float32')
    VOI = Img[int(z/2+12):int(z/2+ZRange+12), int(y/2-XYRange/2+3):int(y/2+XYRange/2+3), int(x/2-XYRange/2-5):int(x/2+XYRange/2-5)]
    '''
    plt.figure('VOI[0]')
    plt.imshow(VOI[0, :, :])
    plt.figure('VOI[19]')
    plt.imshow(VOI[19, :, :])
    plt.show()
    '''
    # The central 10mm length of the rods shall be averaged to obtain
    # a single image slice of lower noise
    ROI = np.mean(VOI, axis = 0)
    '''
    plt.figure('ROI')
    plt.imshow(ROI)
    plt.show()
    '''

    # Around each rod with diameters twice the physical diamater of the rods
    # The maximum values in each of these ROIs shall be measured
    Rod1mmMaxPixelIndex = np.argmax(ROI[35-2:35+2, 53-2:53+2])
    Rod1mmY = int(Rod1mmMaxPixelIndex % 4 + 35-2)
    Rod1mmX = int(Rod1mmMaxPixelIndex / 4 + 53-2)
    #print(Rod1mmY, Rod1mmX)
    Rod1mmLineProfile = VOI[:, Rod1mmY, Rod1mmX] / (PixelSizeXY ** 2 * ThicknessZ)
    Rod1mmMeanRC = np.mean(Rod1mmLineProfile / AvgActivityConcentration)
    Rod1mmSTDRC = np.sqrt((np.std(Rod1mmLineProfile / AvgActivityConcentration, ddof = 1) / Rod1mmMeanRC)**2 + UniformitySTD**2)


    Rod2mmMaxPixelIndex = np.argmax(ROI[51-4:51+4, 51-4:51+4])
    Rod2mmY = int(Rod2mmMaxPixelIndex % 8 + 51-4)
    Rod2mmX = int(Rod2mmMaxPixelIndex / 8 + 51-4)
    #print(Rod2mmY, Rod2mmX)
    Rod2mmLineProfile = VOI[:, Rod2mmY, Rod2mmX] / (PixelSizeXY ** 2 * ThicknessZ)
    Rod2mmMeanRC = np.mean(Rod2mmLineProfile / AvgActivityConcentration)
    Rod2mmSTDRC = np.sqrt((np.std(Rod2mmLineProfile / AvgActivityConcentration, ddof = 1) / Rod2mmMeanRC)**2 + UniformitySTD**2)


    Rod3mmMaxPixelIndex = np.argmax(ROI[54-6:54+6, 35-6:35+6])
    Rod3mmY = int(Rod3mmMaxPixelIndex % 12 + 54-6)
    Rod3mmX = int(Rod3mmMaxPixelIndex / 12 + 35-6)
    #print(Rod3mmY, Rod3mmX)
    Rod3mmLineProfile = VOI[:, Rod3mmY, Rod3mmX] / (PixelSizeXY ** 2 * ThicknessZ)
    Rod3mmMeanRC = np.mean(Rod3mmLineProfile / AvgActivityConcentration)
    Rod3mmSTDRC = np.sqrt((np.std(Rod3mmLineProfile / AvgActivityConcentration, ddof = 1) / Rod3mmMeanRC)**2 + UniformitySTD**2)


    Rod4mmMaxPixelIndex = np.argmax(ROI[39-8:39+8, 27-8:27+8])
    Rod4mmY = int(Rod4mmMaxPixelIndex % 16 + 39-8)
    Rod4mmX = int(Rod4mmMaxPixelIndex / 16 + 27-8)
    #print(Rod4mmY, Rod4mmX)
    Rod4mmLineProfile = VOI[:, Rod4mmY, Rod4mmX] / (PixelSizeXY ** 2 * ThicknessZ)
    Rod4mmMeanRC = np.mean(Rod4mmLineProfile / AvgActivityConcentration)
    Rod4mmSTDRC = np.sqrt((np.std(Rod4mmLineProfile / AvgActivityConcentration, ddof = 1) / Rod4mmMeanRC)**2 + UniformitySTD**2)


    Rod5mmMaxPixelIndex = np.argmax(ROI[27-10:27+10, 38-10:38+10])
    Rod5mmY = int(Rod5mmMaxPixelIndex % 20 + 27-10)
    Rod5mmX = int(Rod5mmMaxPixelIndex / 20 + 38-10)
    #print(Rod5mmY, Rod5mmX)
    Rod5mmLineProfile = VOI[:, Rod5mmY, Rod5mmX] / (PixelSizeXY ** 2 * ThicknessZ)
    Rod5mmMeanRC = np.mean(Rod5mmLineProfile / AvgActivityConcentration)
    Rod5mmSTDRC = np.sqrt((np.std(Rod5mmLineProfile / AvgActivityConcentration, ddof = 1) / Rod5mmMeanRC)**2 + UniformitySTD**2)

    '''
    plt.figure('Line Profile')
    plt.plot(Rod1mmLineProfile)
    plt.plot(Rod2mmLineProfile)
    plt.plot(Rod3mmLineProfile)
    plt.plot(Rod4mmLineProfile)
    plt.plot(Rod5mmLineProfile)
    plt.show()
    '''
    return Rod1mmMeanRC, Rod1mmSTDRC, Rod2mmMeanRC, Rod2mmSTDRC, Rod3mmMeanRC, Rod3mmSTDRC, Rod4mmMeanRC, Rod4mmSTDRC, Rod5mmMeanRC, Rod5mmSTDRC

def getAccuracyOfCorrections(Img, ZRange, XYRange, HotRegionMeanValue, HotRegionUniformitySTD):
    z, y, x = Img.shape
    VOI = np.zeros([int(ZRange), int(80), int(80)], dtype = 'float32')
    VOI = Img[int(z/2-ZRange-30):int(z/2-30), int(y/2-80/2+3):int(y/2+80/2+3), int(x/2-80/2-5):int(x/2+80/2-5)]
    '''
    plt.figure('VOI[0]')
    plt.imshow(VOI[0, :, :])
    plt.figure('VOI[7]')
    plt.imshow(VOI[7, :, :])
    plt.figure('VOI[14]')
    plt.imshow(VOI[14, :, :])
    plt.show()
    '''
    WaterVOI = np.zeros([int(ZRange), int(XYRange), int(XYRange)], dtype = 'float32')
    WaterVOI = VOI[:, int(30-XYRange/2):int(30+XYRange/2), int(31-XYRange/2):int(31+XYRange/2)]
    
    AirVOI = np.zeros([int(ZRange), int(XYRange), int(XYRange)], dtype = 'float32')
    AirVOI = VOI[:, int(50-XYRange/2):int(50+XYRange/2), int(52-XYRange/2):int(52+XYRange/2)]
    '''
    plt.figure('WaterVOI[0]')
    plt.imshow(WaterVOI[0, :, :])
    plt.figure('WaterVOI[14]')
    plt.imshow(WaterVOI[14, :, :])
    plt.figure('AirVOI[0]')
    plt.imshow(AirVOI[0, :, :])
    plt.figure('AirVOI[14]')
    plt.imshow(AirVOI[14, :, :])
    plt.show()
    '''
    WaterPixelList = []
    AirPixelList = []
    for i in range(int(ZRange)):
        for j in range(int(XYRange)):
            for k in range(int(XYRange)):
                if(np.sqrt(((j-XYRange/2)**2 + (k-XYRange/2)**2)) < XYRange/2):
                    WaterPixelList.append(WaterVOI[i, j, k])
                    AirPixelList.append(AirVOI[i, j, k])
    WaterMeanValue = np.mean(WaterPixelList)
    WaterSOR = WaterMeanValue / HotRegionMeanValue
    WaterSTD = np.sqrt((np.std(WaterPixelList, ddof = 1) / WaterMeanValue)**2 + UniformitySTD**2)
    AirMeanValue = np.mean(AirPixelList)
    AirSOR = AirMeanValue / HotRegionMeanValue
    AirSTD = np.sqrt((np.std(AirPixelList, ddof = 1) / AirMeanValue)**2 + UniformitySTD**2)

    '''
    # Method - Profile	
    WaterROI = np.mean(WaterVOI, axis = 0)
    WaterROIMinPixelIndex = np.argmin(WaterROI[:, :])
    WaterMinPixelY = int(WaterROIMinPixelIndex % 8)
    WaterMinPixelX = int(WaterROIMinPixelIndex / 8)
    print(WaterMinPixelY, WaterMinPixelX)
    WaterLineProfile = WaterVOI[:, WaterMinPixelY, WaterMinPixelX]
    WaterSTD = np.sqrt((np.std(WaterLineProfile, ddof = 1) / np.mean(WaterLineProfile))**2 + UniformitySTD**2)

    AirROI = np.mean(AirVOI, axis = 0)
    AirROIMinPixelIndex = np.argmin(AirROI[:, :])
    AirMinPixelY = int(AirROIMinPixelIndex % 8)
    AirMinPixelX = int(AirROIMinPixelIndex / 8)
    print(AirMinPixelY, AirMinPixelX)
    AirLineProfile = AirVOI[:, AirMinPixelY, AirMinPixelX]
    AirSTD = np.sqrt((np.std(AirLineProfile, ddof = 1) / np.mean(AirLineProfile))**2 + UniformitySTD**2)
  
    plt.figure('WaterLineProfile and AirLineProfile')
    plt.plot(WaterLineProfile)
    plt.plot(AirLineProfile)
    plt.show()
    '''
	'''
    plt.figure('WaterROI')
    plt.imshow(WaterROI)
    plt.figure('AirROI')
    plt.imshow(AirROI)
    plt.show()
    '''
    
    return WaterSOR, WaterSTD, AirSOR, AirSTD 

# Read Images from files
def readImg(SlicesNum, ColumnsNum, RowsNum, ImgPathList, dType = 'int16'):
    z, y, x = SlicesNum, ColumnsNum, RowsNum
    Img = np.zeros([z, y, x], dtype = dType)
    for i in range(z):
        ImgSlice = np.fromfile(ImgPathList[i], dtype = dType)
        ImgSlice = ImgSlice.reshape([y, x])
        Img[i,:,:] = ImgSlice
    # Well Counter
    Img = Img * 38.985
    return Img 

if __name__ == '__main__':
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Settings 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    CudaPSFImgFolder = './2.PET_Img_CudaPSF/'
    ImgPathList = []
    PixelSizeXY = 0.5
    ThicknessZ = 0.5
    InitialActivity = 102 * 37000
    InitialPreparationTime = 5 * 60
    AcqTime = 20 * 60
    HalfLife = 6588

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Calculation - CudaPSF
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for i in range(400):
        ImgPath = CudaPSFImgFolder + 'I' + str(i+1)
        ImgPathList.append(ImgPath)
    Img = readImg(400, 320, 320, ImgPathList)
    

    # Uniformity
    AvgActivityConcentration, MaxValue, MinValue, MeanValue, UniformitySTD = getUniformity(Img, 10/ThicknessZ, 22.5/PixelSizeXY, ThicknessZ, PixelSizeXY)
    print('AvgActivityConcentration, Max, Min, Mean, STD = ', AvgActivityConcentration, MaxValue, MinValue, MeanValue, UniformitySTD)


    # Recovery Coefficients
    Rod1mmMeanRC, Rod1mmSTDRC, Rod2mmMeanRC, Rod2mmSTDRC, Rod3mmMeanRC, Rod3mmSTDRC, Rod4mmMeanRC, Rod4mmSTDRC, Rod5mmMeanRC, Rod5mmSTDRC = getRC(Img, 10/ThicknessZ, 40/PixelSizeXY, ThicknessZ, PixelSizeXY, AvgActivityConcentration, UniformitySTD)
    print('Rod1mmMeanRC, Rod1mmSTDRC = ', Rod1mmMeanRC, Rod1mmSTDRC)
    print('Rod2mmMeanRC, Rod2mmSTDRC = ', Rod2mmMeanRC, Rod2mmSTDRC)
    print('Rod3mmMeanRC, Rod3mmSTDRC = ', Rod3mmMeanRC, Rod3mmSTDRC)
    print('Rod4mmMeanRC, Rod4mmSTDRC = ', Rod4mmMeanRC, Rod4mmSTDRC)
    print('Rod5mmMeanRC, Rod5mmSTDRC = ', Rod5mmMeanRC, Rod5mmSTDRC)
    
    # Accuracy of Corrections
    WaterSOR, WaterSTD, AirSOR, AirSTD = getAccuracyOfCorrections(Img, 7.5/ThicknessZ, 4/PixelSizeXY, MeanValue, UniformitySTD)
    print('WaterSOR, WaterSTD, AirSOR, AirSTD = ', WaterSOR, WaterSTD, AirSOR, AirSTD)


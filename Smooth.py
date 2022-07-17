#!/usr/bin/env python

import starfile
import pandas as pd
import os
import fire
from fire import core
import shutil
import imutils
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
pd.options.mode.chained_assignment = None
path_current=os.getcwd()
sorted_folder='star_fitting'
cryo_star='sperated_star_smooth'
folder='separeted_star'
image_folder='images'

class smooth:
    """
    \n This programe is to smooth the Euler angles\n
    For detailed descreption, please run the following commands
    python3 Smooth.py seperate -h
    python3 Smooth.py fitting -h
    python3 Smoooth.py plot -h
    python3 Smooth.py update -h

    """
    def seperate(self,files):
        """
        This command is to parse the file according to its name and will output the star files in the folder seperated_star
        python3 Smooth.py seperate <file>
        :param files: this is a star file
        """
        print('processing......')
        star_spa_all_orig=starfile.read(files)
        star_spa_all=pd.DataFrame([star_spa_all_orig])
        star_spa_all=star_spa_all.iloc[0,1]
        Tomonames=star_spa_all['rlnTomoName'].unique()
        after_spa_path = os.path.join(path_current, folder)
        if os.path.exists(after_spa_path):
            shutil.rmtree(after_spa_path)
        os.mkdir(after_spa_path)
        for Tomoname in Tomonames:
            output_name=Tomoname+'.star'
            output_path=os.path.join(after_spa_path,output_name)
            star_spa=star_spa_all[star_spa_all['rlnTomoName']==Tomoname]
            starfile.write(star_spa,output_path,overwrite=True)
        print('Seperating process is Done!!!')
        
        
    def fitting(self,foldername):
        """
        This command is to curve fitting process
        python3 Smooth.py fitting [foldername]
        :param foldername: this folder is produced from seperating process named seperated_star
        """
        print('processing......')
        sorted_path = os.path.join(path_current, sorted_folder)
        if os.path.exists(sorted_path):
            shutil.rmtree(sorted_path)
        os.mkdir(sorted_path)
        star_files=sorted(os.listdir(foldername))
        for star_file in star_files:
            if star_file[-5:]=='.star':
                star=pd.DataFrame()
                outputstar_name=star_file.replace('.star','_sorted.star')
                outputstar_smooth_name=star_file.replace('.star','_sorted_smooth.star')
                outputstar_path=os.path.join(sorted_path,outputstar_name)
                outputstar_smooth_path=os.path.join(sorted_path,outputstar_smooth_name)
                star=starfile.read(os.path.join(path_current,foldername,star_file))
                idx = star['rlnTomoParticleName'].str.split('/', expand=True)
                idx[1] = pd.to_numeric(idx[1])
                star['A']=idx[1]
                star=star.sort_values(by=['A']).reset_index(drop=True)
                star=star.drop(columns=['A'])
                starfile.write(star, outputstar_path,overwrite=True)
                def func(x,t,y):
                     return x[0] * t**3+x[1] * t**2 + x[2]*t+x[3]-y
                def func1(x, a, b, c,d):
                     return a * x**3+b * x**2 + c*x+d
                ## select the tube ID
                x0 = np.array([1.0, 1.0, 1,1])
                dupes = star['rlnHelicalTubeID'].unique()
                star2=pd.DataFrame()
                for i in dupes:
                    star1=star[star['rlnHelicalTubeID']==i]
                    if len(star1)>4:
                    ## relationship between x and y
                        popt1, pcov1 = curve_fit(func1, star1['rlnCoordinateX'], star1['rlnCoordinateY'])
                        y1=func1(star1['rlnCoordinateX'], *popt1)
                        star1['rlnCoordinateY_update']=y1     

                        if abs(max(star1['rlnAngleRot'])-min(star1['rlnAngleRot']))>300:
                            for i in star1.index:
                                if star1['rlnAngleRot'][i]<50:
                                       star1['rlnAngleRot'][i]=star1['rlnAngleRot'][i]+360                   
                        res_soft_l1 = least_squares(func, x0, loss='soft_l1', f_scale=2,args=(star1.index, star1['rlnAngleRot']))
                        prediction_rot=[]
                        y=[]
                        for i in star1.index:
                            y=res_soft_l1.x[0] * i**3+res_soft_l1.x[1] * i**2 + res_soft_l1.x[2]*i+res_soft_l1.x[3]
                            prediction_rot.append(y)
                        star1['rlnAngleRot_update']=prediction_rot

                        if abs(max(star1['rlnAngleTilt'])-min(star1['rlnAngleTilt']))>300:
                            for i in star1.index:
                                if star1['rlnAngleTilt'][i]<50:
                                       star1['rlnAngleTilt'][i]=star1['rlnAngleTilt'][i]+360
                        res_soft_l1 = least_squares(func, x0, loss='soft_l1', f_scale=2,args=(star1.index, star1['rlnAngleTilt']))
                        prediction_tilt=[]
                        y=[]
                        for i in star1.index:
                            y=res_soft_l1.x[0] * i**3+res_soft_l1.x[1] * i**2 + res_soft_l1.x[2]*i+res_soft_l1.x[3]
                            prediction_tilt.append(y)
                        star1['rlnAngleTilt_update']=prediction_tilt

                        if abs(max(star1['rlnAnglePsi'])-min(star1['rlnAnglePsi']))>300:
                            for i in star1.index:
                                if star1['rlnAnglePsi'][i]<0:
                                       star1['rlnAnglePsi'][i]=star1['rlnAnglePsi'][i]+360
                        res_soft_l1 = least_squares(func, x0, loss='soft_l1', f_scale=2,args=(star1.index, star1['rlnAnglePsi']))
                        prediction_psi=[]
                        y=[]
                        for i in star1.index:
                            y=res_soft_l1.x[0] * i**3+res_soft_l1.x[1] * i**2 + res_soft_l1.x[2]*i+res_soft_l1.x[3]
                            prediction_psi.append(y)
                        star1['rlnAnglePsi_update']=prediction_psi

                        star2=star2.append(star1)
                    else:
                        star1['rlnCoordinateY_update']=star1['rlnCoordinateY']
                        star1['rlnAngleRot_update']=star1['rlnAngleRot']
                        star1['rlnAngleTilt_update']=star1['rlnAngleTilt']
                        star1['rlnAnglePsi_update']=star1['rlnAnglePsi']
                        star2=star2.append(star1)
                starfile.write(star2, outputstar_smooth_path,overwrite=True)
            

        foldername=os.path.join(path_current,sorted_folder)
        star_file_updates=sorted(os.listdir(foldername))
        cryo_path = os.path.join(path_current, cryo_star)
        if os.path.exists(cryo_path):
            shutil.rmtree(cryo_path)
        os.mkdir(cryo_path)
        for star_file_update in star_file_updates:
            if star_file_update[-11:]=='smooth.star':
                star_file_update_path=os.path.join(path_current,sorted_folder,star_file_update)
                output_name=star_file_update
                output_path=os.path.join(cryo_path,output_name)
                star3=starfile.read(star_file_update_path)
                star3['rlnCoordinateY']= star3['rlnCoordinateY_update']
                star3['rlnAngleRot']= star3['rlnAngleRot_update']
                star3['rlnAngleTilt']= star3['rlnAngleTilt_update']
                star3['rlnAnglePsi']= star3['rlnAnglePsi_update']
                star3.drop(['rlnCoordinateY_update','rlnAngleRot_update','rlnAngleTilt_update','rlnAnglePsi_update'],axis=1, inplace=True)
                starfile.write(star3,output_path,overwrite=True)
        print('Fitting process is Done!!!')
    
    def plot(self,foldername):
        """
        This command is to plot the star files
        python3 Smooth.py combine [foldername]
        :param foldername: this folder is produced from fitting processing named star_fitting
        
        """
        image_path = os.path.join(path_current, image_folder)
        if os.path.exists(image_path):
            shutil.rmtree(image_path)
        os.mkdir(image_path)
        star_file_updates=sorted(os.listdir(foldername))
        #print(star_file_updates)
        for star_file_update in star_file_updates:
            if star_file_update[-11:]=='smooth.star':
                star_file_update_path=os.path.join(path_current,sorted_folder,star_file_update)
                image_name=star_file_update.replace('_sorted_smooth.star','.eps')
                image_output=os.path.join(image_path,image_name)
                star_smooth=starfile.read(star_file_update_path)
                star_smooth1=star_smooth[['rlnAngleRot','rlnAngleTilt','rlnAnglePsi','rlnAngleRot_update','rlnAngleTilt_update','rlnAnglePsi_update']]
                star_smooth2=star_smooth[['rlnCoordinateX','rlnCoordinateY','rlnCoordinateY_update']]
                #print(star_smooth1['rlnHelicalTubeID'].unique())
                import matplotlib.pyplot as plt
                plt.figure(figsize=(20, 25))
                plt.subplot(211)
                plt.scatter(star_smooth.index,star_smooth['rlnAngleRot'],s=120, marker='+', c='r',label='Rot_Orig')
                plt.scatter(star_smooth.index,star_smooth['rlnAngleRot_update'],s=100, marker='D',facecolors='none', edgecolors='r',label='Rot_smooth')
                plt.scatter(star_smooth.index,star_smooth['rlnAngleTilt'],s=120, marker='+', c='black',label='Tilt_Orig')
                plt.scatter(star_smooth.index,star_smooth['rlnAngleTilt_update'],s=100, marker='D',facecolors='none', edgecolors='black',label='Tilt_smooth')
                plt.scatter(star_smooth.index,star_smooth['rlnAnglePsi'],s=120, marker='+', c='blue',label='Psi_Orig')
                plt.scatter(star_smooth.index,star_smooth['rlnAnglePsi_update'],s=100, marker='D',facecolors='none', edgecolors='blue',label='Psi_smooth')
                plt.legend(loc='best',fontsize=30)
                plt.xlabel('index',fontsize=30)
                plt.ylabel('Euler Angles',fontsize=30)
                #plt.xticks(np.arange(star_smooth1.index[0],star_smooth.index[len(star_smooth1)-1]+1,1),fontsize=20)
                plt.xticks(fontsize=30)
                plt.yticks(fontsize=30)
                plt.title(image_name,fontsize=20)
                plt.subplot(212)
                plt.scatter(star_smooth['rlnCoordinateX'],star_smooth['rlnCoordinateY'],s=120, marker='+', c='black',label='Orig')
                plt.scatter(star_smooth['rlnCoordinateX'],star_smooth['rlnCoordinateY_update'],s=100, marker='D',facecolors='none', edgecolors='b',label='Smooth')
                plt.legend(loc='best',fontsize=20)
                plt.xlabel('x',fontsize=30)
                plt.ylabel('y',fontsize=30)
                #plt.xticks(np.arange(star_smooth1.index[0],star_smooth.index[len(star_smooth1)-1]+1,1),fontsize=20)
                plt.xticks(fontsize=30)
                plt.yticks(fontsize=30)
                plt.tight_layout()
                plt.savefig(image_output, format='eps')
    
    
    
    def update(self,foldername,output_name):
        """
        This command is to combine all the stars into one for single particel analysis
        python3 Smooth.py update [foldername] [output_name]
        :param foldername: this folder is produced from fitting process named sperated_star_smooth
        :param outputname: this is the file you want to update that should be the same with the seperating process
        """
        print('Processsing.......')
        star_files=sorted(os.listdir(foldername))
        output_path=os.path.join(path_current,output_name)
        new_star=pd.DataFrame()
        for star_file in star_files:
            input_path=os.path.join(foldername,star_file)
            star=starfile.read(input_path)
            new_star=new_star.append(star)
        starfile.write(new_star,output_path,overwrite=True)
        print('Combining process is Done!!!')
        
        
def Display(lines, out):
    text = "\n".join(lines) + "\n"
    out.write(text)


if __name__ == "__main__":
    core.Display = Display
    fire.Fire(smooth)


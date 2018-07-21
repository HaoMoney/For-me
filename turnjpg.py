#coding=utf-8
from PIL import Image
import os.path
import glob
import numpy as np

def convertjpg(jpgfile,outdir):
    img=Image.open(jpgfile)
    #new_img=img.rotate(angle)                  
    img.save(outdir+os.path.splitext(os.path.split(jpgfile)[1])[0]+".jpg") 
for jpgfile in glob.glob("cloud_hq/*"):             
    convertjpg(jpgfile,"hq_cloud/")   
for jpgfile in glob.glob("non-cloud_hq/*"):       
    convertjpg(jpgfile,"hq_cloud_non/")   

#coding=utf-8
from PIL import Image
import os.path
import glob
import numpy as np

def convertjpg(jpgfile,outdir,angle):
    img=Image.open(jpgfile)
    new_img=img.rotate(angle)                  
    new_img.save(outdir+os.path.splitext(os.path.split(jpgfile)[1])[0]+str(angle)+".jpg") 
for jpgfile in glob.glob("cloud/*"):             
    convertjpg(jpgfile,"rot90/",90)   
    convertjpg(jpgfile,"rot180/",180)   
    convertjpg(jpgfile,"rot270/",270)   
for jpgfile in glob.glob("non-cloud/*"):       
    convertjpg(jpgfile,"rot90_non/",90)   
    convertjpg(jpgfile,"rot180_non/",180)   
    convertjpg(jpgfile,"rot270_non/",270)   

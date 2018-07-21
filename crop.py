#coding=utf-8
from PIL import Image
im = Image.open("cloud.jpg")
# 图片的宽度和高度
img_size = im.size
print("图片宽度和高度分别是{}".format(img_size))
# 把图片平均分成100块
w = img_size[0]/10.0 #图片宽
h = img_size[1]/10.0 #图片高
for i in range(10):
    for j in range(10):
        x=j*w
        y=i*h
        region = im.crop((x, y, x+w, y+h))
        region.save("./crop_image/crop3_average-"+str(i)+"_"+str(j)+".jpg") #保存图片，命名为"crop_average行_列.jpg"
        

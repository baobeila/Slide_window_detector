import glob
import cv2

'''cut the img_480*640 into  img_64*64,the sampling interval is 30'''
# src_dir = "/home/yangzekun/下载/Micro surface defect database/SDI"  # the scr_img
src_dir = "/home/yangzekun/下载/Micro surface defect database/SPdI"  # the scr_img
filepath = glob.glob(src_dir + '/*.bmp')  # return a list
count = 0
flag = False
# for i in range(len(filepath)):
for i in range(len(filepath)-1,-1,-1):
    img_src = cv2.imread(filepath[i])
    for y in range(0, (480 - 64), 30):
        for x in range(0, (640 - 64), 30):
            img_cut = img_src[y:y + 64, x:x + 64]# 裁剪坐标为[y0:y1, x0:x1]
            save_dir = "/home/yangzekun/PycharmProjects/slidewin_detect/data/train/to_crop/cropa{}.jpg".format(count)
            count += 1
            if count==130:
                flag = True
                break
            else:
                cv2.imwrite(save_dir, img_cut)
        if flag == True:
            break
    if flag == True:
        break



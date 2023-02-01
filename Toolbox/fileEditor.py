import shutil

i = 599
for x in range(0, 41, 5):
    
    filepath = r'/Users/brianprzezdziecki/Research/Mechatronics/Anomaly_Detection/DIW_Frames/Tip_Detection/T_6f/frame' + str(x) + '.jpg'
    original = filepath
    target = r'/Users/brianprzezdziecki/Research/Mechatronics/Anomaly_Detection/TipDatasetv1/image' + str(i) + '.jpg'
    i+=1
    shutil.copyfile(original, target)




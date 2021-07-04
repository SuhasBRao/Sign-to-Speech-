import tensorflow as tf
import cv2
import numpy as np

classes = {0:'1',1:'2',2:'3', 3:'4', 4:'5', 5:'6', 6:'7', 7:'8',
           8:'9',9:'A',10:'B',11:'C',12:'D',13:'E',14:'F',15:'G',16:'H',17:'I',
           18:'J',19:'K',20:'L',21:'M',22:'N',23:'O',24:'P',25:'Q',26:'R',27:'S',28:'T',29:'U',
           30:'V',31:'W',32:'X',33:'Y',34:'Z'}

reloaded = tf.keras.models.load_model('C:/Users/suhas/Downloads/best_model_ISL.h5') # this loads the model saved in my local machine                                                                                     laptop at specified location
img = cv2.imread('cutout.jpg')
img = cv2.resize(img,(150,150))
cv2.imshow('Input',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# reshape the img captured to feed it to the model
img = np.reshape(img,(1,img.shape[0], img.shape[1],-1))
pred = reloaded.predict(img)
val = np.argmax(pred)
print(val)
print(classes[val])
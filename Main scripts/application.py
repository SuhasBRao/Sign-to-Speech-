'''
Main python script that creates an interactive screen and allows the user to either
show gestures as live action or to upload a pre-existing image (Black and white) for
prediction.
There are mainly 3 types of predictions
[1] number prediction - we use numbers_model.h5 
[2] alphabet prediction - we use alpha_model.h5
[3] words prediction - we use best_model.h5
Respective model files are loaded based on what the user wants to predict.
'''

################################################################################
##################### Import Necessary models ##################################
try:
    import os   # accessing folder paths
    import sys
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.image as img
    import cv2 
    import pyttsx3

    import tkinter as tk
    from PIL import ImageTk, Image
    from tkinter import filedialog
    
    import tensorflow as tf
    
except ModuleNotFoundError as e:
    print()
    print(e.name)
    print("[ERROR] It seems the above module was not found!")
    sys.exit()
    
################################################################################
##### Creating a dictionary that is later used for prediction #################
# The dictionary contains three classes 
# [1] num_classes - used for number prediction
# [2] alpha_classes - used for alphabet prediction
# [3] words_class - used for word prediction
num_classes = {1:'1',2:'2',3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8',
           9:'9'}
alpha_classes = {1:'A',2:'B',3:'C',4:'D',5:'E',6:'F',7:'G',8:'H',9:'I',
           10:'J',11:'K',12:'L',13:'M',14:'N',15:'O',16:'P',17:'Q',18:'R',19:'S',20:'T',21:'U',
           22:'V',23:'W',24:'X',25:'Y',26:'Z'}

words_class = {1:'All_The_Best', 2:'Hi!!', 3: 'I_Love_you', 4: 'No', 5:'Super!!', 6:'Yes'}
################################################################################

########## Few necessary variables ##############
background = None
accumulated_weight = 0.7
mask_color = (0.0,0.0,0.0)

# During Live prediction we need a portion of the screen 
# where the user can show the gestures. 
# This portion is Region of Interest(ROI). 
# Here we set the boundary for ROI in pixels.
ROI_top = 100
ROI_bottom = 300
ROI_right = 300
ROI_left = 500
#################################################

# This function is used to calculate accumulated_weights in the frame
def cal_accum_avg(frame, accumulated_weight):

    global background
    
    if background is None:
        background = frame.copy().astype("float")
        return None

    cv2.accumulateWeighted(frame, background, accumulated_weight)

# This function segments the hand region found in the frame, if not found returns None.
def segment_hand(frame, threshold=50):
    global background
    
    diff = cv2.absdiff(background.astype("uint8"), frame)

    
    _ , thresholded = cv2.threshold(diff, threshold, 255,cv2.THRESH_BINARY)
    
    edges = cv2.Canny(thresholded, threshold1= 50, threshold2=250)
    cv2.imshow('edges',thresholded)
    
     #Fetching contours in the frame (These contours can be of hand
    #or any other object in foreground) …

    contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    # If length of contours list = 0, means we didn't get any
    #contours...
    if len(contours) == 0:
        return None
    else:
        # The largest external contour should be the hand
        # contour_info = [(c, cv2.contourArea(c),) for c in contours[1]]

        #cntrs, heirs = cv2.findContours(thresholded.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        contour_info = [(c, cv2.contourArea(c),) for c in contours]
        #for c in contours[1]:
        #    contour_info.append((c,cv2.contourArea(c),))
        
        hand_segment_max_cont = max(contours, key=cv2.contourArea)
        
        # Returning the hand segment(max contour) and the
  # thresholded image of hand and contour_info list
    return (thresholded, hand_segment_max_cont, contour_info)

def predict():
    '''
    This function is used for live predcition, When the user shows gestures
    Directly to the camera. 
    The gestures predicted are show on screen as well
    as are said aloud as speech.
    ####################################################################
    If you want to stop the live camera press 'esc' key during run time.
    ####################################################################
    '''
    global text_to_speak, model
    
    cam = cv2.VideoCapture(0)
    num_frames =0
    pred = None
    
    # Below loop reads live gestures from the camera and tries to predict.
    # One can use Esc key to close the live camera 
    while True:
        ret, frame = cam.read()

        # flipping the frame to prevent inverted image of captured
        #frame...
        
        frame = cv2.flip(frame, 1)

        frame_copy = frame.copy()

        # ROI from the frame
        roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]

        gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)

        # We fetch the background from initial 70 frames 
        # After 70 frames, during prediction we subtract his background from 
        # every frame to get the foreground gesture
        if num_frames < 70:
            
            cal_accum_avg(gray_frame, accumulated_weight)
            
            cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT",
    (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        
        else: 
            # segmenting the hand region
            hand = segment_hand(gray_frame)
            
            # Checking if we are able to detect the hand...
            if hand is not None:
                
                thresholded, hand_segment,contour_info = hand

                # Drawing contours around hand segment
                cv2.drawContours(frame_copy, [hand_segment + (ROI_right,ROI_top)], -1, (255, 0, 0),1)
                
                cv2.imshow("Thesholded Hand Image", thresholded)
                
                thresholded = cv2.resize(thresholded, (64, 64))
                thresholded = cv2.cvtColor(thresholded,cv2.COLOR_GRAY2RGB)
                thresholded = np.reshape(thresholded,(1,thresholded.shape[0],thresholded.shape[1],3))

                prev = text_to_speak[np.argmax(pred) + 1]
                pred = model.predict(thresholded)
                
                #print(pred)
                cv2.putText(frame_copy, text_to_speak[np.argmax(pred) + 1],
    (300, 45), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0,0,255), 2)
                
                if text_to_speak[np.argmax(pred) + 1] != prev:
                    engine = pyttsx3.init()
                    engine.say(text_to_speak[np.argmax(pred) + 1])
                    engine.runAndWait()
                    prev_num_frames = num_frames
                        
                
        # Draw ROI on frame_copy
        cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right,
        ROI_bottom), (255,128,0), 3)

        # incrementing the number of frames for tracking
        num_frames += 1
        #print(pred)
        #if pred != None:
            

        # Display the frame with segmented hand
        cv2.putText(frame_copy, "Indian sign language recognition_ _ _",(10, 20), 
                    cv2.FONT_ITALIC, 0.5, (51,255,51), 1)
        cv2.imshow("Sign Detection", frame_copy)


        # Close window with Esc
        k = cv2.waitKey(1) & 0xFF

        if k == 27:
            break

    # Release the camera and destroy all the windows
    cam.release()
    cv2.destroyAllWindows()

def load_img():
    global img, image_data
    for img_display in frame.winfo_children():
        img_display.destroy()

    image_data = filedialog.askopenfilename(initialdir="/", title="Choose an image",
                                       filetypes=(("all files", "*.*"), ("jpg files", "*.jpg")))
    basewidth = 150
    img = Image.open(image_data)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    file_name = image_data.split('/')
    panel = tk.Label(frame, text= str(file_name[len(file_name)-1]).upper()).pack()
    panel_image = tk.Label(frame, image=img).pack()

def classify_loaded_image():
    global model
    original = cv2.imread(image_data)
    thresholded = cv2.resize(original, (64, 64))
    
    thresholded = np.reshape(thresholded,(1,thresholded.shape[0],thresholded.shape[1],3))

    pred = model.predict(thresholded)
        
    string = text_to_speak[np.argmax(pred) + 1]

    table = tk.Label(frame, text="Predicted sign ").pack()

    result = tk.Label(frame, text= string.upper()).pack()

def what_to_predict():
    print('What do you want to predict? \n')
    print("[1] Numbers \n[2] Words \n[3] Alphabets.")
    user_choice = int(input('Enter 1 , 2 or 3 \n'))
    matching_dict = {1:'Numbers', 2:'Words', 3:'Alphabets'}
    user_choice_name = matching_dict[user_choice]
    return user_choice_name

def check_for_model_file(file_name):
    file_exists = os.path.exists("Main scripts\\Trained Models\\" + file_name)
    print(file_exists)
    if not file_exists:
        raise file_name + "Not Found. Please ensure the model file is present in Trained models dir!!"
    return None
             
# {
# Driver Code starts
################################################################################################################
if __name__ == "__main__":
    
    # Loading pre-trained model based on user choice and assign the expected 
    # text_to_speak at the same time.
    # The text_to_speak is used by pyttsx3 at run time to speak out the
    # predicted sign.
    user_choice = what_to_predict()
    
    if user_choice == "Numbers":
        check_for_model_file("numbers_model.h5")
        model = tf.keras.models.load_model('Main scripts\\Trained Models\\numbers_model.h5')
        text_to_speak = num_classes
        
    elif user_choice == "Words":
        check_for_model_file("best_model.h5")
        model = tf.keras.models.load_model('Main scripts\\Trained Models\\best_model.h5')
        text_to_speak = words_class
    
    elif user_choice == "Alphabets":
        check_for_model_file("alpha_model.h5")
        model = tf.keras.models.load_model('Main scripts\\Trained Models\\alpha_model.h5')
        text_to_speak = alpha_classes
        

    ##########################################################################################
    ################ We create an interactive window using Tkinter library ###################
    root = tk.Tk()
    root.title('Sign predictor')
    
    root.resizable(False, False)

    tit = tk.Label(root, text="Sign predictor", padx=25, pady=6, font=("", 12)).pack()

    canvas = tk.Canvas(root, height=400, width=600, bg='#76c3fb')
    canvas.pack()

    frame = tk.Frame(root, bg='#d776fb')
    frame.place(relwidth=0.8, relheight=0.7, relx=0.1, rely=0.1)

    load_image_btn = tk.Button(root, text='Load Image',
                            padx=20, pady=10,
                            fg="white", bg="#8c04b5", command=load_img)
    load_image_btn.pack(side=tk.LEFT)


    live_predcition_btn = tk.Button(root, text='Live prediction',
                            padx=20, pady=10,
                            fg="white", bg="#0a7c2e",command=predict)
    live_predcition_btn.pack(side=tk.RIGHT)


    classify_btn = tk.Button(root, text='Classify loaded Image',
                            padx=20, pady=10,
                            fg="white", bg="#0475b5",command=classify_loaded_image)
    classify_btn.pack(side=tk.LEFT)
        
    
    root.mainloop()
    ###########################################################################################
    
################################################################################################################
# } Driver Code ends


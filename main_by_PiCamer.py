import RPi.GPIO as GPIO
import time
from picamera import PiCamera
from time import sleep
from PIL import Image

# Importing the Opencv Library
import cv2
import numpy as np
import argparse

#pygame
import pygame
from pygame.locals import *
import os
import subprocess

#lcd display
import lcddriver # *****

####sevol control####
#import sys
#import pigpio

#run cmd
import subprocess

# Plate databases [Masta]
licensedList = ["123456", "654624", "213456", "2345653"]
stolenList = ["626233", "234324", "546546", "413462"]
#unLicensedList = ["6161", "1251", "6316", "134135"]

######################################################################################

#initialize
# Set for broadcom numbering so that GPIO# can be used as reference
GPIO.setmode(GPIO.BCM)  

LCD = lcddriver.lcd()

GPIO.setup(17, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(27, GPIO.IN, pull_up_down=GPIO.PUD_UP)

##IR sersor signal input   
GPIO.setup(19, GPIO.IN)

GPIO.setup(16, GPIO.OUT)   #Green_led 
GPIO.setup(20, GPIO.OUT)   #red_led  
GPIO.setup(21, GPIO.OUT)   #Yellow_led

#camera = PiCamera()
camera = cv.VideoCapture(0, cv.CAP_DSHOW) #captureDevice = camera   

os.putenv('SDL_VIDEODRIVER', 'fbcon') # Display on piTFT
os.putenv('SDL_FBDEV', '/dev/fb1')
os.putenv('SDL_MOUSEDRV', 'TSLIB') # Track mouse clicks on piTFT
os.putenv('SDL_MOUSEDEV', '/dev/input/touchscreen')
pygame.init()#pygame initialization
pygame.mouse.set_visible(False)
#RGB color
WHITE = 255, 255, 255
BLACK = 0,0,0
RED = 255,0,0
GREEN = 0,255,0
start_time=time.time()#tiem when program start
screen = pygame.display.set_mode((320, 240))

my_font = pygame.font.Font(None,25)


########################################################################################

#functions
def imageCap():
  camera.resolution = (2592, 1944)
  #camera.rotation=180
  camera.framerate = 15
  camera.start_preview()
  camera.brightness = 65#default50
  camera.rotation=0
  time.sleep(2)
  camera.capture('image.jpg')
  camera.stop_preview()




def imageProcess():
  ##number plate localization and background delete
  # Importing NumPy,which is the fundamental package for scientific computing with Python
  global start_time
  start_time=time.time()
  # Reading Image
  img_0 = cv2.imread("image.jpg")
  img = img_0[900:1700,500:1700] #can be restrict to smaller region
  cv2.imwrite('display0.jpg',img)

  # RGB to Gray scale conversion
  img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

  # Noise removal with iterative bilateral filter(removes noise while preserving edges)
  noise_removal = cv2.bilateralFilter(img_gray,9,75,75)

  # Histogram equalisation for better results
  equal_histogram = cv2.equalizeHist(noise_removal)


  # Morphological opening with a rectangular structure element
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
  morph_image = cv2.morphologyEx(equal_histogram,cv2.MORPH_OPEN,kernel,iterations=15)

  # Image subtraction(Subtracting the Morphed image from the histogram equalised Image)
  sub_morp_image = cv2.subtract(equal_histogram,morph_image)

  # Thresholding the image
  ret,thresh_image = cv2.threshold(sub_morp_image,0,255,cv2.THRESH_OTSU)


  # Applying Canny Edge detection
  canny_image = cv2.Canny(thresh_image,250,255)

  canny_image = cv2.convertScaleAbs(canny_image)

  # dilation to strengthen the edges
  kernel = np.ones((3,3), np.uint8)
  # Creating the kernel for dilation
  dilated_image = cv2.dilate(canny_image,kernel,iterations=1)

  # Finding Contours in the image based on edges
  contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  contours= sorted(contours, key = cv2.contourArea, reverse = True)[:10]
  # Sort the contours based on area ,so that the number plate will be in top 10 contours
  screenCnt = None
  # loop over our contours
  loop = 1
  for c in contours:
    print ("loop #: " + str(loop))
    loop = loop+1
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.06 * peri, True)  # Approximating with 6% error
    print ("approx: " + str(len(approx)))
    # if our approximated contour has four points, then
    # we can assume that we have found our screen
    if len(approx) == 4:  # Select the contour with 4 corners
      screenCnt = approx
      top_left = approx[0][0] #[x y]
      top_right = approx[1][0]
      bottom_left = approx[2][0]
      bottom_right = approx[3][0]
    
      top_idx = min(top_left[1], top_right[1])
      bottom_idx = max(bottom_left[1], bottom_right[1])
      left_idx=min(min(top_left[0], top_right[0]),min(bottom_left[0], bottom_right[0]))
      right_idx=max(max(top_left[0], top_right[0]),max(bottom_left[0], bottom_right[0]))

      print ("Yay, find one")
      break

  ## Masking the part other than the number plate
  mask = np.zeros(img_gray.shape,np.uint8)
  new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
  new_image = cv2.bitwise_and(img,img,mask=mask)

  # Histogram equal for enhancing the number plate for further processing
  y,cr,cb = cv2.split(cv2.cvtColor(new_image,cv2.COLOR_BGR2YCR_CB))
  # Converting the image to YCrCb model and splitting the 3 channels
  y = cv2.equalizeHist(y)
  # Applying histogram equalisation
  final_image = cv2.cvtColor(cv2.merge([y,cr,cb]),cv2.COLOR_YCR_CB2BGR)
  # Merging the 3 channels

  #cv2.namedWindow("12_Extract",cv2.WINDOW_NORMAL)
  #print(new_image.shape)
  final_new_image = new_image[top_idx:bottom_idx,left_idx:right_idx ]
  print(final_new_image.shape)
  #cv2.imshow("12_Extract", final_new_image)

  cv2.imwrite('result1.jpg',new_image)
  cv2.imwrite('result2.jpg',final_new_image)

  im = final_new_image
  im[np.where((im <[20,20,20]).all(axis = 2))] = [255,255,255]

  gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
  ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 6))
  binl = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
  open_out = cv2.morphologyEx(binl, cv2.MORPH_OPEN, kernel)
  cv2.bitwise_not(open_out, open_out)
  #cv2.namedWindow("Transfered",cv2.WINDOW_NORMAL)
  #cv2.imshow("Transfered", open_out)
  cv2.imwrite('output1.jpg', open_out)


  

def correctSkew():
  ###correct Skew

  # load the image from disk
  image = cv2.imread("output1.jpg")

  # convert the image to grayscale and flip the foreground
  # and background to ensure foreground is now "white" and
  # the background is "black"
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  gray = cv2.bitwise_not(gray)
   
  # threshold the image, setting all foreground pixels to
  # 255 and all background pixels to 0
  thresh = cv2.threshold(gray, 0, 255,
    cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

  # grab the (x, y) coordinates of all pixel values that
  # are greater than zero, then use these coordinates to
  # compute a rotated bounding box that contains all
  # coordinates
  coords = np.column_stack(np.where(thresh > 0))
  angle = cv2.minAreaRect(coords)[-1]
   
  # the `cv2.minAreaRect` function returns values in the
  # range [-90, 0); as the rectangle rotates clockwise the
  # returned angle trends to 0 -- in this special case we
  # need to add 90 degrees to the angle
  if angle < -45:
    angle = -(90 + angle)
   
  # otherwise, just take the inverse of the angle to make
  # it positive
  else:
    angle = -angle

  # rotate the image to deskew it
  (h, w) = image.shape[:2]
  center = (w // 2, h // 2)
  M = cv2.getRotationMatrix2D(center, angle, 1.0)
  rotated = cv2.warpAffine(image, M, (w, h),
    flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


  cv2.imwrite("output2.jpg", rotated)



def image_resize(filename, mwidth, mheight):
    image = Image.open(filename)
    w,h = image.size
    if w<=mwidth and h<=mheight:
        print(filename,'is OK.')
        return 
    if (1.0*w/mwidth) > (1.0*h/mheight):
        scale = 1.0*w/mwidth
        new_im = image.resize((int(w/scale), int(h/scale)), Image.ANTIALIAS)
 
    else:
        scale = 1.0*h/mheight
        new_im = image.resize((int(w/scale),int(h/scale)), Image.ANTIALIAS)     

    
    new_im.save('display.jpg')
    new_im.close()

def displayControl():
  #pi_hw = pigpio.pi()  # connect to pi gpio daemon
  #up_t=0.00225 #pulse duration
  #T=up_t+0.02 #1 period time
  #f=1/T #46.51Hz
  #dc=(up_t/T)*1000000
  #pi_hw.hardware_PWM(13,f,dc)
  #time.sleep(3)
  #pi_hw.hardware_PWM(13,0,0)


  #time_servo=time.time()#
  flag=1
  count=0
  try:
    while flag:
      time.sleep(0.2)

      screen.fill(BLACK) # Erase the Work space
      time.sleep(0.2)  # Without sleep, no screen output!
      display="Number plate is: " + text
      text_surface = my_font.render(display, True, WHITE)#display Left servo History coloum
      rect = text_surface.get_rect(center=(160,30))
      screen.blit(text_surface, rect)

      #dispaly the captured image on piTFT
      image_resize('display0.jpg', 160, 160)
      imDisplay = pygame.image.load("display.jpg")
      #draw rectangle
      imRec = imDisplay.get_rect()
      #initil position of two balls
      imRec.right=240
      imRec.top=60
      screen.blit(imDisplay, imRec) # Combine surface with workspace surface
      
    
      if(text in licensedList): #test whether match with the one in database
        display="Allow Pass"
        LCD.lcd_display_string("Allow Pass", 1)
        GPIO.output(16, 1)  #Green Led
        text_surface = my_font.render(display, True, WHITE)#display Left servo History coloum
        rect = text_surface.get_rect(center=(160,210))
        screen.blit(text_surface, rect)
        pygame.display.flip()#dispaly on actual screen 
     
      
      elif ( not GPIO.input(27) ):
          print (" ") 
          print "Button 27 has been pressed"
          flag=False

           
        elif(text in stolenList):
          display="Not Allow"
          LCD.lcd_display_string("Not Allow", 1)
          LCD.lcd_display_string("Stolen Car", 2) 
          GPIO.output(21, 1)   # Yellow Led
          text_surface = my_font.render(display, True, WHITE)#display Left servo History coloum
          rect = text_surface.get_rect(center=(160,210))
          screen.blit(text_surface, rect)
          pygame.display.flip()#dispaly on actual screen
          
        else:
          display="Not Allow"
          LCD.lcd_display_string("Not Allow", 1)
          GPIO.output(20, 1)  # Red Led
          LCD.lcd_display_string("Unlicensed Car", 2) 
          text_surface = my_font.render(display, True, WHITE)#display Left servo History coloum
          rect = text_surface.get_rect(center=(160,210))
          screen.blit(text_surface, rect)
          pygame.display.flip()#dispaly on actual screen
        if((time.time()-time_servo)>5):
            flag=False

        
    screen.fill(BLACK) # Erase the Work space
    pygame.display.flip()#dispaly on actual screen
  except KeyboardInterrupt:
      pass

  #pi_hw.stop() #close pi gpio DMA resources

#######################################################################################

#main loop

flag=True
while flag:
    screen.fill(BLACK) # Erase the Work space
    time.sleep(0.2)  # Without sleep, no screen output!
    display="Welcome to 677 Parking"
    text_surface = my_font.render(display, True, WHITE)#display Left servo History coloum
    rect = text_surface.get_rect(center=(160,60))
    screen.blit(text_surface, rect)
    pygame.display.flip()#dispaly on actual screen 

    if ( not GPIO.input(19) ):#when button pressed pin connected to ground, GPIO.input(17)=0;
        print (" ") 
        print "IR sensor1!"
        imageCap()
        imageProcess() #start_time here
        correctSkew()
        ##OCR recognize
        #use command line ro do ocr
        cmd="tesseract output2.jpg file -l eng -psm 7" 
        #english  single line
        print subprocess.check_output(cmd, shell=True)
        #monitor the image processing time
        #print ("time for image processing and recognition is:" + str(time.time()-start_time))

        f = open("file.txt","r")
        text = f.readline()
        text=text.replace("\n", "")
        f.close()

        displayControl()  #lift up servo if number plate is mach
                        #display not allow if number plate not match



    
                                                                                                                
    if ( not GPIO.input(27) ):
        print (" ") 
        print "Button 27 has been pressed system out"
        flag=False


GPIO.cleanup()

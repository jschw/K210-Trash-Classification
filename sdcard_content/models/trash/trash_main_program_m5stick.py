# Trash Classifier - By: Julian - Di Mai 5 2020

# ===== Library imports =====
import sensor, image, time
import random
import utime

import lcd
from Maix import GPIO
from fpioa_manager import *

import KPU as kpu

from machine import I2C


# ===== Configuration =====

# Operation mode
# Comment out when set in boot.py
# 0 -> continuous/test , 1 -> single shot , 2 -> multi shot
op_mode = 2

# Number of images to take in multi shot mode
img_num_mshot = 10

# Image logging
img_logging = False
img_log_path = '/sd/saved_img'
# Note: Logging is only available in mode 0 and 1

# Path to model file
kmodel_path = '/sd/models/trash/trash.kmodel'

# Prediction threshold
threshold = 0.7




# ===== Functions =====

# Generates a random filename number
# Checks if a file with the same name already exists and generate a new number
def get_filename(path='/sd'):
    filelist = os.listdir(path)
    name_found = False

    while(True):
        rand_num = random.randint(1000,5000)

        for f in filelist:
            if f in str(rand_num):
                name_found = True
                break

        if name_found==True:
            name_found = False
        else: return ( str(rand_num))

# Save two image: Raw image and image with annotation
# Default annotation -> 'None (0.0 %)' at (10,10)
def save_image(src_img, path, annotation='None', x=10, y=10):
    src_img.save(path + '_001.bmp')
    print('Saved raw image to: ' + path + '_001.bmp')
    src_img.draw_string(x,y, annotation, color=(255,0,0), scale=1)
    src_img.save(path + '_002.bmp')
    print('Saved annotated image to: ' + path + '_002.bmp')


# ===== Inits =====
# Init LCD
lcd.init()
lcd.direction(0x60)
lcd.clear()
lcd.draw_string(10,10,'Abfall Klassifizierung')
lcd.draw_string(10,30,'Lade Modell ' + kmodel_path + ' ...')

# Init camers
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_windowing((224, 224))
#sensor.set_vflip(1)
sensor.run(1)

# Init button (A, front)
fm.register(board_info.BUTTON_A, fm.fpioa.GPIO1)
btn_a = GPIO(GPIO.GPIO1, GPIO.IN, GPIO.PULL_UP)


# Load model
task = kpu.load(kmodel_path)

img = image.Image()
img_icon = image.Image()
img_icon_tmp = image.Image()


# ===== Configure trash classification specific parameters =====

# Class label vector
# Last entry is the label for values below the threshold -> unknown
class_labels = ['Pappe', 'Glas', 'Metall', 'Papier', 'Kunststoffflasche', 'Kunststoff', 'Restmuell']



# ===== Main program loop =====
first_cycle = True
pause = False


while(True):
    btn_trigger = False
    

    # Switch to mode
    if op_mode==0:
        clk = time.clock()
        
        # Operation mode continuous
        while(True):
            clk.tick()
            
            # Take image
            img = sensor.snapshot()

            # Feed forward
            fmap = kpu.forward(task, img)

            plist = fmap[:]
            pmax = max(plist)
            labelText = ''
            unknown = False

            if pmax > threshold:
                max_index=plist.index(pmax)
                labelText = class_labels[max_index].strip()
            else:
                labelText = class_labels[len(class_labels)-1].strip()
                unknown = True

            acc = ('{:.1f}%'.format(pmax*100))
            
            fps = clk.fps()
            time_per_frame = 1000.0/fps

            lcd.display(img)
            lcd.draw_string(10, 10, labelText)
            if unknown==False: lcd.draw_string(10, 30, acc)
            lcd.draw_string(10, 60, (str(int(fps)) + ' fps'))
            lcd.draw_string(10, 80, (str(round(time_per_frame, 1)) + ' ms'))
            
            
            # Read btn
            if btn_a.value() == 0:
                btn_trigger = True
                utime.sleep_ms(500)
            
            # Store image of button/ts was pressed
            if btn_trigger and img_logging:
                annotation = labelText + ' (' + acc + ')'
                log_name = get_filename(img_log_path)
                log_path = img_log_path + '/' + log_name
                save_image(img,log_path , annotation, 10, 10)
                btn_trigger = False


    elif op_mode==1:
        # Operation mode single shot
        icon_dir = '/sd/media/'

            
        # Read btn
        if btn_a.value() == 0:
            btn_trigger = True
            if pause: 
                pause = False
                first_cycle = True
            utime.sleep_ms(200)

        
        if not pause:
            # Take image
            img = sensor.snapshot()
            # Display image
            lcd.display(img)

        # Run classification task if button/screen is pressed
        if btn_trigger and not pause and not first_cycle:
            pause = True

            # Feed forward
            fmap = kpu.forward(task, img)

            plist = fmap[:]
            # plist structure
            #(1.009735e-11, 9.50724e-13, 6.103466e-16, 0.1090703, 3.985165e-10, 0.8909298)

            pmax = max(plist)
            max_index=plist.index(pmax)

            trash_type = 0
            '''
            Types:
            0 -> Restmuell
            1 -> Papier
            2 -> Gelber Sack
            3 -> Glascontainer
            '''
            is_bottle = False


            if pmax > threshold:
                # Max confidence is over threshold
                # Select correct trash type
                if max_index==0 or max_index==3:
                    # Papier
                    trash_type = 1
                elif max_index==2 or max_index==4 or max_index==5:
                    # Gelber Sack
                    trash_type = 2

                    if max_index==4:
                        is_bottle=True

                elif max_index==1:
                    # Glas
                    trash_type = 3
                    is_bottle = True

                # Display result
                lcd.clear()

                # Display corresponding icon
                img_icon.clear()
                img_icon = image.Image(icon_dir + str(trash_type) + '.jpg')
                img_icon_tmp.draw_image(img_icon,65,5,0.5,0.5)
                lcd.display(img_icon_tmp)
                if is_bottle: lcd.draw_string(10, 10, 'Pfandflasche?')

                # Display strings (-> debug)
                #acc = ('{:.1f}%'.format(pmax*100))
                #lcd.draw_string(10, 10, str(trash_type) + ' , Flasche: ' + str(is_bottle))
                #lcd.draw_string(10, 30, acc)

                
            else:
                # Max confidence is below threshold
                trash_type = 0

                # Display result
                lcd.clear()

                # Display corresponding icon
                img_icon.clear()
                img_icon = image.Image(icon_dir + str(trash_type) + '.jpg')
                img_icon_tmp.draw_image(img_icon,65,5,0.5,0.5)
                lcd.display(img_icon_tmp)
                if is_bottle: lcd.draw_string(10, 10, 'Pfandflasche?')

                # Display string (-> debug)
                #lcd.draw_string(10, 10, 'Restmuell')
                

            # Store image of button/ts was pressed
            if img_logging:
                annotation = 'Class: ' + str(max_index) + ' (' + ('{:.1f}%'.format(pmax*100)) + ')'
                log_name = get_filename(img_log_path)
                log_path = img_log_path + '/' + log_name
                save_image(img,log_path , annotation, 10, 10)

        first_cycle=False

    elif op_mode==2:
        # Operation mode multi shot
        icon_dir = '/sd/media/'

            
        # Read btn
        if btn_a.value() == 0:
            btn_trigger = True
            if pause: 
                pause = False
                first_cycle = True
            utime.sleep_ms(200)

        
        if not pause:
            # Take image
            img = sensor.snapshot()
            # Display image
            lcd.display(img)

        # Run classification task if button/screen is pressed
        if btn_trigger and not pause and not first_cycle:
            pause = True

            lcd.draw_string(10, 10, '...')

            plist = [0.0]*6

            # Take the number of multishot pictures
            for i in range(img_num_mshot):
                img = sensor.snapshot()
                lcd.display(img)

                # Feed forward
                fmap = kpu.forward(task, img)

                plist_tmp = fmap[:]

                # add values to predicition vector element wise
                # Example:
                #list1=[1, 2, 3]
                #list2=[4, 5, 6]
                #result_list=[5,7,9]
                i=0
                for val in plist_tmp:
                    plist[i] = plist[i]+val
                    i=i+1


            # Divide through number of images taken (-> mean) element wise
            # Example:
            #result_list=[5,7,9] / img_num_mshot
            #result_list=[0.5, 0.7, 0.9]
            i=0
            for val in plist:
                plist[i] = val/img_num_mshot
                i=i+1

            # plist structure
            #(1.009735e-11, 9.50724e-13, 6.103466e-16, 0.1090703, 3.985165e-10, 0.8909298)

            pmax = max(plist)
            max_index=plist.index(pmax)

            trash_type = 0
            '''
            Types:
            0 -> Restmuell
            1 -> Papier
            2 -> Gelber Sack
            3 -> Glascontainer
            '''
            is_bottle = False


            if pmax > threshold:
                # Max confidence is over threshold
                # Select correct trash type
                if max_index==0 or max_index==3:
                    # Papier
                    trash_type = 1
                elif max_index==2 or max_index==4 or max_index==5:
                    # Gelber Sack
                    trash_type = 2

                    if max_index==4:
                        is_bottle=True

                elif max_index==1:
                    # Glas
                    trash_type = 3
                    is_bottle = True

                # Display result
                lcd.clear()

                # Display corresponding icon
                img_icon.clear()
                img_icon = image.Image(icon_dir + str(trash_type) + '.jpg')
                img_icon_tmp.draw_image(img_icon,65,5,0.5,0.5)
                lcd.display(img_icon_tmp)
                if is_bottle: lcd.draw_string(10, 10, 'Pfandflasche?')

                # Display strings (-> debug)
                #acc = ('{:.1f}%'.format(pmax*100))
                #lcd.draw_string(10, 10, str(trash_type) + ' , Flasche: ' + str(is_bottle))
                #lcd.draw_string(10, 30, acc)

                
            else:
                # Max confidence is below threshold
                trash_type = 0

                # Display result
                lcd.clear()

                # Display corresponding icon
                img_icon.clear()
                img_icon = image.Image(icon_dir + str(trash_type) + '.jpg')
                img_icon_tmp.draw_image(img_icon,65,5,0.5,0.5)
                lcd.display(img_icon_tmp)
                if is_bottle: lcd.draw_string(10, 10, 'Pfandflasche?')

                # Display string (-> debug)
                #lcd.draw_string(10, 10, 'Restmuell')


        first_cycle=False


a = kpu.deinit(task)




"""
Obtain meta data for bait experiments with a 3m-arena by kaitlin
 ----TODO will be added to main as a function to get meta data of the experiments? maybe
"""

import os
import pickle
import cv2
import numpy as np
import pyautogui as pag
from keyboard import press

os.chdir("C:/Users/nobua/Dropbox/research/with_dropbox/papers_and_projects/2022/TManual/examples/kaitlin_large_arena")
# 1. 
def get_metadata():
    out_dir = "tmanual"
	# Data read
    if os.path.exists(out_dir + "/res.pickle"):
        with open(out_dir+ os.sep  + 'res.pickle', mode='rb') as f:
            tmanual_output = pickle.load(f)
    else:
        print("error")
        return 0

    unique_id = list(set(tmanual_output[0]))

    ii = 0
    while ii < len(unique_id):
        # region --- 0. Load data ---#
        indice_unique_id = [i for i, x in enumerate(tmanual_output[0]) if x == unique_id[0]]
        serial_list = [tmanual_output[1][i] for i in indice_unique_id]
        print(serial_list)
        index_base_image = indice_unique_id[serial_list.index(min(serial_list))]
        unique_id = tmanual_output[2][index_base_image][1]
        image_name = tmanual_output[2][index_base_image][0]
        img_read = cv2.imread(image_name)
        
        if img_read is None:
            print("Error. file is not readable: " + image_name + ". Skip.")
            ii = ii + 1
            continue

        img_shape = np.array([img_read.shape[1], img_read.shape[0]])

        # create window
        window_name = "window"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        scr_w, scr_h = pag.size()
        if scr_h > scr_w * img_shape[1] / img_shape[0]:
            cv2.resizeWindow(window_name, scr_w, int(scr_w * img_shape[1] / img_shape[0]))
        else:
            cv2.resizeWindow(window_name, int(scr_h * img_shape[0] / img_shape[1]), scr_h)
        # endregion ------
        
        # region --- 1. Arena size ---#
        font_size = 1
        img = cv2.putText(img_read.copy(), '1.Arena size '+'('+unique_id+')',
                          (100, 50), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), font_size, cv2.LINE_AA)
        cv2.imshow(window_name, img)

        def arena_size(event, x, y, flags, param):
            nonlocal x0, y0, x1, y1, drawing, end
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                x0, y0 = x, y
            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    cv2.rectangle(img_copy, (x0, y0), (x, y), (0, 0, 255), thickness = 1)
            elif event == cv2.EVENT_LBUTTONUP:
                x1, y1 = x, y
                cv2.rectangle(img_copy, (x0, y0), (x1, y1), (0, 0, 255), thickness = 1)
                drawing = False
            elif event == cv2.EVENT_RBUTTONDOWN:
                if x0 > x1:
                    x0, x1 = x1, x0
                if y0 > y1:
                    y0, y1 = y1, y0
                end = 1
            press('enter')

        x0, y0, x1, y1 = 0, 0, 0, 0
        img_copy = img.copy()
        end = 0
        drawing = False
        while True:
            cv2.imshow('window', img_copy)
            if drawing:
                img_copy = img.copy()
            cv2.putText(img_copy, 'L DOWN -> L UP. R click to finish', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 1, cv2.LINE_AA)
            cv2.setMouseCallback('window', arena_size)
            k = cv2.waitKey(0)
            if end == 1:
                break
            if k == 27:
                break

        if k == 27:
            break

        # region --- 2.  Define initial point --- #
        img = cv2.putText(img_read.copy(), '2. Initial point ' + '(' + unique_id + ')',
                          (100, 50), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), font_size, cv2.LINE_AA)
        cv2.imshow('window', img)

        def get00(event, x, y, flags, params):
            img, img_data = params
            if event == cv2.EVENT_LBUTTONDOWN:
                img_data.change_ref(np.array([x, y]))
                press("enter")
            elif event == cv2.EVENT_RBUTTONDOWN:
                press("enter")

        cv2.setMouseCallback('window', get00, [img, img_data])
        cv2.waitKey()

        # endregion

        # region --- 3. Bait locate ---#
        img = cv2.putText(img_read.copy(), '3. Bait ' + '(' + unique_id + ')',
                          (100, 50), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), font_size, cv2.LINE_AA)
        cv2.imshow(window_name, img)

        def bait_draw(event, x, y, flags, param):
            nonlocal x0, y0, diameter, drawing, end
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                x0, y0 = x, y
            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    cv2.circle(img_copy, center=(int((x - x0) / 2) + x0, int((y - y0) / 2) + sy0),
                               radius=int(math.sqrt((x0 - x) ** 2 + (y0 - y) ** 2) / 2),
                               color=(0, 0, 255), thickness=1)
            elif event == cv2.EVENT_LBUTTONUP:
                diameter = math.sqrt((x0 - x) ** 2 + (y0 - y) ** 2)
                cv2.circle(img_copy, center=(int((x - x0) / 2) + x0, int((y - y0) / 2) + y0),
                           radius=int(diameter / 2),
                           color=(0, 0, 255), thickness=1)
                x0 = ((x - x0) / 2) + x0
                y0 = ((y - y0) / 2) + y0
                drawing = False
            elif event == cv2.EVENT_RBUTTONDOWN:
                end = 1
            press('enter')

        x0, y0, diameter = 0, 0, 0
        img_copy = img.copy()
        end = 0
        drawing = False
        while True:
            cv2.imshow('window', img_copy)
            if drawing:
                img_copy = img.copy()
            cv2.putText(img_copy, 'L DOWN -> L UP. R click to finish', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 1, cv2.LINE_AA)
            cv2.setMouseCallback('window', bait_draw)
            k = cv2.waitKey(0)
            if end == 1:
                break
            if k == 27:
                break

        if k == 27:
            break

        cv2.destroyAllWindows()

get_metadata()


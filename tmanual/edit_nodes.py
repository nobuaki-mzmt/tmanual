"""
GUI to edit the locations of nodes after measurement
can be used to adjust manual errors, especially when find inconsistency during post-analysis
"""


import os
import pickle
import cv2
import copy
import numpy as np
import glob
import re
import math
from keyboard import press
import pyautogui as pag

from tmanual.image import tunnel_draw, outlined_text, object_drawing, image_format, ImgData
from tmanual.measurement import zoom_func, output_measurement

vcol = [[37, 231, 253], [98, 201, 94], [140, 145, 33],  [139, 82, 59], [84, 1, 68]]  # viridis colors in BGR

def edit_nodes(in_dir, in_files, out_dir, file_extension, object_size, font_size, text_drawing):
    # Data read
    if os.path.exists(out_dir + "/res.pickle"):
        print("existing analysis loaded")
        with open(out_dir + '/res.pickle', mode='rb') as f:
            tmanual_output = pickle.load(f)
    else:
        print("no res.picke file found")
        return "error. no res.picke file found"

    if in_files == 0:
        name1 = glob.glob(in_dir + r'\*.' + file_extension)
    else:
        name1 = in_files.split(';')
    num_file = len(name1)

    # Analysis
    ii = 0
    while ii < num_file:
        # region ----- 0. load image (or skip) -----#
        # meta data
        i = name1[ii]
        img_data = ImgData(i)
        print(str(ii) + ": " + img_data.name)

        cur_data_index = list(set([i for i, x in enumerate(tmanual_output[1]) if x == img_data.serial]) & set(
            [i for i, x in enumerate(tmanual_output[0]) if x == img_data.id]))
        if len(cur_data_index) > 0:
            cur_data = copy.deepcopy(tmanual_output[2][cur_data_index[0]])
            img_data.name = cur_data[0]
            img_data.id = cur_data[1]
            img_data.serial = cur_data[2]
            img_data.ref_xy = cur_data[3]
            img_data.tunnel = cur_data[4]
            img_data.node = cur_data[5]
            img_data.scale_xy = cur_data[6]
            img_data.analyze_flag = cur_data[7]
        else:
            print("no image file found")
            return "error. no res.picke file found"

        # endregion ------

        img_read = cv2.imread(i)
        img_read = image_format(img_read)
        img_shape = np.array([img_read.shape[1], img_read.shape[0]])

        window_name = "window"

        # region ----- 1. Check if analyze the video -----#
        img = img_data.note_plot(img_read.copy(), '1.Edit nodes? ', font_size)
        img = img_data.object_plot(img, 0, vcol[4], object_size, font_size, draw_number=True)
        
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        scr_w, scr_h = pag.size()
        if scr_h > scr_w * img_shape[1] / img_shape[0]:
            cv2.resizeWindow(window_name, scr_w, int(scr_w * img_shape[1] / img_shape[0]))
        else:
            cv2.resizeWindow(window_name, int(scr_h * img_shape[0] / img_shape[1]), scr_h)

        cv2.imshow(window_name, img)

        def want_to_analyze(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                press('v')
            elif event == cv2.EVENT_RBUTTONDOWN:
                press('n')

        cv2.setMouseCallback(window_name, want_to_analyze)
        k = cv2.waitKey()

        if k == ord("b"):
            if ii > 0:
                ii = ii - 1
            continue
        if k == ord("n"):
            print("do not edit. next")
            ii = ii + 1
            continue
        if k == 27:
            cv2.destroyAllWindows()
            break
        # endregion ----------

        # region----- 2. Edit nodes -----#
        img = img_data.note_plot(img_read.copy(), '2.Nodes  ', font_size)
        img = img_data.object_plot(img, 0, vcol[4], object_size, font_size, draw_number=False)
        zoom, zoom_xy = [1, 1, 1], [np.array([0, 0]), np.array([0, 0]), np.array([0, 0])]  # for x2, x4, x8
        count = 0
        img_undo = img.copy()
        mouse_xy, node_xy = np.array([0, 0]), np.array([0, 0])

        while count < len(img_data.node):
            cv2.circle(img, ((img_data.node[count]*zoom[0]-zoom_xy[0])*zoom[1]-zoom_xy[1])*zoom[2]-zoom_xy[2], object_size*zoom[0]*zoom[1]*zoom[2], vcol[1], -1)
            cv2.imshow('window', img)

            def node_identify(event, x, y, flags, param):
                nonlocal img, node_xy, mouse_xy, zoom, zoom_xy
                if event == cv2.EVENT_MOUSEMOVE:
                    mouse_xy = np.array([x, y])
                if event == cv2.EVENT_LBUTTONDOWN:
                    node_xy = (((np.array([x, y])+zoom_xy[2])/zoom[2]+zoom_xy[1])/zoom[1]+zoom_xy[0])/zoom[0]
                    node_xy = node_xy.astype(int)
                    press('p')
                elif event == cv2.EVENT_RBUTTONDOWN:
                    press('n')

            cv2.setMouseCallback('window', node_identify)
            k = cv2.waitKey(0)
            if k == ord("p"):
                img_data.node[count] = node_xy
                break
            elif k == ord("n"):
                count = count + 1
            elif k == ord("r"):
                img_data.node.pop(count)
                node_xy = []
                break
            elif k == ord("q"):
                if count > 0:
                    count = count - 1
            elif k == ord("z"):
                if zoom[1] == 2 and zoom[2] == 1:
                    img, zoom_xy[2], zoom[2] = zoom_func(img, mouse_xy, img_shape, zoom[2])
                elif zoom[0] == 2 and zoom[1] == 1:
                    img, zoom_xy[1], zoom[1] = zoom_func(img, mouse_xy, img_shape, zoom[1])
                elif zoom[0] == 1:
                    img, zoom_xy[0], zoom[0] = zoom_func(img, mouse_xy, img_shape, zoom[0])
            if k == ord("x") or k == ord("q"):
                img = img_undo.copy()
                img = object_drawing(img, None, None, None, img_data.node[0:count], 0, vcol[4], object_size, font_size, draw_number=False)
                zoom, zoom_xy = [1, 1, 1], [np.array([0, 0]), np.array([0, 0]), np.array([0, 0])]  # for x2, x4, x8
        # endregion

        # region----- 3. Output -----#
        tmanual_output = output_measurement(img_data, img_read.copy(), tmanual_output, out_dir, object_size, font_size, text_drawing)

        # correct all subsequent images in the same experimental setup
        cur_name = copy.copy(img_data.name)
        subsequent_data_index = list(set([i for i, x in enumerate(tmanual_output[1]) if x > img_data.serial]) & set(
            [i for i, x in enumerate(tmanual_output[0]) if x == img_data.id]))
        if len(subsequent_data_index) > 0:
            for i_dindex in range(len(subsequent_data_index)):
                subsequent_data = copy.deepcopy(tmanual_output[2][subsequent_data_index[0]])
                img_data.name = subsequent_data[0]
                img_data.id = subsequent_data[1]
                img_data.serial = subsequent_data[2]
                img_data.ref_xy = subsequent_data[3]
                img_data.tunnel = subsequent_data[4]
                img_data.node = subsequent_data[5]
                img_data.scale_xy = subsequent_data[6]
                img_data.analyze_flag = subsequent_data[7]

                if len(node_xy) > 0:
                    img_data.node[count] = node_xy
                else:
                    img_data.node.pop(count)
                img_read = cv2.imread(i.replace(cur_name, img_data.name))
                img_read = image_format(img_read)
                tmanual_output = output_measurement(img_data, img_read, tmanual_output, out_dir, object_size, font_size, text_drawing)


        else:
            print("no image file found")
            return "error. no res.picke file found"

        ii = ii + 1
        # endregion

    cv2.destroyAllWindows()
    print("Finished. Next to Post-analysis.")

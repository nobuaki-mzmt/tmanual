"""
GUI to measure tunnel length 
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
v_col = [[37, 231, 253], [98, 201, 94], [140, 145, 33],  [139, 82, 59], [84, 1, 68]]  # viridis colors in BGR


def zoom_func(img_z, mouse_xy, img_shape, zoom):
    mouse_xy[0] = max(mouse_xy[0], img_shape[0] / 4)
    mouse_xy[1] = max(mouse_xy[1], img_shape[1] / 4)
    mouse_xy[0] = min(mouse_xy[0], img_shape[0] * 3 / 4)
    mouse_xy[1] = min(mouse_xy[1], img_shape[1] * 3 / 4)
    img_zoom = cv2.resize(img_z, dsize=(img_shape * 2))
    img_zoom = img_zoom[int(mouse_xy[1] * 2 - img_shape[1] / 2):int(mouse_xy[1] * 2 + img_shape[1] / 2),
               int(mouse_xy[0] * 2 - img_shape[0] / 2):int(mouse_xy[0] * 2 + img_shape[0] / 2)]
    zoom_xy = mouse_xy * 2 - img_shape / 2
    zoom_xy = zoom_xy.astype(int)
    return img_zoom, zoom_xy, zoom*2


def output_measurement(img_data, img, tmanual_output, out_dir, object_size, font_size, text_drawing):
    img_data.analyze_done()

    # delete old data
    duplicate_data_index = list(
        set([i for i, x in enumerate(tmanual_output[1]) if x == img_data.serial]) & 
        set([i for i, x in enumerate(tmanual_output[0]) if x == img_data.id])
    )
    if len(duplicate_data_index) > 0:
        print("delete duplicate data")
        tmanual_output[0].pop(duplicate_data_index[0])
        tmanual_output[1].pop(duplicate_data_index[0])
        tmanual_output[2].pop(duplicate_data_index[0])

    # add new data
    tmanual_output[0].append(img_data.id)
    tmanual_output[1].append(img_data.serial)
    tmanual_output[2].append(img_data.output_image_data())
    img_data.image_output(img, out_dir, object_size, font_size, text_drawing)

    # write
    with open(out_dir + os.sep + 'res.pickle', mode='wb') as f:
        pickle.dump(tmanual_output, f)
    return tmanual_output


def measurement(in_dir, in_files, out_dir, skip_analyzed, file_extension, object_size, font_size, text_drawing):
    # Data read
    if os.path.exists(out_dir + "/res.pickle"):
        print("existing analysis loaded")
        with open(out_dir+ os.sep  + 'res.pickle', mode='rb') as f:
            tmanual_output = pickle.load(f)

        # --- todo this part will be removed future
        # remove node object from old version tmanual res.pickle
        if len(tmanual_output[2][0]) > 7:
            for ii in range(len(tmanual_output[0])):
                tmanual_output[2][ii].pop(5)
            with open(out_dir + os.sep + 'res.pickle', mode='wb') as f:
                pickle.dump(tmanual_output, f)
        # ----------

    else:
        print("new analysis start")
        tmanual_output = [[], [], []]  # store Ids, Serial, Results

    if in_files == 0:
        name1 = glob.glob(in_dir + os.sep + '*.' + file_extension)
    else:
        name1 = in_files.split(';')
    num_file = len(name1)

    # Analysis
    ii = 0
    while ii < num_file:

        # region --- Load image (or skip) ---#
        i = name1[ii]
        img_name = re.sub("."+file_extension, "", os.path.basename(i))
        try:
            int(img_name.split('_')[1])
        except:
            return("Error. Invalid filename: " + os.path.basename(i))
        img_data = ImgData(os.path.basename(i), None, file_extension)

        print(str(ii) + ": " + img_data.name)

        cur_data, pre_data = [], []
        cur_data_index = list(
            set([i for i, x in enumerate(tmanual_output[1]) if x == img_data.serial]) &
            set([i for i, x in enumerate(tmanual_output[0]) if x == img_data.id])
        )
        pre_data_index = list(
            set([i for i, x in enumerate(tmanual_output[1]) if x < img_data.serial]) &
            set([i for i, x in enumerate(tmanual_output[0]) if x == img_data.id])
        )

        if len(cur_data_index) > 0:
            cur_data = copy.deepcopy(tmanual_output[2][cur_data_index[0]])
            img_data = ImgData(None, cur_data)
        if len(pre_data_index) > 0:
            close_pre_data_index = pre_data_index[0]
            for p_ii in pre_data_index:
                if tmanual_output[1][close_pre_data_index] < tmanual_output[1][p_ii]:
                    close_pre_data_index = p_ii
            pre_data = copy.deepcopy(tmanual_output[2][close_pre_data_index])

        # skip analyzed video
        if img_data.analyze_flag > 0:
            if skip_analyzed == "true":
                ii = ii + 1
                continue

        img_read = cv2.imread(i)
        if img_read is None:
            print("Error. file is not readable: " + os.path.basename(i) + ". Skip.")
            ii = ii + 1
            continue
        #img_read = image_format(img_read)
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

        # region --- 1. Check if analyze the video ---#
        img = img_data.note_plot(img_read.copy(), '1.Check ', font_size)

        # if data of current image exist, draw object
        if img_data.analyze_flag > 0:
            img = img_data.object_plot(img, 0, v_col[4], object_size, font_size, draw_number=True)
        # else if data of previous image exist, draw object
        elif len(pre_data) > 0:
            img_data.ref_xy = pre_data[3]
            img_data.tunnel = pre_data[4]
            img_data.scale_xy = pre_data[5]
            img = img_data.object_plot(img, 0, v_col[4], object_size, font_size, draw_number=True)

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
            skip_analyzed = False
            continue
        if k == ord("n"):
            print("do not analyze. next")
            if img_data.analyze_flag == 0:
                output_measurement(img_data, img_read.copy(), tmanual_output, out_dir, object_size, font_size, text_drawing)
            ii = ii + 1
            continue
        if k == ord("r"):
            # reanalyze, appending to the data of previous image
            if len(pre_data) > 0:
                img_data.ref_xy = pre_data[3]
                img_data.tunnel = pre_data[4]
                img_data.scale_xy = pre_data[5]
            else:
                img_data.tunnel = []
        if k == ord("a"):
            # reanalyze, from scratch
            img_data.tunnel = []
        if k == 27:
            cv2.destroyAllWindows()
            break
        # endregion ----------

        # region --- 2.  Define Ref point --- #
        img = img_data.note_plot(img_read.copy(), '2.Ref point  ', font_size)
        cv2.circle(img, img_data.ref_xy, object_size * 5, v_col[0], object_size)
        cv2.circle(img, img_data.ref_xy, object_size, (0, 0, 0), -1)

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

        # region --- 3.  Measure tunnel length --- #
        img = img_data.note_plot(img_read.copy(), '3.Measure  ', font_size)

        # draw previous tunnels
        num_old_tunnel = len(img_data.tunnel)
        img = img_data.object_plot(img, 0, v_col[2], object_size, font_size, draw_number=False, end_node_draw=False)
        img_undo = img.copy()
        tunnel_pre = img_data.tunnel

        # todo: code for zooming is not great. but I have no idea how to improve yet.
        count, end, mouse_xy = 0, 0, np.array([0, 0])
        zoom, zoom_xy = [1, 1, 1], [np.array([0, 0]), np.array([0, 0]), np.array([0, 0])]  # for x2, x4, x8
        img_data.tunnel = []

        while True:
            cv2.imshow('window', img)
            current_tunnel = np.empty((0, 2), int)
            if num_old_tunnel > 0 and count < num_old_tunnel:
                current_tunnel = copy.copy(tunnel_pre[count])

            def tunnel_length(event, x, y, flags, param):
                nonlocal img, current_tunnel, end, mouse_xy, zoom, zoom_xy

                img = tunnel_draw(img,
                                  ((current_tunnel*zoom[0]-zoom_xy[0])*zoom[1]-zoom_xy[1])*zoom[2]-zoom_xy[2],
                                  v_col[1], object_size*zoom[0]*zoom[1]*zoom[2], False)

                if event == cv2.EVENT_MOUSEMOVE:
                    mouse_xy = np.array([x, y])

                if event == cv2.EVENT_LBUTTONDOWN:
                    current_tunnel = np.append(current_tunnel, (((np.array([[x, y]])+zoom_xy[2])/zoom[2]+zoom_xy[1])/zoom[1]+zoom_xy[0])/zoom[0], axis=0)
                    current_tunnel = current_tunnel.astype(int)

                if event == cv2.EVENT_RBUTTONDOWN:
                    if len(current_tunnel) > 0:
                        img = object_drawing(img, None, None, [((current_tunnel*zoom[0]-zoom_xy[0])*zoom[1]-zoom_xy[1])*zoom[2]-zoom_xy[2]],
                                             count, v_col[4], object_size*zoom[0]*zoom[1]*zoom[2], font_size*zoom[0]*zoom[1]*zoom[2], draw_number=False)
                        press('p')
                    else:
                        press('f')

                cv2.imshow('window', img)

            cv2.setMouseCallback('window', tunnel_length)
            k = cv2.waitKey(0)
            if k == ord("p"):
                count = count + 1
                img_data.tunnel.append(current_tunnel)
            elif k == ord("f"):
                break
            elif k == ord("z"):
                if zoom[1] == 2 and zoom[2] == 1:
                    img, zoom_xy[2], zoom[2] = zoom_func(img, mouse_xy, img_shape, zoom[2])
                elif zoom[0] == 2 and zoom[1] == 1:
                    img, zoom_xy[1], zoom[1] = zoom_func(img, mouse_xy, img_shape, zoom[1])
                elif zoom[0] == 1:
                    img, zoom_xy[0], zoom[0] = zoom_func(img, mouse_xy, img_shape, zoom[0])
                
            elif k == ord("q"):
                if count > 0:
                    img_data.tunnel.pop(-1)
                    count = count - 1

            if k == ord("x") or k == ord("q"):
                # cancel zoom when redo
                img = img_undo.copy()
                img = object_drawing(img, img_data.ref_xy, None, img_data.tunnel[0:count], 
                                     0, v_col[4], object_size, font_size, draw_number=False)
                zoom, zoom_xy = [1, 1, 1], [np.array([0, 0]), np.array([0, 0]), np.array([0, 0])]

            if k == ord("e"):
                if num_old_tunnel > 0 and count < num_old_tunnel:
                    count_temp = count
                    for i_count in range(count_temp, num_old_tunnel):
                        current_tunnel = copy.copy(tunnel_pre[i_count])
                        img = object_drawing(img, None, None, [((current_tunnel*zoom[0]-zoom_xy[0])*zoom[1]-zoom_xy[1])*zoom[2]-zoom_xy[2]],
                                             count, v_col[4], object_size*zoom[0]*zoom[1]*zoom[2], font_size*zoom[0]*zoom[1]*zoom[2], draw_number=False)
                        img_data.tunnel.append(current_tunnel)
                        count = count + 1

            if k == 27:
                break
        if k == 27:
            break

        # endregion

        # region --- 4.  Scaling --- #
        img = img_data.note_plot(img_read.copy(), '4.Scale  ', font_size)
        cv2.line(img, img_data.scale_xy[0], img_data.scale_xy[1], (0, 255, 0), object_size)
        end, drawing = 0, False

        def scale_length(event, x, y, flags, param):
            nonlocal drawing, end, img_copy
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                img_data.scale_xy[0] = np.array([x, y])
            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    img_copy = img.copy()
                    cv2.line(img_copy, img_data.scale_xy[0], (x, y), (0, 0, 255), 2)
            elif event == cv2.EVENT_LBUTTONUP:
                img_data.scale_xy[1] = np.array([x, y])
                cv2.line(img_copy, img_data.scale_xy[0], img_data.scale_xy[1], (0, 0, 255), 2)
                drawing = False
            elif event == cv2.EVENT_RBUTTONDOWN:
                end = 1
            cv2.imshow('window', img_copy)

        img_copy = img.copy()
        cv2.imshow('window', img_copy)
        while True:
            cv2.setMouseCallback('window', scale_length)
            if cv2.waitKey(1) & end == 1:
                break
        # endregion

        # region----- Output -----#
        tmanual_output = output_measurement(img_data, img_read.copy(), tmanual_output, out_dir, object_size, font_size, text_drawing)
        ii = ii + 1
        # endregion

    cv2.destroyAllWindows()
    print("Finished. Next to Post-analysis.")

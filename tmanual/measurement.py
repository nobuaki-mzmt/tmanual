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
vcol = [[37, 231, 253], [98, 201, 94], [140, 145, 33],  [139, 82, 59], [84, 1, 68]]  # viridis colors in BGR

def zoom_func(img_z, mouse_xy, img_shape):
    mouse_xy[0] = max(mouse_xy[0], img_shape[0] / 4)
    mouse_xy[1] = max(mouse_xy[1], img_shape[1] / 4)
    mouse_xy[0] = min(mouse_xy[0], img_shape[0] * 3 / 4)
    mouse_xy[1] = min(mouse_xy[1], img_shape[1] * 3 / 4)
    img_zoom = cv2.resize(img_z, dsize=(img_shape * 2))
    img_zoom = img_zoom[int(mouse_xy[1] * 2 - img_shape[1] / 2):int(mouse_xy[1] * 2 + img_shape[1] / 2),
               int(mouse_xy[0] * 2 - img_shape[0] / 2):int(mouse_xy[0] * 2 + img_shape[0] / 2)]
    zoom_xy = mouse_xy * 2 - img_shape / 2
    zoom_xy = zoom_xy.astype(int)
    return img_zoom, zoom_xy, 2


def output_measurement(img_data, img, tmanual_output, out_dir, object_size, font_size):
    img_data.analyze_done()

    # delete old data
    duplicate_data_index = list(set([i for i, x in enumerate(tmanual_output[1]) if x == img_data.serial]) & set(
            [i for i, x in enumerate(tmanual_output[0]) if x == img_data.id]))
    if len(duplicate_data_index) > 0:
        print("delete duplicate data")
        tmanual_output[0].pop(duplicate_data_index[0])
        tmanual_output[1].pop(duplicate_data_index[0])
        tmanual_output[2].pop(duplicate_data_index[0])

    # add new data
    tmanual_output[0].append(img_data.id)
    tmanual_output[1].append(img_data.serial)
    tmanual_output[2].append(img_data.output_image_data())
    img_data.image_output(img, out_dir, object_size, font_size)

    # write
    with open(out_dir + '/res.pickle', mode='wb') as f:
        pickle.dump(tmanual_output, f)

    return tmanual_output


def measurement(in_dir, in_files, out_dir, skip_analyzed, file_extension, object_size, font_size):
    # Data read
    if os.path.exists(out_dir + "/res.pickle"):
        print("existing analysis loaded")
        with open(out_dir + '/res.pickle', mode='rb') as f:
            tmanual_output = pickle.load(f)
    else:
        print("new analysis start")
        tmanual_output = [[], [], []]  # store Ids, Serial, Results

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

        cur_data, pre_data = [], []
        cur_data_index = list(set([i for i, x in enumerate(tmanual_output[1]) if x == img_data.serial]) & set(
            [i for i, x in enumerate(tmanual_output[0]) if x == img_data.id]))
        pre_data_index = list(set([i for i, x in enumerate(tmanual_output[1]) if x == img_data.serial-1]) & set(
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
        if len(pre_data_index) > 0:
            pre_data = copy.deepcopy(tmanual_output[2][pre_data_index[0]])

        # skip analyzed video
        if img_data.analyze_flag > 0:
            if skip_analyzed == "true":
                ii = ii + 1
                continue

        # endregion ------

        img_read = cv2.imread(i)
        img_read = image_format(img_read)
        img_shape = np.array([img_read.shape[1], img_read.shape[0]])

        window_name = "window"

        # region ----- 1. Check if analyze the video -----#
        img = img_data.note_plot(img_read.copy(), '1.Analyze? ', font_size)

        # if data of current image exist, draw object
        if img_data.analyze_flag > 0:
            img = img_data.object_plot(img, 0, vcol[4], object_size, font_size, draw_number=True)
        # else if data of previous image exist, draw object
        elif len(pre_data) > 0:
            img_data.ref_xy = pre_data[3]
            img_data.tunnel = pre_data[4]
            img_data.node = pre_data[5]
            img_data.scale_xy = pre_data[6]
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
            skip_analyzed = False
            continue
        if k == ord("n"):
            print("do not analyze. next")
            if img_data.analyze_flag == 0:
                output_measurement(img_data, img_read.copy(), tmanual_output, out_dir, object_size, font_size)
            ii = ii + 1
            continue
        if k == ord("r"):
            # if reanalyze it, want to append to the data of previous image
            if len(pre_data) > 0:
                img_data.ref_xy = pre_data[3]
                img_data.tunnel = pre_data[4]
                img_data.node = pre_data[5]
                img_data.scale_xy = pre_data[6]
            else:
                img_data.tunnel, img_data.node = [], []
        if k == ord("a"):
            img_data.tunnel, img_data.node = [], []
        if k == 27:
            cv2.destroyAllWindows()
            break
        # endregion ----------

        # region----- 2. Analyze the image -----#
        # region --- 2-1.  Define (0,0) --- #
        img = img_data.note_plot(img_read.copy(), '2.Ref point  ', font_size)
        cv2.circle(img, img_data.ref_xy, object_size * 5, vcol[0], object_size)
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

        # region --- 2-2.  Measure tunnel length --- #
        img = img_data.note_plot(img_read.copy(), '3.Tunnels  ', font_size)

        # draw previous tunnel
        num_old_tunnel = len(img_data.tunnel)
        img = img_data.object_plot(img, 0, vcol[2], object_size, font_size, draw_number=False)
        img_undo = img.copy()
        tunnel_pre = img_data.tunnel
        img_data.tunnel = []
        count, end, zoom, zoom_xy, mouse_xy, node_xy = 0, 0, 1, np.array([0, 0]), np.array([0, 0]), np.array([0, 0])

        while True:
            cv2.imshow('window', img)
            current_tunnel = np.empty((0, 2), int)
            if num_old_tunnel > 0 and count < num_old_tunnel:
                current_tunnel = copy.copy(tunnel_pre[count])

            def tunnel_length(event, x, y, flags, param):
                nonlocal img, current_tunnel, end, mouse_xy, zoom, zoom_xy

                img = tunnel_draw(img, current_tunnel * zoom - zoom_xy, vcol[1], object_size*zoom)

                if event == cv2.EVENT_MOUSEMOVE:
                    mouse_xy = np.array([x, y])

                if event == cv2.EVENT_LBUTTONDOWN:
                    current_tunnel = np.append(current_tunnel, (np.array([[x, y]]) + zoom_xy) / zoom, axis=0)
                    current_tunnel = current_tunnel.astype(int)

                if event == cv2.EVENT_RBUTTONDOWN:
                    if len(current_tunnel) > 0:
                        img = object_drawing(img, None, None, [current_tunnel * zoom - zoom_xy],
                                             None, count, vcol[4], object_size*zoom, font_size*zoom, draw_number=False)
                        press('p')
                    else:
                        press('e')

                cv2.imshow('window', img)

            cv2.setMouseCallback('window', tunnel_length)
            k = cv2.waitKey(0)
            if k == ord("p"):
                count = count + 1
                img_data.tunnel.append(current_tunnel)
            elif k == ord("e"):
                break
            elif k == ord("z"):
                if zoom == 1:
                    img, zoom_xy, zoom = zoom_func(img, mouse_xy, img_shape)
            elif k == ord("q"):
                if count > 0:
                    img_data.tunnel.pop(-1)
                    count = count - 1

            if k == ord("x") or k == ord("q"):
                # cancel zoom when redo
                img = img_undo.copy()
                img = object_drawing(img, img_data.ref_xy, None, img_data.tunnel[0:count], None,
                                     0, vcol[4], object_size, font_size, draw_number=False)
                zoom, zoom_xy = 1, np.array([0, 0])

            if k == 27:
                if num_old_tunnel > 0 and count < num_old_tunnel:
                    for i_count in range(count, num_old_tunnel):
                        current_tunnel = copy.copy(tunnel_pre[i_count])
                        img = object_drawing(img, None, None, [current_tunnel * zoom - zoom_xy],
                                            None, i_count, vcol[4], object_size*zoom, font_size*zoom, draw_number=False)
                        img_data.tunnel.append(current_tunnel)
                break

        # endregion

        # region --- 2-3.  Identify branch --- #
        img = img_data.note_plot(img_read.copy(), '4.Nodes  ', font_size)
        img = img_data.object_plot(img, 0, vcol[4], object_size, font_size, draw_number=False)
        zoom = 1
        count, count_0 = len(img_data.node), len(img_data.node)
        img_undo = img.copy()

        while True:
            cv2.imshow('window', img)

            def node_identify(event, x, y, flags, param):
                nonlocal img, node_xy, mouse_xy, zoom, zoom_xy
                if event == cv2.EVENT_MOUSEMOVE:
                    mouse_xy = np.array([x, y])
                if event == cv2.EVENT_LBUTTONDOWN:
                    node_xy = (np.array([x, y]) + zoom_xy) / zoom
                    node_xy = node_xy.astype(int)
                    press('p')
                elif event == cv2.EVENT_RBUTTONDOWN:
                    press('n')

            cv2.setMouseCallback('window', node_identify)
            k = cv2.waitKey(0)
            if k == ord("p"):
                img_data.node.append(node_xy)
                img = object_drawing(img, node_d=[node_xy*zoom-zoom_xy], offset=count, object_size=object_size, font_size=font_size, draw_number=False)

                count = count + 1
            if k == ord("n"):
                break
            elif k == ord("q"):
                if count > count_0:
                    img_data.node.pop(-1)
                    count = count - 1
            elif k == ord("z"):
                if zoom == 1:
                    img, zoom_xy, zoom = zoom_func(img, mouse_xy, img_shape)
            if k == ord("x") or k == ord("q"):
                img = img_undo.copy()
                img = object_drawing(img, None, None, None, img_data.node[0:count], 0, vcol[4], object_size, font_size, draw_number=False)
                zoom = 1
        # endregion

        # region --- 2-4.  Scaling --- #
        img = img_data.note_plot(img_read.copy(), '5.Scale  ', font_size)
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
        # endregion

        # region----- 3. Output -----#
        tmanual_output = output_measurement(img_data, img_read.copy(), tmanual_output, out_dir, object_size, font_size)
        ii = ii + 1
        # endregion

    cv2.destroyAllWindows()

    return "Finished. Next to Post-analysis."

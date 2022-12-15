"""
Standalone program, just combine all image.py, gui.py, measurement.py, and postanalysis.py
"""

import os
import pickle
import cv2
import copy
import numpy as np
import glob
import re
import csv
import math
from keyboard import press
import pyautogui as pag
from tqdm import tqdm
import PySimpleGUI as sg

v_col = [[37, 231, 253], [98, 201, 94], [140, 145, 33],  [139, 82, 59], [84, 1, 68]]  # viridis colors in BGR

note_pos = [40, 100]
note_pos2 = [40, 200]
note_pos3 = [40, 300]

def tunnel_draw(img_t, current_tunnel_t, col_t, object_size, end_node_draw=True):
    l_t = len(current_tunnel_t)
    if l_t > 0:
        for t_seg_iter in range(l_t-1):
            cv2.line(img_t, current_tunnel_t[t_seg_iter], current_tunnel_t[t_seg_iter+1], col_t, object_size)
        cv2.circle(img_t, current_tunnel_t[0], object_size, v_col[0], -1)
        if l_t > 1 and end_node_draw:
            cv2.circle(img_t, current_tunnel_t[l_t-1], object_size, v_col[0], round(object_size/2))
    return img_t


def outlined_text(img_o, text_o, ref_o, col_o, font_size):
    cv2.putText(img_o, text_o, ref_o, cv2.FONT_HERSHEY_PLAIN, font_size, (0, 0, 0), font_size*5, cv2.LINE_AA)
    cv2.putText(img_o, text_o, ref_o, cv2.FONT_HERSHEY_PLAIN, font_size, col_o, font_size, cv2.LINE_AA)
    return img_o


def object_drawing(img_d, ref_d=None, scale_d=None, tunnel_d=None, offset=0,
                   col_t=None, object_size = 5, font_size = 2, draw_number = True, end_node_draw=True):
    if ref_d is not None:
        cv2.circle(img_d, ref_d, object_size*5, v_col[0], 5)
        cv2.circle(img_d, ref_d, object_size, (0, 0, 0), -1)
    if col_t is None:
        col_t = [0, 0, 0]
    if scale_d is not None:
        cv2.line(img_d, scale_d[0], scale_d[1], (0, 0, 255), object_size)
    if tunnel_d is not None:
        for tt in range(len(tunnel_d)):
            img_d = tunnel_draw(img_d, tunnel_d[tt], col_t, object_size, end_node_draw)
            if draw_number:
                img_d = outlined_text(img_d, str(tt+offset), tunnel_d[tt][0]-np.array([object_size, 0]), v_col[0], font_size)
    return img_d


def image_format(img):  # all images are reformatted in 2000xH for measurement
    h, w = img.shape[:2]
    if w < h:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img = cv2.resize(img, dsize=(2000, round(h*2000/w)))
    return img


class ImgData:
    def __init__(self, img_loc, data_values=None, file_extension=None):
        if data_values is None:
            self.name = os.path.basename(img_loc)
            self.id = self.name.split('_')[0]
            if len(self.name.split("_")) > 1:
                self.serial = int(re.sub("."+file_extension, "", self.name.split('_')[1]))
            else:
                self.serial = 0
            data_values = [self.name, self.id, self.serial, np.array([0, 0]), [], [[0, 0], [0, 0]], 0]
        self.name = data_values[0]
        self.id = data_values[1]
        self.serial = data_values[2]
        self.ref_xy = data_values[3]
        self.tunnel = data_values[4]
        self.scale_xy = data_values[5]
        self.analyze_flag = data_values[6]

    def output_image_data(self):
        return [self.name, self.id, self.serial, self.ref_xy, self.tunnel, self.scale_xy, self.analyze_flag]

    def note_plot(self, img, note_message, font_size):
        cv2.putText(img, note_message+'('+self.name+')',
                    note_pos, cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), font_size, cv2.LINE_AA)
        return img

    def object_plot(self, img, offset, col_t, object_size, font_size, draw_number, end_node_draw=True):
        img = object_drawing(img, self.ref_xy, self.scale_xy, self.tunnel, offset, col_t, object_size, font_size, draw_number, end_node_draw)
        return img

    def change_ref(self, ref_xy_new):
        self.scale_xy[0] = self.scale_xy[0] - self.ref_xy + ref_xy_new
        self.scale_xy[1] = self.scale_xy[1] - self.ref_xy + ref_xy_new
        for tt in range(len(self.tunnel)):
            self.tunnel[tt] = self.tunnel[tt] - self.ref_xy + ref_xy_new
        self.ref_xy = ref_xy_new

    def measure_tunnel_length(self, scale_object_length):
        self.scale_xy[0] = np.array(self.scale_xy[0])
        self.scale_xy[1] = np.array(self.scale_xy[1])
        scale = math.sqrt(sum((self.scale_xy[1]-self.scale_xy[0])**2)) 
        if len(self.tunnel) > 0 and scale == 0:
            scale = 1
            print("Caution! Length of scale object is 0. Use 1 instead. Otherwise, recheck the image:", self.name)
        tunnel_len = []
        for tt in range(len(self.tunnel)):
            tl = 0
            for ttt in range(len(self.tunnel[tt])-1):
                tl = tl+math.sqrt(sum((self.tunnel[tt][ttt+1]-self.tunnel[tt][ttt])**2))
            tunnel_len.append(tl/scale*scale_object_length)
        return tunnel_len, scale

    def obtain_nodes(self):
        start_node, end_node = [], []
        for tt in range(len(self.tunnel)):
            start_node.append(self.tunnel[tt][0])
            end_node.append(self.tunnel[tt][len(self.tunnel[tt])-1])
        return start_node, end_node

    def image_output(self, img, out_dir, object_size, font_size, text_drawing):
        cv2.putText(img, self.id+"_"+str(self.serial), note_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), font_size, cv2.LINE_AA)
        if len(self.tunnel) < 1:
            cv2.putText(img, "no tunnel", note_pos2,
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), font_size, cv2.LINE_AA)
        if text_drawing:
            img = object_drawing(img, self.ref_xy, self.scale_xy, self.tunnel, 0, v_col[4], object_size, font_size, draw_number=text_drawing)
            cv2.imwrite(out_dir+"/" + self.name, img)
        else:
            img = object_drawing(img, self.ref_xy, self.scale_xy, self.tunnel, 0, v_col[4], object_size, font_size, draw_number=text_drawing)
            cv2.imwrite(out_dir+"/wotext_" + self.name, img)
            img = object_drawing(img, self.ref_xy, self.scale_xy, self.tunnel, 0, v_col[4], object_size, font_size, draw_number=True)
            cv2.imwrite(out_dir+"/" + self.name, img)

    def colored_image_output(self, img, tunnel_sequence, out_dir, object_size, font_size, text_drawing):
        cv2.putText(img, self.id+"_"+str(self.serial), note_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), font_size, cv2.LINE_AA)
        if len(self.tunnel) < 1:
            cv2.putText(img, "no tunnel", note_pos2,
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), font_size, cv2.LINE_AA)
        for tt in range(len(self.tunnel)):
                img = tunnel_draw(img, self.tunnel[tt], v_col[5-tunnel_sequence[tt]], object_size)
        if not text_drawing:
            cv2.imwrite(out_dir+"colored_wotext_"+self.name, img)

        for tt in range(len(self.tunnel)):
            img = outlined_text(img, str(tt), self.tunnel[tt][0]-np.array([object_size, 0]), v_col[5-tunnel_sequence[tt]], font_size)
        cv2.imwrite(out_dir+"colored_"+self.name, img)

    def analyze_done(self):
        self.analyze_flag = 1


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
    with open(out_dir + '/res.pickle', mode='wb') as f:
        pickle.dump(tmanual_output, f)
    return tmanual_output


def measurement(in_dir, in_files, out_dir, skip_analyzed, file_extension, object_size, font_size, text_drawing):
    # Data read
    if os.path.exists(out_dir + "/res.pickle"):
        print("existing analysis loaded")
        with open(out_dir + '/res.pickle', mode='rb') as f:
            tmanual_output = pickle.load(f)

        # --- todo this part will be removed future
        # remove node object from old version tmanual res.pickle
        if len(tmanual_output[2][0]) > 7:
            for ii in range(len(tmanual_output[0])):
                tmanual_output[2][ii].pop(5)
            with open(out_dir + '/res.pickle', mode='wb') as f:
                pickle.dump(tmanual_output, f)
        # ----------

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

        # region --- Load image (or skip) ---#
        i = name1[ii]
        img_data = ImgData(i, None, file_extension)
        print(str(ii) + ": " + img_data.name)

        cur_data, pre_data = [], []
        cur_data_index = list(
            set([i for i, x in enumerate(tmanual_output[1]) if x == img_data.serial]) &
            set([i for i, x in enumerate(tmanual_output[0]) if x == img_data.id])
        )
        pre_data_index = list(
            set([i for i, x in enumerate(tmanual_output[1]) if x == img_data.serial-1]) &
            set([i for i, x in enumerate(tmanual_output[0]) if x == img_data.id])
        )

        if len(cur_data_index) > 0:
            cur_data = copy.deepcopy(tmanual_output[2][cur_data_index[0]])
            img_data = ImgData(None, cur_data)
        if len(pre_data_index) > 0:
            pre_data = copy.deepcopy(tmanual_output[2][pre_data_index[0]])

        # skip analyzed video
        if img_data.analyze_flag > 0:
            if skip_analyzed == "true":
                ii = ii + 1
                continue

        img_read = cv2.imread(i)
        img_read = image_format(img_read)
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


def postanalysis(in_dir, out_dir, scale_object_len, contact_threshold, network_out, output_image, object_size, font_size, text_drawing):
    def node_tunnel_distance(node_p, t_seg):
        # calculate the distance between line AB and point P
        # also obtain the nearest point on a line AB
        # using inner product of vector
        # ---XXX todo: if the AP and AB is parallel but P is not on AB, this function does not work.
        # ---However, I believe that this practically does not happen if manual analysis.
        # ---Thus, I leave this as it is. Maybe need to be fixed in the future.
        ap = node_p - t_seg[0]
        bp = node_p - t_seg[1]
        ab = t_seg[1] - t_seg[0]
        if np.dot(ab, ap) < 0:
            # A is the nearest
            nt_distance = norm(ap)
            nearest_ab_point = t_seg[0]
        elif np.dot(ab, bp) > 0:
            # B is the nearest
            nt_distance = norm(bp)
            nearest_ab_point = t_seg[1]
        else:
            # first obtain the nearest point on ab
            unit_vec_ab = ab/norm(ab)
            nearest_ab_point = t_seg[0] + unit_vec_ab*(np.dot(ap, ab)/norm(ab))
            nt_distance = norm(node_p-nearest_ab_point)
        return nt_distance, nearest_ab_point

    # Data read
    if os.path.exists(out_dir + "/res.pickle"):
        with open(out_dir + '/res.pickle', mode='rb') as f:
            tmanual_output = pickle.load(f)

            # --- todo this part will be removed future
            # Removing node object from old version tmanual res.pickle
            if len(tmanual_output[2][0]) > 7:
                for ii in range(len(tmanual_output[0])):
                    tmanual_output[2][ii].pop(5)
            with open(out_dir + '/res.pickle', mode='wb') as f:
                pickle.dump(tmanual_output, f)
            # ----------

    else:
        return "no res.pickle file in " + out_dir

    df_tunnel = [["serial", "id", "name", "tunnel_length", "tunnel_sequence"]]
    df_summary = [['serial', 'id', 'name', 'tunnel_length_total', 'tunnel_length_1st', 'tunnel_length_2nd',
                   'tunnel_length_3rd', 'tunnel_length_4more', 'tunnel_num_total', 'tunnel_num_1st',
                   'tunnel_num_2nd', 'tunnel_num_3rd', 'tunnel_num_4more']]
    df_net = [["serial", "id", "name", "edge_from", "edge_to", "edge_len"]]

    for i_df in tqdm(range(len(tmanual_output[0]))):
        img_data = ImgData(None, tmanual_output[2][i_df])
        tunnel_len, scale = img_data.measure_tunnel_length(scale_object_len)

        tunnel = img_data.tunnel
        node = img_data.obtain_nodes()
        
        len_t = len(tunnel_len) 

        tunnel_sequence = [1]*len_t  # primary, secondary, tertiary, ...
        contact_tunnelID = [[-1]*len_t, [-1]*len_t]  # connect-tunnel id for starts from and ends at
        node_nearest_point = [[np.array([0, 0])]*len_t, [np.array([0, 0])]*len_t]
        net_edge_from, net_edge_to, net_edge_len  = [], [], []

        # calculation
        if len_t > 0:
            #  1. for each tunnel, check from which tunnel starts
            #  Determine Primary tunnel (= not start from tunnel: >"contact_threshold" pixels)
            for tt_n in range(len_t):
                min_dis = 99999
                nearest_point = np.array([0,0])
                for tt_t in range(len_t):
                    if tt_n != tt_t:
                        if norm(node[0][tt_n]-node[0][tt_t]) < contact_threshold:
                            continue
                        ll = len(tunnel[tt_t])
                        for ttt in range(ll - 1):
                            tunnel_segment = tunnel[tt_t][ttt:(ttt + 2)]
                            dis_temp, nearest_point_temp = node_tunnel_distance(node[0][tt_n], tunnel_segment)
                            if dis_temp < min_dis:
                                min_dis = dis_temp
                                nearest_point = nearest_point_temp
                                node_on_tunnel = tt_t
                if min_dis < contact_threshold:
                    contact_tunnelID[0][tt_n] = node_on_tunnel
                    node_nearest_point[0][tt_n] = nearest_point
                    tunnel_sequence[tt_n] = -1

            #  2. for each tunnel, check at which tunnel ends
            for tt_n in range(len_t):
                min_dis = 99999
                nearest_point = np.array([0,0])
                for tt_t in range(len_t):
                    if tt_n != tt_t:
                        if norm(node[1][tt_n]-node[1][tt_t]) < contact_threshold:
                            continue
                        ll = len(tunnel[tt_t])
                        for ttt in range(ll - 1):
                            tunnel_segment = tunnel[tt_t][ttt:(ttt + 2)]
                            dis_temp, nearest_point_temp = node_tunnel_distance(node[1][tt_n], tunnel_segment)
                            if dis_temp < min_dis:
                                min_dis = dis_temp
                                nearest_point = nearest_point_temp
                                node_on_tunnel = tt_t
                if min_dis < contact_threshold:
                    contact_tunnelID[1][tt_n] = node_on_tunnel
                    node_nearest_point[1][tt_n] = nearest_point

            
            # determine Secondary, Tertiary, ..., tunnel
            tunnel_seq_count = 1
            while True:
                check_tunnel = [i for i, x in enumerate(tunnel_sequence) if x == tunnel_seq_count]
                for tt in range(len_t):
                    if contact_tunnelID[0][tt] in check_tunnel:
                        tunnel_sequence[tt] = tunnel_seq_count + 1
                tunnel_seq_count = tunnel_seq_count + 1
                if len(check_tunnel) == 0:
                    if min(tunnel_sequence) < 0:
                        return "Unexpected error in " + img_data.name + ": cannot get tunnel id."
                    break

            # reconstruct network structure
            if network_out:

                # naming all nodes
                node_name = [[0]*len_t, [0]*len_t]
                for tt in range(len_t):
                    if contact_tunnelID[0][tt] < 0:
                        node_name[0][tt] = "t0" + str(tt).zfill(3) + "_0"
                    else:
                        node_name[0][tt] = "no"+str(tt).zfill(3)+"_0"
                    if contact_tunnelID[1][tt] < 0:
                        node_name[1][tt] = "t1" + str(tt).zfill(3) + "_1"
                    else:
                        node_name[1][tt] = "no"+str(tt).zfill(3)+"_1"

                # check same node with different name
                for tt_0 in range(len_t):
                    for tt_1 in range(len_t):
                        if tt_0 < tt_1:
                            # start node is the same?
                            if norm(node[0][tt_0] - node[0][tt_1]) < contact_threshold:
                                print(node_name[0][tt_1], "->", node_name[0][tt_0] )
                                node_name[0][tt_1] = copy.copy(node_name[0][tt_0])
                        # start-end node is the same?
                        if norm(node[0][tt_0] - node[1][tt_1]) < contact_threshold:
                            print(node_name[1][tt_1], "->", node_name[0][tt_0] )
                            node_name[1][tt_1] = copy.copy(node_name[0][tt_0])
                for tt_0 in range(len_t):
                    for tt_1 in range(len_t):
                        if tt_0 < tt_1:
                            # end node is the same?
                            if norm(node[1][tt_0] - node[1][tt_1]) < contact_threshold:
                                print(node_name[1][tt_1], "->", node_name[1][tt_0] )
                                node_name[1][tt_1] = copy.copy(node_name[1][tt_0])



                # create edges
                tunnel_seq_count = 1
                while True:
                    check_tunnel = [i for i, x in enumerate(tunnel_sequence) if x == tunnel_seq_count]
                    for tt in check_tunnel:
                        ll = len(tunnel[tt])

                        # make a list of nodes that exist on the check_tunnel
                        list_start_node_on_tunnel = [i for i, x in enumerate(contact_tunnelID[0]) if x == tt]
                        list_end_node_on_tunnel   = [i for i, x in enumerate(contact_tunnelID[1]) if x == tt]
                        list_node_on_tunnel = list_start_node_on_tunnel + list_end_node_on_tunnel
                        list_start_or_end = [0]*len(list_start_node_on_tunnel) + [1]*len(list_end_node_on_tunnel)
                        list_tunnel_seg_len = np.array([0]*len(list_node_on_tunnel))

                        # prep for measuring edge length
                        for nn in range(len(list_node_on_tunnel)):
                            tunnel_seg_len = 0
                            nearest_point = node_nearest_point[list_start_or_end[nn]][list_node_on_tunnel[nn]]

                            for ttt in range(ll-1):
                                tunnel_segment = tunnel[tt][ttt:(ttt + 2)]
                                dis_temp = node_tunnel_distance(nearest_point, tunnel_segment)[0]
                                if dis_temp < 0.00001: # == 0 may be affected by float
                                    tunnel_seg_len = tunnel_seg_len + norm(nearest_point-tunnel_segment[0])
                                    break  
                                else:
                                    tunnel_seg_len = tunnel_seg_len + norm(tunnel_segment[1]-tunnel_segment[0])
                            list_tunnel_seg_len[nn] = tunnel_seg_len
                        
                        list_tunnel_seg_len = list_tunnel_seg_len/scale*scale_object_len

                        # reconstruct node-edge structures
                        net_edge_from.append(node_name[0][tt])

                        if len(list_node_on_tunnel) > 0:
                            tunnel_seg_len_order = np.argsort(list_tunnel_seg_len)

                            for nn in range(len(list_node_on_tunnel)):
                                node_temp = tunnel_seg_len_order[nn]
                                net_edge_to.append(node_name[list_start_or_end[node_temp]][list_node_on_tunnel[node_temp]])
                                if nn > 0:
                                    net_edge_len.append(list_tunnel_seg_len[tunnel_seg_len_order[nn]] - list_tunnel_seg_len[tunnel_seg_len_order[nn-1]])
                                else:
                                    net_edge_len.append(list_tunnel_seg_len[tunnel_seg_len_order[nn]])
                                net_edge_from.append(node_name[list_start_or_end[node_temp]][list_node_on_tunnel[node_temp]])

                            net_edge_to.append(node_name[1][tt])
                            net_edge_len.append(tunnel_len[tt] - list_tunnel_seg_len[tunnel_seg_len_order[len(list_node_on_tunnel)-1]])
                        else:
                            net_edge_to.append(node_name[1][tt])
                            net_edge_len.append(tunnel_len[tt])

                    tunnel_seq_count = tunnel_seq_count + 1
                    if len(check_tunnel) == 0:
                        break

                # remove edge from/to the same node
                for tt in reversed(range(len(net_edge_from))):
                    if net_edge_from[tt] == net_edge_to[tt]:
                        print(tt)
                        net_edge_from.pop(tt)
                        net_edge_to.pop(tt)
                        net_edge_len.pop(tt)

        # output
        if network_out:
            for tt in range(len(net_edge_from)):
                df_net.append([img_data.serial, img_data.id, img_data.name, net_edge_from[tt], net_edge_to[tt], net_edge_len[tt]])
        
        for tt in range(len(tunnel_len)):
            df_tunnel.append([img_data.serial, img_data.id, img_data.name, tunnel_len[tt], tunnel_sequence[tt]])

        tunnel_length_total = sum(tunnel_len)
        tunnel_length_1st, tunnel_length_2nd, tunnel_length_3rd, tunnel_length_4more = 0, 0, 0, 0
        tunnel_sequence = [4 if i > 3 else i for i in tunnel_sequence]
        for tt in range(len(tunnel_len)):
            if tunnel_sequence[tt] == 1:
                tunnel_length_1st = tunnel_length_1st + tunnel_len[tt]
            if tunnel_sequence[tt] == 2:
                tunnel_length_2nd = tunnel_length_2nd + tunnel_len[tt]
            if tunnel_sequence[tt] == 3:
                tunnel_length_3rd = tunnel_length_3rd + tunnel_len[tt]
            if tunnel_sequence[tt] == 4:
                tunnel_length_4more = tunnel_length_4more + tunnel_len[tt]

        df_append = [img_data.serial, img_data.id, img_data.name, tunnel_length_total, tunnel_length_1st,
                     tunnel_length_2nd, tunnel_length_3rd, tunnel_length_4more, len(tunnel_len),
                     tunnel_sequence.count(1), tunnel_sequence.count(2), tunnel_sequence.count(3),
                     tunnel_sequence.count(4)]
        df_summary.append(df_append)

        # image output
        if output_image:
            if os.path.exists(in_dir + img_data.name):
                img = cv2.imread(in_dir + img_data.name)
                img = image_format(img)
                img_data.colored_image_output(img, tunnel_sequence, out_dir, object_size, font_size, text_drawing)
            else:
                print(img_data.name + ": not find image file")

    f = open(out_dir+'df_tunnel.csv', 'w', newline='')
    writer = csv.writer(f)
    writer.writerows(df_tunnel)
    f.close()
    
    f = open(out_dir+'df_summary.csv', 'w', newline='')
    writer = csv.writer(f)
    writer.writerows(df_summary)
    f.close()

    if network_out:
        f = open(out_dir+'df_net.csv', 'w', newline='')
        writer = csv.writer(f)
        writer.writerows(df_net)
        f.close()

    return "Post-analysis finished"


def gui():
    sg.theme('Dark')
    frame_file = sg.Frame('Files', [
        [sg.Text("In   "),
         sg.InputText('Input folder', enable_events=True, size=(20, 1)),
         sg.FolderBrowse(button_text='select', size=(6, 1), key="-IN_FOLDER_NAME-"),
         sg.InputText(' or files', enable_events=True, size=(20, 1)),
         sg.FilesBrowse(button_text='select', size=(6, 1), key="-IN_FILES_NAME-")
         ],
        [sg.Text("Out"),
         sg.InputText('Output folder', enable_events=True, size=(20, 1)),
         sg.FolderBrowse(button_text='select', size=(6, 1), key="-OUT_FOLDER_NAME-"),
         sg.Text("(* will be created if not specified)")
         ],
        [sg.Text("File extension (default = jpg)"),
         sg.In(key='-FILE_EXTENSION-', size=(15, 1))]
    ], size=(800, 150))

    frame_param = sg.Frame('Parameters', [
        [sg.Text("Measurement:", size=(12,1)),
         sg.Text("skip analyzed files", size=(15,1)),
         sg.Combo(['true', 'false'], default_value="true", size=(6, 1), key="-SKIP_ANALYZED-")
        ],
        [sg.Text("Post-analysis:", size=(12,1)), 
         sg.Text("scale length (mm)", size=(15,1)),
         sg.In(key='-SCALE_OBJECT-', size=(6, 1)),
         sg.Text("output image", size=(12,1)),
         sg.Combo(['true', 'false'], default_value="true", size=(6, 1), key="-OUTPUT_IMAGE-")],
        [sg.Text("", size=(12,1)),
         sg.Text("contact thld (def 10 px)"),
          sg.In(key='-CONTACT_THRESHOLD-', size=(6, 1)),
         sg.Text("network produce", size=(12,1)),
         sg.Combo(['true', 'false'], default_value="true", size=(6, 1), key="-NETWORK-")
        ],
        [sg.Text("Drawing:", size=(12,1)),
         sg.Text("line width (def 5)"),
         sg.In(key='-LINE_WIDTH-', size=(6, 1)),

         sg.Text("font size (def 2)"),
         sg.In(key='-FONT_SIZE-', size=(6, 1)),

         sg.Text("num draw"),
         sg.Combo(['true', 'false'], default_value="true", size=(6, 1), key="-TEXT_DRAWNING-")
        ]
    ], size=(800, 160))

    frame_measure_buttom = sg.Frame('', [
        [sg.Submit(button_text='Measurement start', size=(20, 3), key='measurement_start')]], size=(180, 100))
    
    frame_post_buttom = sg.Frame('', [
        [sg.Submit(button_text='Post-analysis start', size=(20, 3), key='post_analysis_start',
                   button_color=('white', 'chocolate'))]], size=(180, 100))
    frame3 = sg.Frame('Manual', [
        [sg.Text("Images should be named in 'id_number.jpg'\n"
                 "    e.g., TunnelA_00.jpg, TunnelA_01.jpg, ..., TunnelA_20.jpg, TunnelB_00.jpg, TunnelB_01.jpg, ...")],
        [sg.Text("Measurement", size=(12,1)),
         sg.Text("sequentially process images with below process (LC: left click, RC: right click)")],
        [sg.Text("", size=(1,3)),
         sg.Text("1. Check", size=(10,3)),
         sg.Text("-LC(or V):analyze  -RC(or N):skip \n"
                 "-Esc:exit (saved)  -B:previous image\n"
                 "-R:re-analyze (append to previous)  -A:re-analyze (from the scratch)")],
        [sg.Text("", size=(1,2)),
         sg.Text("2. Ref point", size=(10,2)),
         sg.Text("-LC:the same landscape point across images (used for calibration).\n"
                 "-RC:skip")],
        [sg.Text("", size=(1,3)),
         sg.Text("3. Measure", size=(10,3)),
         sg.Text("-LC:measure tunnel length.  -RC to next or finish at the end.\n"
                 "-Q:undo   -Z:zoom in (x2-x8)  -X:stop zoom  -E:go-to-end  -Esc:finish\n"
                 " Branching tunnels should be on the previous tunnels line")],
        [sg.Text("", size=(1,1)),
         sg.Text("4. Set scale", size=(10,1)),
         sg.Text("-Drag to set the scale  -RC to finish.")],
        [sg.Text("Post-analysis", size=(12,1)),
         sg.Text("use smaller node-gallery contact threshold for small galleries relative to image")]
        ], size=(1000, 400))
    
    frame_buttons = sg.Column([[frame_measure_buttom], [frame_post_buttom]])
    frame_input = sg.Column([[frame_file],[frame_param]])
    layout = [[frame_input, frame_buttons], [frame3]]
    
    window = sg.Window('TManual, a tool to assist in measuring length development of structures',
                       layout, resizable=True)
    
    while True:
        event, values = window.read()
    
        if event is None:
            print('exit')
            break
        else:
            if event == 'measurement_start':

                # file info
                if len(values["-IN_FOLDER_NAME-"]) == 0 and len(values["-IN_FILES_NAME-"]) == 0:
                    print("no input!")
                    continue

                elif len(values["-IN_FILES_NAME-"]) > 0:  # file names provided
                    in_files = values["-IN_FILES_NAME-"]
                    if len(values["-OUT_FOLDER_NAME-"]) == 0:
                        if len(values["-IN_FOLDER_NAME-"]) > 0:
                            in_dir = values["-IN_FOLDER_NAME-"] + "/"
                            out_dir = in_dir+"/tmanual/"
                            if not os.path.exists(out_dir):
                                os.makedirs(out_dir)
                        else:
                            print("no output directly!")
                            continue
                    else:
                        out_dir = values["-OUT_FOLDER_NAME-"]+"/"
                    in_dir = 0

                else:
                    in_dir = values["-IN_FOLDER_NAME-"]+"/"
                    in_files = 0
                    if len(values["-OUT_FOLDER_NAME-"]) == 0:
                        out_dir = in_dir+"/tmanual/"
                        if not os.path.exists(out_dir):
                            os.makedirs(out_dir)
                    else:
                        out_dir = values["-OUT_FOLDER_NAME-"]+"/"
                
                # parameters
                skip_analyzed = values["-SKIP_ANALYZED-"]
                if len(values["-FILE_EXTENSION-"]) == 0:
                    file_extension = "jpg"
                else:
                    file_extension = values["-FILE_EXTENSION-"]

                if len(values["-LINE_WIDTH-"]) == 0:
                    object_size = 5
                else:
                    object_size = int(values["-LINE_WIDTH-"])
                
                if len(values["-FONT_SIZE-"]) == 0:
                    font_size = 2
                else:
                    font_size = int(values["-FONT_SIZE-"])
                
                text_drawing = values["-TEXT_DRAWNING-"]
                if text_drawing == "true":
                    text_drawing = True
                else:
                    text_drawing = False

                print("input dir: "+str(in_dir))
                print("input files: "+str(in_files))
                print("output dir: "+out_dir)
                measurement(in_dir, in_files, out_dir, skip_analyzed, file_extension, object_size, font_size, text_drawing)
    
            elif event == 'post_analysis_start':
                output_image = values["-OUTPUT_IMAGE-"]
                if output_image:
                    if len(values["-IN_FOLDER_NAME-"]) == 0:
                        print("no input!")
                        continue
                    else:
                        in_dir = values["-IN_FOLDER_NAME-"] + "/"

                if len(values["-OUT_FOLDER_NAME-"]) == 0:
                    if len(values["-IN_FOLDER_NAME-"]) > 0:
                        out_dir = in_dir + "/tmanual/"
                    else:
                        print("no input!")
                else:
                    out_dir = values["-OUT_FOLDER_NAME-"] + "/"


                try:
                    float(values['-SCALE_OBJECT-'])
                except ValueError:
                    scale_object_len = float(1)
                    print("Warning: Scale object length is not indicated. Put 1 (mm) instead.")
                else:
                    scale_object_len = float(values["-SCALE_OBJECT-"])

                if len(values["-LINE_WIDTH-"]) == 0:
                    object_size = 5
                else:
                    object_size = int(values["-LINE_WIDTH-"])
                
                if len(values["-FONT_SIZE-"]) == 0:
                    font_size = 2
                else:
                    font_size = int(values["-FONT_SIZE-"])

                text_drawing = values["-TEXT_DRAWNING-"]
                if text_drawing == "true":
                    text_drawing = True
                else:
                    text_drawing = False

                if len(values["-CONTACT_THRESHOLD-"]) == 0:
                    contact_threshold = 10
                else:
                    contact_threshold = int(values["-CONTACT_THRESHOLD-"])

                network = values["-NETWORK-"]
                if network == "true":
                    network = True
                else:
                    network = False
                
                
                message = postanalysis(in_dir, out_dir, scale_object_len, contact_threshold, network, output_image, object_size, font_size, text_drawing)
                sg.popup(message)            

    window.close()

gui()
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

vcol = [[37, 231, 253], [98, 201, 94], [140, 145, 33],  [139, 82, 59], [84, 1, 68]]  # viridis colors in BGR

note_pos = [40, 100]
note_pos2 = [40, 200]
note_pos3 = [40, 300]


def tunnel_draw(img_t, current_tunnel_t, col_t, object_size):
    l_t = len(current_tunnel_t)
    for t_seg_iter in range(l_t-1):
        cv2.line(img_t, current_tunnel_t[t_seg_iter], current_tunnel_t[t_seg_iter+1], col_t, object_size)
    return img_t


def outlined_text(img_o, text_o, ref_o, col_o, font_size):
    cv2.putText(img_o, text_o, ref_o, cv2.FONT_HERSHEY_PLAIN, font_size, (0, 0, 0), font_size*5, cv2.LINE_AA)
    cv2.putText(img_o, text_o, ref_o, cv2.FONT_HERSHEY_PLAIN, font_size, col_o, font_size, cv2.LINE_AA)
    return img_o


def object_drawing(img_d, ref_d=None, scale_d=None, tunnel_d=None, node_d=None, offset=0,
                   col_t=None, object_size = 5, font_size = 2, draw_number = True):
    if ref_d is not None:
        cv2.circle(img_d, ref_d, object_size*5, vcol[0], 5)
        cv2.circle(img_d, ref_d, object_size, (0, 0, 0), -1)
    if col_t is None:
        col_t = [0, 0, 0]
    if scale_d is not None:
        cv2.line(img_d, scale_d[0], scale_d[1], (0, 0, 255), object_size)
    if tunnel_d is not None:
        for tt in range(len(tunnel_d)):
            img_d = tunnel_draw(img_d, tunnel_d[tt], col_t, object_size)
            if draw_number:
                img_d = outlined_text(img_d, str(tt+offset), tunnel_d[tt][0]-np.array([object_size, 0]), col_t, font_size)
    if node_d is not None:
        for tt in range(len(node_d)):
            cv2.circle(img_d, node_d[tt], object_size, vcol[0], -1)
            if draw_number:
                img_d = outlined_text(img_d, str(tt+offset), node_d[tt]+np.array([object_size, 0]), vcol[0], font_size)
    return img_d


def image_format(img):  # all images are reformatted in 2000xH for measurement
    h, w = img.shape[:2]
    if w < h:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img = cv2.resize(img, dsize=(2000, round(h*2000/w)))
    return img


class ImgData:
    def __init__(self, img_loc, data_values=None):
        if data_values is None:
            self.name = os.path.basename(img_loc)
            self.id = self.name.split('_')[0]
            self.serial = int(re.sub(".jpg", "", self.name.split('_')[1]))
            data_values = [self.name, self.id, self.serial, np.array([0, 0]), [], [], [[0, 0], [0, 0]], 0]
        self.name = data_values[0]
        self.id = data_values[1]
        self.serial = data_values[2]
        self.ref_xy = data_values[3]
        self.tunnel = data_values[4]
        self.node = data_values[5]
        self.scale_xy = data_values[6]
        self.analyze_flag = data_values[7]

    def output_image_data(self):
        return [self.name, self.id, self.serial, self.ref_xy, self.tunnel, self.node, self.scale_xy, self.analyze_flag]

    def note_plot(self, img, note_message, font_size):
        cv2.putText(img, note_message+'('+self.name+')',
                    note_pos, cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), font_size, cv2.LINE_AA)
        return img

    def object_plot(self, img, offset, col_t, object_size, font_size, draw_number):
        img = object_drawing(img, self.ref_xy, self.scale_xy, self.tunnel, self.node, offset, col_t, object_size, font_size, draw_number)
        return img

    def change_ref(self, ref_xy_new):
        self.scale_xy[0] = self.scale_xy[0] - self.ref_xy + ref_xy_new
        self.scale_xy[1] = self.scale_xy[1] - self.ref_xy + ref_xy_new
        for tt in range(len(self.node)):
            self.node[tt] = self.node[tt] - self.ref_xy + ref_xy_new
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
        return tunnel_len

    def image_output(self, img, out_dir, object_size, font_size):
        img = object_drawing(img, self.ref_xy, self.scale_xy, self.tunnel, self.node, 0, vcol[4], object_size, font_size, draw_number=True)
        cv2.putText(img, self.id+"_"+str(self.serial), note_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), font_size, cv2.LINE_AA)
        if len(self.tunnel) < 1:
            cv2.putText(img, "no tunnel", note_pos2,
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), font_size, cv2.LINE_AA)
        cv2.imwrite(out_dir+"/" + self.name, img)

    def colored_image_output(self, img, assigned_nodes, tunnel_sequence, out_dir, object_size, font_size):
        cv2.putText(img, self.id+"_"+str(self.serial), note_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), font_size, cv2.LINE_AA)
        if len(self.tunnel) < 1:
            cv2.putText(img, "no tunnel", note_pos2,
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), font_size, cv2.LINE_AA)
        for tt in range(len(self.tunnel)):
                img = tunnel_draw(img, self.tunnel[tt], vcol[5-tunnel_sequence[tt]], object_size)
        for nn in range(len(self.node)):
            if assigned_nodes[nn] == 1:
                cv2.circle(img, self.node[nn], object_size, vcol[0], -1)
            else:
                cv2.circle(img, self.node[nn], object_size, vcol[0], 2)
        for tt in range(len(self.tunnel)):
            img = outlined_text(img, str(tt), self.tunnel[tt][0]-np.array([object_size, 0]), vcol[5-tunnel_sequence[tt]], font_size)
        for nn in range(len(self.node)):
            img = outlined_text(img, str(nn), self.node[nn]+np.array([object_size, 0]), vcol[0], font_size)
        cv2.imwrite(out_dir+"colored_"+self.name, img)

    def analyze_done(self):
        self.analyze_flag = 1

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
        img = img_data.note_plot(img_read.copy(), '3.Length  ', font_size)

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


def postanalysis(in_dir, out_dir, scale_object_len, output_image, object_size, font_size):
    def collision_detect(point, line, collision_distance=10):
        if line[0][0] == line[1][0] and line[0][1] == line[1][1]:
            return False
        vec_oa = line[0] - point
        vec_ob = line[1] - point

        len_oa2 = sum(vec_oa ** 2)
        len_ob2 = sum(vec_ob ** 2)
        inner_ab = sum(vec_oa * vec_ob)

        a = float(len_oa2 + len_ob2 - 2 * inner_ab)
        b = float(-2 * len_oa2 + 2 * inner_ab)
        c = float(len_oa2 - collision_distance * collision_distance)

        det = b * b - 4 * a * c
        if det < 0:
            return False
        s1 = (-b - math.sqrt(det)) / (2 * a)
        s2 = (-b + math.sqrt(det)) / (2 * a)
        return (s1 <= 1) and (0 <= s2)

    # Data read
    if os.path.exists(out_dir + "/res.pickle"):
        with open(out_dir + '/res.pickle', mode='rb') as f:
            tmanual_output = pickle.load(f)
    else:
        return "no res.pickle file in " + out_dir

    df_tunnel = [["serial", "id", "name", "tunnel_length", "tunnel_sequence"]]
    df_summary = [['serial', 'id', 'name', 'tunnel_length_total', 'tunnel_length_1st', 'tunnel_length_2nd',
                  'tunnel_length_3rd', 'tunnel_length_4more', 'tunnel_num_total', 'tunnel_num_1st',
                  'tunnel_num_2nd', 'tunnel_num_3rd', 'tunnel_num_4more', 'node_num']]
    for i_df in tqdm(range(len(tmanual_output[0]))):
        img_data = ImgData(None, tmanual_output[2][i_df])
        tunnel_len = img_data.measure_tunnel_length(scale_object_len)

        tunnel_sequence = [1] * len(tunnel_len)
        node = img_data.node
        tunnel = img_data.tunnel
        node_start_tunnelID = [-1] * len(node)
        assigned_nodes = [-1] * len(node)

        if len(node) > 0:
            #  Determine Primary tunnel (= not start from nodes: >10 pixels)
            #  Otherwise (<10 pixels), which tunnel each node starts from?
            for tt in range(len(tunnel_len)):
                node_tunnel0_dis = np.sqrt(np.sum((node - tunnel[tt][0]) ** 2, axis=1))
                min_dis = min(node_tunnel0_dis)
                if min_dis < 10:
                    node_start_tunnelID = np.where(node_tunnel0_dis > min_dis, node_start_tunnelID, tt) ## np.where(condition, true, otherwise)
                    tunnel_sequence[tt] = -1

            for iter_n in range(len(assigned_nodes)):
                if node_start_tunnelID[iter_n] < 0:
                    assigned_nodes[iter_n] = 2
            # Secondary, Tertiary, ..., tunnel
            end, tunnel_seq_count = 0, 1
            while end == 0:
                nodes_to_check = [i for i, x in enumerate(assigned_nodes) if x < 0]
                tunnels_to_check = [i for i, x in enumerate(tunnel_sequence) if x == tunnel_seq_count]
                for nn in nodes_to_check:
                    for tt in tunnels_to_check:
                        ll = len(tunnel[tt])
                        for ttt in range(ll - 1):
                            tunnel_segment = tunnel[tt][ttt:(ttt + 2)]
                            if collision_detect(node[nn], tunnel_segment, 10):
                                tunnel_sequence[node_start_tunnelID[nn]] = tunnel_seq_count + 1
                                assigned_nodes[nn] = 1
                                break
                tunnel_seq_count = tunnel_seq_count + 1
                if min(assigned_nodes) > 0:
                    end = 1
                if tunnel_seq_count > 100:
                    return "Error in " + img_data.name + ": Invalid nodes. Recheck if all nodes are on the tunnel lines"
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
                     tunnel_sequence.count(4), len(node)]
        df_summary.append(df_append)

        # image output
        if output_image:
            img = cv2.imread(in_dir + img_data.name)
            img = image_format(img)
            img_data.colored_image_output(img, assigned_nodes, tunnel_sequence, out_dir, object_size, font_size)

    f = open(out_dir+'df_tunnel.csv', 'w', newline='')
    writer = csv.writer(f)
    writer.writerows(df_tunnel)
    f.close()
    
    f = open(out_dir+'df_summary.csv', 'w', newline='')
    writer = csv.writer(f)
    writer.writerows(df_summary)
    f.close()

    return "Post-analysis finished"

def gui():
    sg.theme('Dark')
    frame_file = sg.Frame('Files', [
        [sg.Text("In   "),
         sg.InputText('Input folder', enable_events=True, size=(30, 1)),
         sg.FolderBrowse(button_text='select', size=(6, 1), key="-IN_FOLDER_NAME-"),
         sg.InputText(' or files', enable_events=True, size=(25, 1)),
         sg.FilesBrowse(button_text='select', size=(6, 1), key="-IN_FILES_NAME-")
         ],
        [sg.Text("Out"),
         sg.InputText('Output folder', enable_events=True, size=(30, 1)),
         sg.FolderBrowse(button_text='select', size=(6, 1), key="-OUT_FOLDER_NAME-"),
         sg.Text("(* will be created if not specified)")
         ],
        [sg.Text("File extension (default = jpg)"),
         sg.In(key='-FILE_EXTENSION-', size=(15, 1))]
    ], size=(1000, 160))

    frame_measure_param = sg.Frame('Measurement parameters', [
        [sg.Text("skip analyzed files"),
         sg.Combo(['true', 'false'], default_value="true", size=(6, 1), key="-SKIP_ANALYZED-")
        ],
        [sg.Text("Line width (default = 5)"),
         sg.In(key='-LINE_WIDTH-', size=(8, 1)),

         sg.Text("Font size (default = 2)"),
         sg.In(key='-FONT_SIZE-', size=(8, 1))
        ]
    ], size=(1000, 100))
    
    frame_post_param = sg.Frame('Post-analysiss parameters', [
        [sg.Text("length of the scale object (in mm)"),
         sg.In(key='-SCALE_OBJECT-', size=(8, 1)),
         sg.Text("output image (post)"),
         sg.Combo(['true', 'false'], default_value="true", size=(6, 1), key="-OUTPUT_IMAGE-")
         ]    ], size=(1000, 60))

    frame2 = sg.Frame('Measurement', [
        [sg.Submit(button_text='Measurement start', size=(20, 3), key='measurement_start')]], size=(180, 100))
    
    frame4 = sg.Frame('Post-analysis', [
        [sg.Submit(button_text='Post-analysis start', size=(20, 3), key='post_analysis_start',
                   button_color=('white', 'chocolate'))]], size=(180, 100))
    
    frame3 = sg.Frame('README', [
        [sg.Text("TManual, a tool to assist in manually measuring length development of gallery structures")],
        [sg.Text("Files should be consecutive image files, named 'id_number.jpg'\n"
                 "    e.g., TunnelA_00.jpg, TunnelA_01.jpg, TunnelA_02.jpg, ..., "
                 "TunnelA_20.jpg, TunnelB_00.jpg, TunnelB_01.jpg, ...")],
        [sg.Text("The program reads all image files in input sequentially to ask the following;\n"
                 "(* LC: left click, RC: right click)")],
        [sg.Text("1. Show the image\n"
                 "   LC (or V):analyze  RC (or N):skip  Esc:exit (saved)  B:previous image  R:analyze from the scratch")],
        [sg.Text("2. Set (0,0) coordinate\n"
                 "   LC the same landscape point across images (used for calibration). RC to skip")],
        [sg.Text("3. Measure tunnel length\n"
                 "   LC to measure tunnel length.\n"
                 "   RC to next or finish at the end.\n"
                 "   Q:undo   Z:zoom in (x2)  X:stop zoom  Esc:finish\n"
                 "   * Branching tunnels should start in contact with the line of previous tunnels")],
        [sg.Text("4. Identify nodes\n"
                 "   LC:node  RC:finish  Q:undo  Z:zoom in (x2)  X:stop zoom\n"
                 "   * Place nodes on the tunnel lines")],
        [sg.Text("5. Set scale\n"
                 "   Drag to set the scale. RC to finish.\n")],
        [sg.Text("Post-analysis\n"
                 "   Identify primary, secondary, tertiary, ..., tunnels and summarize analysis results\n"
                 "   Read res.pickle from output folder")]], size=(1200, 700))
    
    frame_buttons = sg.Column([[frame2], [frame4]])
    frame_param = sg.Column([[frame_file],[frame_measure_param],[frame_post_param]])
    layout = [[frame_param, frame_buttons], [frame3]]
    
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
                
                print("input dir: "+str(in_dir))
                print("input files: "+str(in_files))
                print("output dir: "+out_dir)
                message = measurement(in_dir, in_files, out_dir, skip_analyzed, file_extension, object_size, font_size)
                sg.popup(message)
    
            elif event == 'post_analysis_start':
                output_image = values["-OUTPUT_IMAGE-"]
                if output_image:
                    if len(values["-IN_FOLDER_NAME-"]) == 0:
                        print("no input!")
                        continue
                    else:
                        in_dir = values["-IN_FOLDER_NAME-"] + "/"
                try:
                    float(values['-SCALE_OBJECT-'])
                except ValueError:
                    print("invalid scale value")
                else:
                    if len(values["-OUT_FOLDER_NAME-"]) == 0:
                        if len(values["-IN_FOLDER_NAME-"]) > 0:
                            out_dir = in_dir + "/tmanual/"
                        else:
                            print("no input!")
                    else:
                        out_dir = values["-OUT_FOLDER_NAME-"] + "/"

                    scale_object_len = float(values["-SCALE_OBJECT-"])

                    if len(values["-LINE_WIDTH-"]) == 0:
                        object_size = 5
                    else:
                        object_size = int(values["-LINE_WIDTH-"])
                    
                    if len(values["-FONT_SIZE-"]) == 0:
                        font_size = 2
                    else:
                        font_size = int(values["-FONT_SIZE-"])
                    
                    message = postanalysis(in_dir, out_dir, scale_object_len, output_image, object_size, font_size)
                    sg.popup(message)
    
    window.close()

def tmanual():
    gui()

tmanual()
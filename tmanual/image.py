"""
Functions and class for image analysis
"""


import os
import cv2
import copy
import numpy as np
import math
import re
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
    def __init__(self, img_name, data_values=None):
        if data_values is None:
            self.name = img_name
            self.id = self.name.split('_')[0]
            if len(self.name.split("_")) > 1:
                self.serial = int(self.name.split('_')[1])
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

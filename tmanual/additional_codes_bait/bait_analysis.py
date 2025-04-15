"""
Obtain meta data for bait experiments with a 3m-arena by kaitlin
 ----TODO will be added to main as a function to get meta data of the experiments? maybe
 ----Note plot should be independent function
"""

import glob
import os
import pickle
import cv2
import numpy as np
from numpy.linalg import norm
import pyautogui as pag
from keyboard import press
import math
import tmanual
from tqdm import tqdm
import copy
import csv
#import PySimpleGUI as sg
import FreeSimpleGUI as sg

note_pos = [40, 100]
v_col = [[37, 231, 253], [98, 201, 94], [140, 145, 33],  [139, 82, 59], [84, 1, 68]]  # viridis colors in BGR


class ExperimentMetaData:
    def __init__(self, experiment_id, img_name, data_values=None):
        if data_values is None:
            self.name = img_name
            self.id = experiment_id
            data_values = [self.name, self.id, np.array([0, 0]), [], [], 0]
        self.name = data_values[0]
        self.id = data_values[1]
        self.initial = data_values[2]
        self.baits = data_values[3]
        self.analyze_flag = data_values[4]

    def output_meta_data(self):
        return [self.name, self.id, self.initial, self.baits, self.analyze_flag]

    def note_plot(self, img, note_message, font_size):
        cv2.putText(img, note_message+'('+self.name+')',
                    note_pos, cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), font_size, cv2.LINE_AA)
        return img

    def image_output(self, img, in_dir, object_size, font_size):
        cv2.putText(img, self.id, note_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), font_size, cv2.LINE_AA)
        cv2.circle(img, self.initial, object_size * 5, v_col[0], object_size)
        cv2.circle(img, self.initial, object_size, (0, 0, 0), -1)
        for i in range(len(self.baits)):
            cv2.circle(img, self.baits[i][0], radius=int(self.baits[i][1] / 2),
                       color=(0, 0, 255), thickness=object_size)
        cv2.imwrite(in_dir + os.sep + 'meta' + os.sep + self.id + ".jpg", img)

    def analyze_done(self):
        self.analyze_flag = 1


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
    if norm(ab) == 0 or np.dot(ab, ap) < 0:
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


def check_intersection(t_seg, P, dis):
    """
    Returns a coordinate (x, y) on the tunnel segment AB that intersect with a circle with center = P and radius = dis.
    Returns None if such C does not exist.
    """
    A, B = t_seg[0], t_seg[1]
    AB = B - A
    BP_dis = norm(B-P)
    AP_dis = norm(A-P)

    nearest_dis, nearest_point = node_tunnel_distance(P, t_seg)
    if nearest_dis > dis:
        return None
    if BP_dis < dis and AP_dis < dis:
        return None
    if norm(AB) == 0:
        return None
    
    # Calculate the vertical distance from point P to line overlaying AB
    unit_vec_ab = AB/norm(AB)
    nearest_ab_point = A + unit_vec_ab*(np.dot(P-A, AB)/norm(AB))
    distance_to_line = norm(P-nearest_ab_point)
    
    # Calculate the distance along line AB where point C is located
    distance_along_line = np.sqrt(dis**2 - distance_to_line**2)
    
    # Calculate the coordinates of point C
    if AP_dis > BP_dis:
        C = nearest_ab_point - distance_along_line * unit_vec_ab   
    else:
        C = nearest_ab_point + distance_along_line * unit_vec_ab
    return C



def get_metadata(in_dir, out_dir, file_extension, object_size, font_size):
    # Data read
    if os.path.exists(out_dir + "bait" + os.sep + "meta.pickle"):
        with open(out_dir + "bait" + os.sep + "meta.pickle", mode='rb') as f:
            exp_meta_output = pickle.load(f)
    else:
        print("new analysis start")
        if not os.path.exists(out_dir + "bait"):
            os.makedirs(out_dir + os.sep + "bait")
        if not os.path.exists(out_dir + "angle"):
            os.makedirs(out_dir + os.sep + "angle")
        exp_meta_output = [[], []]  # store Ids, Results

    name1 = glob.glob(in_dir + os.sep + '*.' + file_extension)
    num_file = len(name1)

    ii = 0
    while ii < num_file:

        # region --- 0. Load data ---#
        image_name = os.path.basename(name1[ii])
        experiment_id = image_name.split('_')[0]

        if experiment_id in exp_meta_output[0]:
            ii = ii + 1
            continue

        img_read = cv2.imread(in_dir + image_name)
        if img_read is None:
            print("Error. file is not readable: " + in_dir + image_name + ". Skip.")
            ii = ii + 1
            continue

        exp_meta_data = ExperimentMetaData(experiment_id, image_name, None)

        # create window
        img_shape = np.array([img_read.shape[1], img_read.shape[0]])
        window_name = "window"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        scr_w, scr_h = pag.size()
        if scr_h > scr_w * img_shape[1] / img_shape[0]:
            cv2.resizeWindow(window_name, scr_w-200, int(scr_w * img_shape[1] / img_shape[0])-200)
        else:
            cv2.resizeWindow(window_name, int(scr_h * img_shape[0] / img_shape[1])-200, scr_h-200)
        # endregion ------

        # region --- 1.  Define initial point --- #
        img = exp_meta_data.note_plot(img_read.copy(), '1.Initial point', font_size)
        cv2.circle(img, exp_meta_data.initial, object_size * 5, v_col[0], object_size)
        cv2.circle(img, exp_meta_data.initial, object_size, (0, 0, 0), -1)
        cv2.imshow('window', img)

        def get00(event, x, y, flags, params):
            if event == cv2.EVENT_LBUTTONDOWN:
                exp_meta_data.initial = np.array([x, y])
                press("enter")
            elif event == cv2.EVENT_RBUTTONDOWN:
                press("enter")
        cv2.setMouseCallback('window', get00)
        cv2.waitKey()
        # ----- endregion

        # region --- 2. Bait locate ---#
        img = exp_meta_data.note_plot(img_read.copy(), '2. Bait', font_size)
        cv2.imshow(window_name, img)

        def bait_draw(event, x, y, flags, param):
            nonlocal x0, y0, diameter, drawing, end
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing, end, x0, y0 = True, 0,  x, y
            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    cv2.circle(img_copy, center=(int((x - x0) / 2) + x0, int((y - y0) / 2) + y0),
                               radius=int(math.sqrt((x0 - x) ** 2 + (y0 - y) ** 2) / 2),
                               color=(0, 0, 255), thickness=2)
            elif event == cv2.EVENT_LBUTTONUP:
                diameter = math.sqrt((x0 - x) ** 2 + (y0 - y) ** 2)
                x0 = ((x - x0) / 2) + x0
                y0 = ((y - y0) / 2) + y0
                cv2.circle(img_copy, center=(int(x0), int(y0)), radius=int(diameter / 2),
                           color=(0, 0, 255), thickness=2)
                drawing = False
            elif event == cv2.EVENT_RBUTTONDOWN:
                if end == 1:
                    press('f')
                elif end == 0:
                    exp_meta_data.baits.append([np.array([int(x0), int(y0)]), diameter])
                    cv2.circle(img, center=(int(x0), int(y0)), radius=int(diameter / 2),
                               color=(0, 0, 255), thickness=2)
                    end = 1
            press('enter')

        x0, y0, diameter = 0, 0, 0
        img_copy = img.copy()
        end, drawing = 1, False

        while True:
            cv2.imshow('window', img_copy)
            if drawing:
                img_copy = img.copy()
            cv2.setMouseCallback('window', bait_draw)
            k = cv2.waitKey(0)
            if k == ord('f') or k == 27:
                break
        if k == 27:
            break
        # ----- endregion

        # region----- Output -----#
        exp_meta_data.analyze_done()

        # delete old data
        duplicate_data_index = list(
            set([i for i, x in enumerate(exp_meta_output[0]) if x == exp_meta_data.id])
        )
        if len(duplicate_data_index) > 0:
            print("delete duplicate data")
            exp_meta_output[0].pop(duplicate_data_index[0])
            exp_meta_output[1].pop(duplicate_data_index[0])

        # add new data
        exp_meta_output[0].append(exp_meta_data.id)
        exp_meta_output[1].append(exp_meta_data.output_meta_data())
        exp_meta_data.image_output(img_read.copy(), in_dir, object_size, font_size)

        # write
        with open(out_dir + 'bait' + os.sep + 'meta.pickle', mode='wb') as f:
            pickle.dump(exp_meta_output, f)

        ii = ii + 1
        # ----- endregion
    cv2.destroyAllWindows()


def bait_post_analysis(in_dir, out_dir, scale_object_len, max_num_virtual_bait, concentric_circles, object_size, font_size):
    
    # Data read (res.pickle and meta.pickle)
    if os.path.exists(out_dir + os.sep + "res.pickle"):
        with open(out_dir + os.sep + "res.pickle", mode='rb') as f:
            tmanual_output = pickle.load(f)
    else:
        return "no res.pickle file in " + out_dir 

    if os.path.exists(out_dir + os.sep + 'bait' + os.sep + 'meta.pickle'):
        with open(out_dir + os.sep + 'bait' + os.sep + 'meta.pickle', mode='rb') as f:
            exp_meta_output = pickle.load(f)
    else:
        return "no meta.pickle file in " + out_dir + os.sep + 'bait'


    # Main
    df_bait = [["serial", "id", "name", "num_virtual_bait", "bait_id", "encounter"]]
    df_angle = [["serial", "id", "name", "radius", "angle_rad"]]
    for i_df in tqdm(range(len(tmanual_output[0]))):

        # load data
        img_data = tmanual.ImgData(None, tmanual_output[2][i_df])
        tunnel_len, scale = img_data.measure_tunnel_length(scale_object_len)
        tunnel = img_data.tunnel
        len_t = len(tunnel_len)

        exp_meta_data = ExperimentMetaData(None, None, exp_meta_output[1][exp_meta_output[0].index(img_data.id)])

        img = cv2.imread(in_dir + os.sep + img_data.name)
        if img is None:
            print("There is no file named " + img_data.name + "!! Cannot output the image results.")

        # region --- 1. Virtual bait encounter ---#
        bait_size = int((exp_meta_data.baits[0][1] + exp_meta_data.baits[1][1]) / 2)
        bait1 = exp_meta_data.baits[0][0]
        bait2 = exp_meta_data.baits[1][0]
        if bait1[0] > bait2[0]:
            bait1, bait2 = bait2, bait1

        bait_positions = [[bait1, bait2]]
        for i in range(1, max_num_virtual_bait + 1, 1):
            bait_temp = [bait1, bait2]
            for ii in range(i):
                bait_temp_temp = bait1 + (bait2 - bait1) / (i+1) * (ii+1)
                bait_temp_temp = bait_temp_temp.astype(int)
                bait_temp.append(bait_temp_temp)
            bait_positions.append(bait_temp)

        # for each tunnel, check if contact with bait
        bait_contact = copy.deepcopy(bait_positions)
        for tt_b in range(len(bait_positions)):
            for tt_bb in range(len(bait_positions[tt_b])):
                min_dis = 99999
                nearest_point = np.array([0,0])
                for tt_t in range(len_t):
                    ll = len(tunnel[tt_t])
                    for ttt in range(ll - 1):
                        tunnel_segment = tunnel[tt_t][ttt:(ttt + 2)]
                        dis_temp, nearest_point_temp = node_tunnel_distance(bait_positions[tt_b][tt_bb], tunnel_segment)
                        if dis_temp < min_dis:
                            min_dis = dis_temp
                            nearest_point = nearest_point_temp
                if min_dis < bait_size/2:
                    bait_contact[tt_b][tt_bb] = 1
                else:
                    bait_contact[tt_b][tt_bb] = 0

        # output data
        for tt in range(len(bait_contact)):
            for ttt in range(len(bait_contact[tt])):
                if ttt < 2:
                    df_bait.append([img_data.serial, img_data.id, img_data.name, tt, "bait_" + str(ttt), bait_contact[tt][ttt] ])
                else:
                    df_bait.append([img_data.serial, img_data.id, img_data.name, tt, "virtual_" + str(ttt-1), bait_contact[tt][ttt] ])

        # image
        if img is not None:
            for i in range(len(bait_positions)):
                img_copy = img.copy()
                for ii in range(len(bait_positions[i])):
                    if bait_contact[i][ii] == 1:
                        cv2.circle(img_copy, bait_positions[i][ii], int(bait_size/2), v_col[4], object_size)
                    else:
                        cv2.circle(img_copy, bait_positions[i][ii], int(bait_size/2), v_col[0], object_size)
                cv2.imwrite(out_dir + os.sep + 'bait' + os.sep + "bait_" + img_data.id + "_" + str(img_data.serial) + "_" + str(i) + ".jpg", img_copy)
        # endregion ------


        # region --- 2. Covering angle analysis ---#
        for i in range(len(concentric_circles)):
            circle_radius = int(concentric_circles[i]/scale_object_len*scale)

            circle_intersections = []
            for tt_t in range(len_t):
                ll = len(tunnel[tt_t])
                for ttt in range(ll - 1):
                    tunnel_segment = tunnel[tt_t][ttt:(ttt + 2)]
                    circle_intersection_point = check_intersection(tunnel_segment, exp_meta_data.initial, circle_radius)
                    if circle_intersection_point is not None:
                        circle_intersections.append(circle_intersection_point)

            # output data
            for tt in range(len(circle_intersections)):
                intersection_vec = circle_intersections[tt].astype(int) - exp_meta_data.initial
                angle_rad = math.atan2(intersection_vec[1], intersection_vec[0])
                df_angle.append([img_data.serial, img_data.id, img_data.name, concentric_circles[i], angle_rad])

            # image
            if img is not None:
                img_copy = img.copy()
                cv2.circle(img_copy, exp_meta_data.initial, 1, v_col[2], object_size)
                cv2.circle(img_copy, exp_meta_data.initial, circle_radius, v_col[0], object_size)
                for ii in range(len(circle_intersections)):
                    cv2.circle(img_copy, circle_intersections[ii].astype(int), 1, v_col[4], object_size)
                    cv2.line(img_copy, exp_meta_data.initial, circle_intersections[ii].astype(int), v_col[4], object_size)
                cv2.imwrite(out_dir + os.sep + 'angle' + os.sep + "angle_r" + str(concentric_circles[i]) + "_" + img_data.id + "_" + str(img_data.serial) + ".jpg", img_copy)
        # endregion ------

    f = open(out_dir + os.sep + "bait" + os.sep + "df_bait.csv", 'w', newline='')
    writer = csv.writer(f)
    writer.writerows(df_bait)
    f.close()

    f = open(out_dir + os.sep + "angle" + os.sep + "df_angle.csv", 'w', newline='')
    writer = csv.writer(f)
    writer.writerows(df_angle)
    f.close()

    return "Bait-analysis finished"

def bait_gui():

    # region -- Appearence --
    sg.theme('LightBrown2')
    frame_file = sg.Frame('Files', [
        [sg.Text("In   "),
         sg.InputText('Input folder', enable_events=True, size=(20, 1)),
         sg.FolderBrowse(button_text='select', size=(6, 1), key="-IN_FOLDER_NAME-")
         ],
        [sg.Text("Out"),
         sg.InputText('Output folder', enable_events=True, size=(20, 1)),
         sg.FolderBrowse(button_text='select', size=(6, 1), key="-OUT_FOLDER_NAME-")
         ],
        [sg.Text("File extension (default = jpg)"),
         sg.In(key='-FILE_EXTENSION-', size=(15, 1))]
    ], size=(550, 150))

    frame_param = sg.Frame('Parameters', [
        [sg.Text("Post-analysis:", size=(12,1)), 
         sg.Text("scale length (mm)", size=(15,1)),
         sg.In(key='-SCALE_OBJECT-', size=(6, 1)),
         sg.Text("Num of virtual baits (def = 7)"),
         sg.In(key='-NUM_VIRTUAL_BAITS-', size=(6, 1))
         ],
        [sg.Text("", size=(12,1)),
         sg.Text("Circles for angle calculation (mm) def = 300,500"),
         sg.In(key='-Circle_Radius-', size=(6, 1))
        ],
        [sg.Text("Drawing:", size=(12,1)),
         sg.Text("output image", size=(12,1)),
         sg.Combo(['true', 'false'], default_value="true", size=(6, 1), key="-OUTPUT_IMAGE-")
         ],
        [sg.Text("", size=(12,1)),
         sg.Text("line width (def 5)"),
         sg.In(key='-LINE_WIDTH-', size=(6, 1)),

         sg.Text("font size (def 2)"),
         sg.In(key='-FONT_SIZE-', size=(6, 1))
        ]
    ], size=(750, 160))

    frame_buttons = sg.Frame('', [
        [sg.Submit(button_text='Bait analysis start', size=(20, 3), key='bait_analysis_start')]], size=(180, 100))
    
    frame_manual = sg.Frame('Manual', [
        [sg.Text("Images should be named in 'id_number.jpg'\n"
                 "    e.g., TunnelA_00.jpg, TunnelA_01.jpg, ..., TunnelA_20.jpg, TunnelB_00.jpg, TunnelB_01.jpg, ...")],
        [sg.Text("Obtain meta data", size=(15,1))],
        [sg.Text("", size=(1,3)),
         sg.Text("1. Initial point", size=(15,1)),
         sg.Text("-LC: the tunnel starting point (= center of the circule to obtain angles)")],
        [sg.Text("", size=(1,2)),
         sg.Text("2. Baits", size=(15,2)),
         sg.Text("-Drag to draw circle baits\n"
                 "-RC to go next bait. After 2 baits finished, RC to exit")]
        ], size=(750, 250))
    
    layout = [[frame_file, frame_buttons], [frame_param], [frame_manual]]
    
    window = sg.Window('TManual additional program for bait analysis',
                       layout, resizable=True)
    # endregion ------

    while True:
        event, values = window.read()
    
        if event is None:
            print('exit')
            break
        else:
            if event == 'bait_analysis_start':

                # file info
                if len(values["-IN_FOLDER_NAME-"]) == 0:
                    print("no input!")
                    continue

                else:
                    in_dir = values["-IN_FOLDER_NAME-"]+os.sep
                    if len(values["-OUT_FOLDER_NAME-"]) == 0:
                        out_dir = in_dir+ os.sep +"tmanual" + os.sep
                        if not os.path.exists(out_dir):
                            print("no output directly!")
                    else:
                        out_dir = values["-OUT_FOLDER_NAME-"]+os.sep
                
                # parameters
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

                if len(values["-NUM_VIRTUAL_BAITS-"]) == 0:
                    max_num_virtual_bait = 7
                else:
                    max_num_virtual_bait = int(values["-NUM_VIRTUAL_BAITS-"])

                if len(values["-Circle_Radius-"]) == 0:
                    concentric_circles = [300, 500]
                else:
                    s = values["-Circle_Radius-"]
                    l = [x.strip() for x in s.split(',')]
                    concentric_circles = [int(s) for s in l]

                if len(values["-FILE_EXTENSION-"]) == 0:
                    file_extension = "jpg"
                else:
                    file_extension = values["-FILE_EXTENSION-"]

                print("input dir: "+str(in_dir))
                print("output dir: "+out_dir)

                get_metadata(in_dir, out_dir, file_extension, object_size, font_size)

                message = bait_post_analysis(in_dir, out_dir, scale_object_len, max_num_virtual_bait, concentric_circles, object_size, font_size)
                if message is not None:
                    sg.popup(message)            

    window.close()

bait_gui()


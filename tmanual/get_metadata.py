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

note_pos = [40, 100]
font_size = 2
object_size = 2
#os.chdir("F:/Dropbox/research/papers_and_projects/2022/TManual/examples/Lee-etal-2008_test")
os.chdir("F:/Dropbox/research/papers_and_projects/2022/TManual/bait_project/example1")
v_col = [[37, 231, 253], [98, 201, 94], [140, 145, 33],  [139, 82, 59], [84, 1, 68]]  # viridis colors in BGR
in_files = 0
#in_dir = "F:/Dropbox/research/papers_and_projects/2022/TManual/examples/Lee-etal-2008_test"
in_dir = "F:/Dropbox/research/papers_and_projects/2022/TManual/bait_project/example1"
file_extension = "jpg"


class ExperimentMetaData:
    def __init__(self, experiment_id, img_name, data_values=None):
        if data_values is None:
            self.name = img_name
            self.id = experiment_id
            data_values = [self.name, self.id, np.array([0, 0]), [], [], 0]
        self.name = data_values[0]
        self.id = data_values[1]
        self.initial = data_values[2]
        self.arena = data_values[3]
        self.baits = data_values[4]
        self.analyze_flag = data_values[5]

    def output_meta_data(self):
        return [self.name, self.id, self.initial, self.arena, self.baits, self.analyze_flag]

    def note_plot(self, img, note_message, font_size):
        cv2.putText(img, note_message+'('+self.name+')',
                    note_pos, cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), font_size, cv2.LINE_AA)
        return img

    def image_output(self, img, in_dir, object_size, font_size):
        cv2.putText(img, self.id, note_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), font_size, cv2.LINE_AA)
        cv2.circle(img, self.initial, object_size * 5, v_col[0], object_size)
        cv2.circle(img, self.initial, object_size, (0, 0, 0), -1)
        cv2.rectangle(img, self.arena[0], self.arena[1], (0, 0, 255), thickness=object_size)
        for i in range(len(self.baits)):
            cv2.circle(img, self.baits[i][0], radius=int(self.baits[i][1] / 2),
                       color=(0, 0, 255), thickness=object_size)
        cv2.imwrite(in_dir + os.sep + 'meta' + os.sep + self.id + ".jpg", img)

    def analyze_done(self):
        self.analyze_flag = 1


def get_metadata():
    # Data read
    if os.path.exists(in_dir + os.sep + "tmanual" + os.sep + "bait" + os.sep + "meta.pickle"):
        with open(in_dir + os.sep + "tmanual" + os.sep + "bait" + os.sep + "meta.pickle", mode='rb') as f:
            exp_meta_output = pickle.load(f)
    else:
        print("new analysis start")
        if not os.path.exists(in_dir + os.sep + "tmanual" + os.sep + "bait"):
            os.makedirs(in_dir + os.sep + "tmanual" + os.sep + "bait")
        exp_meta_output = [[], []]  # store Ids, Results

    if in_files == 0:
        name1 = glob.glob(in_dir + os.sep + '*.' + file_extension)
    else:
        name1 = in_files.split(';')
    num_file = len(name1)
    print(name1)
    ii = 0
    while ii < num_file:

        # region --- 0. Load data ---#
        image_name = os.path.basename(name1[ii])
        experiment_id = image_name.split('_')[0]

        if experiment_id in exp_meta_output[0]:
            ii = ii + 1
            continue

        img_read = cv2.imread(image_name)
        if img_read is None:
            print("Error. file is not readable: " + image_name + ". Skip.")
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
        # endregion

        #"""
        # region --- 2. Arena size ---#
        img = exp_meta_data.note_plot(img_read.copy(), '2.Arena size', font_size)
        cv2.circle(img, exp_meta_data.initial, object_size * 5, v_col[0], object_size)
        cv2.circle(img, exp_meta_data.initial, object_size, (0, 0, 0), -1)
        cv2.imshow(window_name, img)

        def arena_size(event, x, y, flags, param):
            nonlocal x0, y0, x1, y1, drawing
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                x0, y0 = x, y
            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    cv2.rectangle(img_copy, (x0, y0), (x, y), (0, 0, 255), thickness=2)
            elif event == cv2.EVENT_LBUTTONUP:
                x1, y1 = x, y
                cv2.rectangle(img_copy, (x0, y0), (x1, y1), (0, 0, 255), thickness=2)
                drawing = False
            elif event == cv2.EVENT_RBUTTONDOWN:
                if x0 > x1:
                    x0, x1 = x1, x0
                if y0 > y1:
                    y0, y1 = y1, y0
                press('f')
            press('enter')

        x0, y0, x1, y1 = 0, 0, 0, 0
        img_copy = img.copy()
        drawing = False
        while True:
            cv2.imshow('window', img_copy)
            if drawing:
                img_copy = img.copy()
            cv2.setMouseCallback('window', arena_size)
            k = cv2.waitKey(0)
            if k == ord("f") or k == 27:
                break
        if k == 27:
            break
        exp_meta_data.arena = [np.array([x0, y0]), np.array([x1, y1])]
        # endregion ------
        #"""

        # region --- 3. Bait locate ---#
        img = exp_meta_data.note_plot(img_read.copy(), '3. Bait', font_size)
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
        # endregion

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
        with open(in_dir + os.sep + 'tmanual' + os.sep + 'bait' + os.sep + 'meta.pickle', mode='wb') as f:
            pickle.dump(exp_meta_output, f)

        ii = ii + 1
        # endregion
    cv2.destroyAllWindows()


def bait_post_analysis():
    scale_object_len = 10
    # Data read
    if os.path.exists(in_dir + os.sep + "tmanual" + os.sep + "res.pickle"):
        with open(in_dir + os.sep + "tmanual" + os.sep + "res.pickle", mode='rb') as f:
            tmanual_output = pickle.load(f)
    else:
        return "no res.pickle file in " + in_dir + "tmanual"

    if os.path.exists(in_dir + os.sep + 'tmanual' + os.sep + 'bait' + os.sep + 'meta.pickle'):
        with open(in_dir + os.sep + 'tmanual' + os.sep + 'bait' + os.sep + 'meta.pickle', mode='rb') as f:
            exp_meta_output = pickle.load(f)
    else:
        return "no meta.pickle file in " + in_dir + 'tmanual' + os.sep + 'bait'

    for i_df in tqdm(range(len(tmanual_output[0]))):
    #for i_df in tqdm(range(1)):
        img_data = tmanual.ImgData(None, tmanual_output[2][i_df])

        exp_meta_data = ExperimentMetaData(None, None, exp_meta_output[1][exp_meta_output[0].index(img_data.id)])

        tunnel_len, scale = img_data.measure_tunnel_length(scale_object_len)

        tunnel = img_data.tunnel
        node = img_data.obtain_nodes()

        len_t = len(tunnel_len)

        # Virtual bait encounter
        bait_size = int((exp_meta_data.baits[0][1] + exp_meta_data.baits[1][1]) / 2)
        bait1 = exp_meta_data.baits[0][0]
        bait2 = exp_meta_data.baits[1][0]
        if bait1[0] > bait2[0]:
            bait1, bait2 = bait2, bait1

        bait_positions = [[bait1, bait2]]
        for i in range(1, 8, 1):
            bait_temp = [bait1]
            for ii in range(i+1):
                bait_temp_temp = bait1 + (bait2 - bait1) / (i+1) * (ii+1)
                bait_temp_temp = bait_temp_temp.astype(int)
                bait_temp.append(bait_temp_temp)
            bait_positions.append(bait_temp)

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

        #  1. for each tunnel, check if contact with bait
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

        print(bait_contact)

        # image
        img = cv2.imread(in_dir + os.sep + img_data.name)
        print(in_dir + os.sep + img_data.name)
        for i in range(len(bait_positions)):
            img_copy = img.copy()
            for ii in range(len(bait_positions[i])):
                if bait_contact[i][ii] == 1:
                    cv2.circle(img_copy, bait_positions[i][ii], int(bait_size/2), v_col[4], object_size)
                else:
                    cv2.circle(img_copy, bait_positions[i][ii], int(bait_size/2), v_col[0], object_size)
            cv2.imwrite(in_dir + os.sep + 'tmanual' + os.sep + 'bait' + os.sep + "bait_" + img_data.id + "_" + str(img_data.serial) + "_" + str(i) + ".jpg", img_copy)


        ## Angle

        def check_intersection(A, B, P, dis):
            """
            Returns a coordinate C = (x, y) on the line AB with which the distance between point P and C equals dis.
            Returns None if such C does not exist.
            """
            # Calculate the direction vector of line AB
            AB = B - A
            BP_dis = norm(B-P)
            AP_dis = norm(A-P)

            nearest_dis, nearest_point = node_tunnel_distance(P, [A, B])
            if nearest_dis > dis:
                return None
            if BP_dis < dis and AP_dis < dis:
                return None
            
            # Calculate the distance from point P to line AB
            unit_vec_ab = AB/norm(AB)
            nearest_ab_point = A + unit_vec_ab*(np.dot(P-A, AB)/norm(AB))
            distance_to_line = norm(P-nearest_ab_point)
            
            # If the distance from point P to line AB is greater than dis,
            # then there is no point on the line AB that is dis away from point P
            if distance_to_line > dis:
                return None
            
            # Calculate the distance along line AB where point C is located
            distance_along_line = np.sqrt(dis**2 - distance_to_line**2)
            
            # Calculate the unit vector in the direction of line AB
            
            # Calculate the coordinates of point C
            C = nearest_ab_point + distance_along_line * unit_vec_ab
            
            return C


        #  2. 
        concentric_circles = 300

        circle_intersections = []
        for tt_t in range(len_t):
            ll = len(tunnel[tt_t])
            for ttt in range(ll - 1):
                tunnel_segment = tunnel[tt_t][ttt:(ttt + 2)]
                circle_intersection_point = check_intersection(tunnel_segment[0], tunnel_segment[1], exp_meta_data.initial, concentric_circles)
                if circle_intersection_point is not None:
                    circle_intersections.append(circle_intersection_point)

        img = cv2.imread(in_dir + os.sep + img_data.name)
        img_copy = img.copy()
        cv2.circle(img_copy, exp_meta_data.initial, 1, v_col[2], object_size)
        cv2.circle(img_copy, exp_meta_data.initial, concentric_circles, v_col[0], object_size)
        for ii in range(len(circle_intersections)):
            print(norm(circle_intersections[ii]-exp_meta_data.initial))
            cv2.circle(img_copy, circle_intersections[ii].astype(int), 1, v_col[4], object_size)
        cv2.imwrite(in_dir + os.sep + 'tmanual' + os.sep + 'angle' + os.sep + "angle_" + img_data.id + "_" + str(img_data.serial) + ".jpg", img_copy)


get_metadata()
bait_post_analysis()

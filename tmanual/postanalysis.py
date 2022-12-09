"""
Post-analysis of the outcome of measurement
Obtain length of Primary, Secondary, Tertiary, ..., tunnels
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
from tqdm import tqdm
from tmanual.image  import tunnel_draw, outlined_text, object_drawing, image_format, ImgData
vcol = [[37, 231, 253], [98, 201, 94], [140, 145, 33],  [139, 82, 59], [84, 1, 68]]  # viridis colors in BGR

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

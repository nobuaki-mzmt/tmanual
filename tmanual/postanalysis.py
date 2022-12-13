"""
Post-analysis of the outcome of measurement
Obtain length of Primary, Secondary, Tertiary, ..., tunnels
"""

import os
import pickle
import cv2
import copy
import numpy as np
from numpy.linalg import norm
import glob
import re
import csv
import math
from tqdm import tqdm
from tmanual.image  import tunnel_draw, outlined_text, object_drawing, image_format, ImgData
vcol = [[37, 231, 253], [98, 201, 94], [140, 145, 33],  [139, 82, 59], [84, 1, 68]]  # viridis colors in BGR

def postanalysis(in_dir, out_dir, scale_object_len, contact_threshold, network_out, output_image, object_size, font_size, text_drawing):
    def node_tunnel_distance(node_p, t_seg):
        # calculate the distance between line AB and point P
        # also obtain nearest point on line AB
        # using inner product of vector
        ap = node_p - t_seg[0]
        bp = node_p - t_seg[1]
        ab = t_seg[1] - t_seg[0]
        if np.dot(ab, ap) < 0:
            # a is the nearest
            nt_distance = norm(ap)
            nearest_ab_point = t_seg[0]
        elif np.dot(ab, bp) > 0:
            # b is the nearest
            nt_distance = norm(bp)
            nearest_ab_point = t_seg[1]
        else:
            # first obtain the nearest point on ab
            unit_vec_ab = ab/norm(ab)
            nearest_ab_point = t_seg[0] + unit_vec_ab * (np.dot(ap, ab)/norm(ab))
            nt_distance = norm(node_p-nearest_ab_point)
        return(nt_distance, nearest_ab_point)


    # Data read
    if os.path.exists(out_dir + "/res.pickle"):
        with open(out_dir + '/res.pickle', mode='rb') as f:
            tmanual_output = pickle.load(f)

            # --- todo this part will be removed future
            # This part removes node object from old version tmanual res.picke
            if len(tmanual_output[2][0]) > 7:
                for ii in range(len(tmanual_output[0])):
                    tmanual_output[2][0].pop(5)
                with open(out_dir + '/res.pickle', mode='wb') as f:
                    pickle.dump(tmanual_output, f)
            # ----------

    else:
        return "no res.pickle file in " + out_dir

    df_tunnel = [["serial", "id", "name", "tunnel_length", "tunnel_sequence"]]
    df_summary = [['serial', 'id', 'name', 'tunnel_length_total', 'tunnel_length_1st', 'tunnel_length_2nd',
                  'tunnel_length_3rd', 'tunnel_length_4more', 'tunnel_num_total', 'tunnel_num_1st',
                  'tunnel_num_2nd', 'tunnel_num_3rd', 'tunnel_num_4more', 'node_num']]
    if network_out:
        df_net = [["serial", "id", "name", "edge_from", "edge_to", "edge_len"]]

    for i_df in tqdm(range(len(tmanual_output[0]))):
        img_data = ImgData(None, tmanual_output[2][i_df])
        tunnel_len, scale = img_data.measure_tunnel_length(scale_object_len)

        tunnel = img_data.tunnel
        node = img_data.obtain_nodes()
        
        len_t = len(tunnel_len) 

        tunnel_sequence = [1] * len_t  # primary, secondary, terially, ...
        contact_tunnelID = [ [-1] * len_t, [-1] * len_t ]  # tunnnel id that tunnel starts from and ends at (-1 for primary or -1 for active tunnels)
        node_nearest_point = [ [np.array([0,0])] * len_t, [np.array([0,0])] * len_t]

        net_edge_from = []
        net_edge_to   = []
        net_edge_len  = []
        
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

            # 3. determine Secondary, Tertiary, ..., tunnel
            tunnel_seq_count = 1
            while True:
                check_tunnel = [i for i, x in enumerate(tunnel_sequence) if x == tunnel_seq_count]
                for tt in range(len_t):
                    if contact_tunnelID[0][tt] in check_tunnel:
                        tunnel_sequence[tt] = tunnel_seq_count + 1
                tunnel_seq_count = tunnel_seq_count + 1
                if len(check_tunnel) == 0:
                    if min(tunnel_sequence) < 0:
                        return "Error in " + img_data.name + ": cannot get tunnel id. As this error is unexpected. please contact author" 
                    break

            # 4. get network structure
            if network_out:
                node_name = [[0]*len_t, [0]*len_t]
                for tt in range(len_t):
                    node_name[0][tt] = "no"+str(tt).zfill(3)+"_0"
                    node_name[1][tt] = "no"+str(tt).zfill(3)+"_1"

                # start node is the same?
                for tt_0 in range(len_t):
                    for tt_1 in range(len_t):
                        if tt_0 < tt_1:
                            if norm(node[0][tt_0] - node[0][tt_1]) < contact_threshold:
                                node_name[0][tt_1] = copy.copy(node_name[0][tt_0])
                # end node is the same?
                for tt_0 in range(len_t):
                    for tt_1 in range(len_t):
                        if tt_0 < tt_1:
                            if norm(node[1][tt_0] - node[1][tt_1]) < contact_threshold:
                                node_name[1][tt_1] = copy.copy(node_name[1][tt_0])

                # start-end node is the same?
                for tt_0 in range(len_t):
                    for tt_1 in range(len_t):
                        if norm(node[0][tt_0] - node[1][tt_1]) < contact_threshold:
                            node_name[1][tt_1] = copy.copy(node_name[0][tt_0])

                tunnel_seq_count = 1

                while True:
                    check_tunnel = [i for i, x in enumerate(tunnel_sequence) if x == tunnel_seq_count]
                    for tt in check_tunnel:
                        ll = len(tunnel[tt])
                        list_start_node_on_tunnel = [i for i, x in enumerate(contact_tunnelID[0]) if x == tt]
                        list_end_node_on_tunnel   = [i for i, x in enumerate(contact_tunnelID[1]) if x == tt]
                        list_node_on_tunnel = list_start_node_on_tunnel + list_end_node_on_tunnel

                        list_start_or_end = [0]*len(list_start_node_on_tunnel) + [1]*len(list_end_node_on_tunnel)

                        list_tunnel_seg_len = np.array([0]*len(list_node_on_tunnel))

                        for nn in range(len(list_node_on_tunnel)):
                            tunnel_seg_len = 0
                            nearest_point = node_nearest_point[list_start_or_end[nn]][list_node_on_tunnel[nn]]

                            for ttt in range(ll-1):
                                tunnel_segment = tunnel[tt][ttt:(ttt + 2)]
                                dis_temp = node_tunnel_distance(nearest_point, tunnel_segment)[0]
                                if dis_temp < 0.00001:
                                    tunnel_seg_len = tunnel_seg_len + norm(nearest_point-tunnel_segment[0])
                                    break                                
                                else:
                                    tunnel_seg_len = tunnel_seg_len + norm(tunnel_segment[1]-tunnel_segment[0])
                            list_tunnel_seg_len[nn] = tunnel_seg_len
                        
                        list_tunnel_seg_len = list_tunnel_seg_len/scale*scale_object_len

                        net_edge_from.append(node_name[0][tt])

                        if len(list_node_on_tunnel) > 0:

                            tunnel_seg_len_order = np.argsort(list_tunnel_seg_len)

                            for nn in range(len(list_node_on_tunnel)):
                                node_temp = tunnel_seg_len_order[nn]
                                node_pos_temp = node_nearest_point[list_start_or_end[node_temp]][list_node_on_tunnel[node_temp]]
                                net_edge_to.append( node_name[list_start_or_end[node_temp]][list_node_on_tunnel[node_temp]] )
                                if nn > 0:
                                    net_edge_len.append(list_tunnel_seg_len[tunnel_seg_len_order[nn]] - list_tunnel_seg_len[tunnel_seg_len_order[nn-1]])
                                else:
                                    net_edge_len.append(list_tunnel_seg_len[tunnel_seg_len_order[nn]])

                                net_edge_from.append( node_name[list_start_or_end[node_temp]][list_node_on_tunnel[node_temp]] )

                            net_edge_to.append( node_name[1][tt] )
                            net_edge_len.append(tunnel_len[tt] - list_tunnel_seg_len[tunnel_seg_len_order[len(list_node_on_tunnel)-1]])

                        else:
                            net_edge_to.append( node_name[1][tt] )
                            net_edge_len.append(tunnel_len[tt])

                    tunnel_seq_count = tunnel_seq_count + 1
                    if len(check_tunnel) == 0:
                        break

                for tt in range(len_t):
                    if net_edge_from[tt] == net_edge_to[tt]:
                        net_edge_from.pop(tt)
                        net_edge_to.pop(tt)
                        net_edge_len.pop(tt)


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
                     tunnel_sequence.count(4), "tentative"] #len(node)
        df_summary.append(df_append)

        # image output
        # print("tunnel_sequence    : ", tunnel_sequence)
        
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
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

def postanalysis(in_dir, out_dir, scale_object_len, contact_threshold, output_image, object_size, font_size, text_drawing):
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
    else:
        return "no res.pickle file in " + out_dir

    df_tunnel = [["serial", "id", "name", "tunnel_length", "tunnel_sequence"]]
    df_summary = [['serial', 'id', 'name', 'tunnel_length_total', 'tunnel_length_1st', 'tunnel_length_2nd',
                  'tunnel_length_3rd', 'tunnel_length_4more', 'tunnel_num_total', 'tunnel_num_1st',
                  'tunnel_num_2nd', 'tunnel_num_3rd', 'tunnel_num_4more', 'node_num']]
    df_net = [["serial", "id", "name", "edge_from", "edge_to", "edge_len"]]
    for i_df in tqdm(range(len(tmanual_output[0]))):
        img_data = ImgData(None, tmanual_output[2][i_df])
        tunnel_len, scale = img_data.measure_tunnel_length(scale_object_len)

        tunnel = img_data.tunnel
        node = img_data.node
        
        tunnel_sequence = [1] * len(tunnel_len)  # primary, secondary, terially, ...
        tunnel_start_nodeID = [-1] * len(tunnel_len)  # node id tunnel starts from. 0 for primary
        tunnel_end_nodeID = [-1] * len(tunnel_len)
        node_on_tunnelID = [-1] * len(node)  # tunnel id that node exists on
        node_on_tunnel_nearest_point = [np.array([0,0])] * len(node)

        net_node = copy.copy(node)
        node_type = ["n"] * len(node)
        node_id = list(range(len(node)))

        if len(node) > 0:
            #  1. for each tunnel, check from which node starts from 
            #  Determine Primary tunnel (= not start from nodes: >"contact_threshold" pixels)
            count = 0
            for tt in range(len(tunnel_len)):
                node_tunnel0_dis = np.sqrt(np.sum((node - tunnel[tt][0]) ** 2, axis=1))
                min_dis = min(node_tunnel0_dis)
                #print(tt, node_tunnel0_dis)
                if min_dis < contact_threshold:
                    tunnel_sequence[tt] = -1
                    starting_node = np.where(node_tunnel0_dis == min_dis)[0]
                    tunnel_start_nodeID[tt] = starting_node[0]
                else:
                    tunnel_start_nodeID[tt] = -1  # primary
                    net_node.append(tunnel[tt][0])
                    node_type.append("t0")
                    node_id.append(count)
                    count = count + 1

            # 3. for each tunnel, check at which node tunnel ends
            count = 0
            for tt in range(len(tunnel_len)):
                node_tunnel_dis = np.sqrt(np.sum((node - tunnel[tt][len(tunnel[tt])-1]) ** 2, axis=1))
                min_dis = min(node_tunnel_dis)
                if min_dis < contact_threshold:
                    ending_node = np.where(node_tunnel_dis == min_dis)[0]
                    tunnel_end_nodeID[tt] = ending_node[0]
                else:
                    tunnel_end_nodeID[tt] = -1  # end tip
                    net_node.append(tunnel[tt][len(tunnel[tt])-1])
                    node_type.append("te")
                    node_id.append(count)
                    count = count + 1

            #  2. for each node with tunnel start, check on which tunnel the node exists
            for nn in range(len(node)):
                min_dis = 99999
                nearest_point = np.array([0,0])
                for tt in range(len(tunnel_len)):
                    if tunnel_start_nodeID[tt] != nn and tunnel_end_nodeID[tt] != nn:  # exclude the tunnel that starts from the node
                        ll = len(tunnel[tt])
                        for ttt in range(ll - 1):
                            tunnel_segment = tunnel[tt][ttt:(ttt + 2)]
                            dis_temp, nearest_point_temp = node_tunnel_distance(node[nn], tunnel_segment)
                            if(dis_temp < min_dis):
                                min_dis = dis_temp
                                nearest_point = nearest_point_temp
                                node_on_tunnel = tt
                node_on_tunnelID[nn] = node_on_tunnel
                node_on_tunnel_nearest_point[nn] = nearest_point
                if min_dis > contact_threshold:
                    print( "Warning in " + img_data.name + ". Node: " + str(nn) + " is not on tunnel lines" )
                    print(min_dis, node_on_tunnel)

            

            # 3. determine Secondary, Tertiary, ..., tunnel
            #print("tunnel_sequence    : ", tunnel_sequence)
            #print("tunnel_start_nodeID: ", tunnel_start_nodeID)
            #print("node_on_tunnelID   : ", node_on_tunnelID)
            end, tunnel_seq_count = 0, 1
            while end == 0:
                for tt in range(len(tunnel_len)):
                    tunnel_start_node = tunnel_start_nodeID[tt]
                    if tunnel_start_node >= 0:
                        stem_tunnel = node_on_tunnelID[tunnel_start_node]
                        if tunnel_sequence[stem_tunnel] == tunnel_seq_count:
                            tunnel_sequence[tt] = tunnel_seq_count + 1
                tunnel_seq_count = tunnel_seq_count + 1
                if min(tunnel_sequence) > 0:
                    end = 1
                if tunnel_seq_count > 1000:
                    print("tunnel_sequence    : ", tunnel_sequence)
                    return "Error in " + img_data.name + ": Invalid nodes. Recheck if all nodes are on the tunnel lines"

            #list_tunnel_seg_len = [-1] * len(node)

            # get network structure
            net_edge_from = []
            net_edge_to   = []
            net_edge_len  = []
            count = 1
            while True:
                check_tunnel = [i for i, x in enumerate(tunnel_sequence) if x == count]
                for tt in check_tunnel:
                    ll = len(tunnel[tt])
                    list_node_on_tunnel = [i for i, x in enumerate(node_on_tunnelID) if x == tt]
                    list_tunnel_seg_len = np.array([0]*len(list_node_on_tunnel))
                    for nn in range(len(list_node_on_tunnel)):
                        tunnel_seg_len = 0
                        for ttt in range(ll-1):
                            tunnel_segment = tunnel[tt][ttt:(ttt + 2)]
                            dis_temp = node_tunnel_distance(node_on_tunnel_nearest_point[list_node_on_tunnel[nn]], tunnel_segment)[0]
                            if dis_temp < 0.0001:
                                tunnel_seg_len = tunnel_seg_len + norm(node_on_tunnel_nearest_point[list_node_on_tunnel[nn]]-tunnel_segment[0])
                                break
                            else:
                                tunnel_seg_len = tunnel_seg_len + norm(tunnel_segment[1]-tunnel_segment[0])
                        list_tunnel_seg_len[nn] = tunnel_seg_len
                    
                    list_tunnel_seg_len = list_tunnel_seg_len/scale*scale_object_len

                    if tunnel_start_nodeID[tt] == -1:
                        net_edge_from.append("t0")
                    else:
                        net_edge_from.append("n-" + str(tunnel_start_nodeID[tt]) )

                    if len(list_node_on_tunnel) > 0:
                        tunnel_seg_len_order = np.argsort(list_tunnel_seg_len)
                        net_edge_to.append("n-" + str(list_node_on_tunnel[tunnel_seg_len_order[0]]))
                        net_edge_len.append(list_tunnel_seg_len[tunnel_seg_len_order[0]])
                        net_edge_from.append("n-" + str(list_node_on_tunnel[tunnel_seg_len_order[0]]) )
                        if len(list_node_on_tunnel) > 1:
                            for nn in range(len(list_node_on_tunnel)-1):
                                net_edge_to.append("n-" + str(list_node_on_tunnel[tunnel_seg_len_order[nn+1]]))
                                net_edge_len.append(list_tunnel_seg_len[tunnel_seg_len_order[nn+1]]-list_tunnel_seg_len[tunnel_seg_len_order[nn]])
                                net_edge_from.append("n-" + str(list_node_on_tunnel[tunnel_seg_len_order[nn+1]]) )
                        if tunnel_end_nodeID[tt] < 0:
                            net_edge_to.append("te-" + str(tt))
                        else:
                            net_edge_to.append("n-" + str(tunnel_end_nodeID[tt]))
                        net_edge_len.append(tunnel_len[tt] - list_tunnel_seg_len[tunnel_seg_len_order[len(list_node_on_tunnel)-1]])
                    else:
                        if tunnel_end_nodeID[tt] < 0:
                            net_edge_to.append("te-" + str(tt))
                        else:
                            net_edge_to.append("n-" + str(tunnel_end_nodeID[tt]))
                        net_edge_len.append(tunnel_len[tt])
                count = count + 1
                if len(check_tunnel) == 0:
                    break
            print("net_edge_from", net_edge_from)
            print("net_edge_to  ", net_edge_to)
            print("net_edge_len ", net_edge_len)

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
                     tunnel_sequence.count(4), len(node)]
        df_summary.append(df_append)

        # image output
        print("tunnel_sequence    : ", tunnel_sequence)
        
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

    f = open(out_dir+'df_net.csv', 'w', newline='')
    writer = csv.writer(f)
    writer.writerows(df_net)
    f.close()

    return "Post-analysis finished"

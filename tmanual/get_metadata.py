"""
Obtain meta data for bait experiments with a 3m-arena by kaitlin
 ----TODO will be added to main as a function to get meta data of the experiments? maybe
"""

import os
import pickle
import cv2
import numpy as np
import pyautogui as pag

# 1. 
def get_metadata():

	# Data read
    if os.path.exists(out_dir + "/res.pickle"):
        with open(out_dir+ os.sep  + 'res.pickle', mode='rb') as f:
            tmanual_output = pickle.load(f)

    unique_id = list(set(tmanual_output[0]))

    ii = 0
    while ii < len(unique_id):

    	# region --- 0. Load data ---#
    	indice_unique_id = [i for i, x in enumerate(tmanual_output[0]) if x == unique_id[0]]
    	serial_list      = [tmanual_output[1][i] for i in indice_unique_id]
		index_base_image = indice_unique_id[serial_list.index(min(serial_list))]

		unique_id        = tmanual_output[2][index_base_image][1]
		image_name       = tmanual_output[2][index_base_image][0]
		img_read = cv2.imread(image_name)

        if img_read is None:
            print("Error. file is not readable: " + image_name + ". Skip.")
            ii = ii + 1
            continue

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

        # region --- 1. Arena size ---#
        img = cv2.putText(img_read.copy(), '1.Arena size '+'('+unique_id+')',
                    note_pos, cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), font_size, cv2.LINE_AA)
		cv2.imshow(window_name, img)


        def arena_size(event, x, y, flags, param):
        	nonlocal sx0, sy0, x0, y0, x1, y1, scale, drawing, end
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                sx0, sy0 = x, y
            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    if abs(x-sx0) > abs(y-sy0):
                        cv2.rectangle(img_copy, (sx0, sy0), (sx0+y-sy0, y), (0, 0, 255), thickness = 1)
                    else:
                        cv2.rectangle(img_copy, (sx0, sy0), (x, sy0+x-sx0), (0, 0, 255), thickness = 1)
            elif event == cv2.EVENT_LBUTTONUP:
                if abs(x-sx0) > abs(y-sy0):
                    sx1 = sx0+y-sy0
                    sy1 = y
                else:
                    sx1 = x
                    sy1 = sy0+x-sx0
                cv2.rectangle(img_copy, (sx0, sy0), (sx1, sy1), (0, 0, 255), thickness = 1)
                drawing = False
            elif event == cv2.EVENT_RBUTTONDOWN:
                x0 = sx0
                y0 = sy0
                x1 = sx1
                y1 = sy1
                end = 1
            press('enter')





            
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

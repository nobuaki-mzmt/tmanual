"""
GUI application for tmanual.
GUI collects parameters for analysis and run either
   1. measurement.py
   2. postanalysys.py
"""


import PySimpleGUI as sg
import os

from tmanual import measurement
from tmanual import postanalysis

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
        ], size=(1000, 500))
    
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

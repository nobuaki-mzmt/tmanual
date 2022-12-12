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
                 "   LC (or V):analyze  RC (or N):skip  Esc:exit (saved)\n  "
                 "   B:previous image  R:re-analyze (append to previous)  A:re-analyze (from the scratch)")],
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

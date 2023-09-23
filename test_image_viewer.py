from utils.qt_imports import QtWidgets
import argparse
import sys
import json

import sys, importlib
from pathlib import Path

def import_parents(current_file, level=1):
    global __package__
    file = Path(current_file).resolve()
    parent, top = file.parent, file.parents[level]
    
    sys.path.append(str(top))
    try:
        sys.path.remove(str(parent))
    except ValueError: # already removed
        pass

    __package__ = '.'.join(parent.parts[len(top.parts):])
    importlib.import_module(__package__) # won't be needed after that

if __name__ == '__main__' and __package__ is None:
    import_parents(__file__)

from .tests_utils.event_recorder import EventRecorder
from .tests_utils.event_player   import EventPlayer
from .tests_utils.qtdump import *
from .image_viewers.qtImageViewer import qtImageViewer
from .image_viewers.glImageViewer import glImageViewer
from .image_viewers.ImageFilterParameters import ImageFilterParameters
from .image_viewers.ImageFilterParametersGui import ImageFilterParametersGui
from .utils.image_reader import image_reader


if __name__ == '__main__':
    # import numpy for generating random data points
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--input_image', help='input image')
    parser.add_argument('-p', '--play', help='events json file', default=None)
    parser.add_argument('-r', '--record', help='record events in given json file', default=None)
    parser.add_argument('--timing', action='store_true', help='display timings')
    parser.add_argument('--gl', action='store_true', help='use opengl viewer')
    args = parser.parse_args()
    _params = vars(args)
    print(f"record {_params['record']}")

    # define a Qt window with an OpenGL widget inside it
    # class TestWindow(QtGui.QMainWindow):
    class TestWindow(QtWidgets.QMainWindow):
        def __init__(self, events):
            super().__init__()

            record_file = _params['record']
            if record_file is not None:
                self.event_recorder = EventRecorder(filename=record_file)
                self.event_recorder.register_widget(id(self), "TestWindow")
            else:
                self.event_recorder = None

            self.main_widget = QtWidgets.QWidget()
            vertical_layout = QtWidgets.QVBoxLayout()
            self.main_widget.setLayout(vertical_layout)

            if _params['gl']:
                self.widget = glImageViewer(event_recorder = self.event_recorder)
            else:
                self.widget = qtImageViewer(event_recorder = self.event_recorder)
            if _params['timing']:
                self.widget.display_timing = True
            if record_file is not None:
                self.event_recorder.register_widget(id(self.widget), "widget")

            im = image_reader.read(_params['input_image'])
            self.widget.set_image(im)
            # put the window at the screen position (100, 100)
            self.setGeometry(0, 0, self.widget.width(), self.widget.height())
            self.setCentralWidget(self.main_widget)

            self.filter_params = ImageFilterParameters()
            self.filter_params_gui = ImageFilterParametersGui(self.filter_params, name="TestViewer")
            self.filter_params_gui.set_event_recorder(self.event_recorder)

            hor_layout = QtWidgets.QHBoxLayout()
            self.filter_params_gui.add_blackpoint(hor_layout, self.update_image_intensity_event)
            # white point adjustment
            self.filter_params_gui.add_whitepoint(hor_layout, self.update_image_intensity_event)
            # Gamma adjustment
            self.filter_params_gui.add_gamma(hor_layout, self.update_image_intensity_event)
            # G_R adjustment
            self.filter_params_gui.add_g_r(hor_layout, self.update_image_intensity_event)
            # G_B adjustment
            self.filter_params_gui.add_g_b(hor_layout, self.update_image_intensity_event)

            vertical_layout.addLayout(hor_layout)
            vertical_layout.addWidget(self.widget)

            self.show()

            if events != None:
                event_list = events['events']
                # self.event_recorder.register_widget(self.id(), "TestWindow")
                # self.event_recorder.register_widget(self.widget.id(), "widget")
                event_player = EventPlayer()
                event_player.register_widget("TestWindow", self)
                event_player.register_widget("widget", self.widget)
                self.filter_params_gui.register_event_player(event_player)
                event_player.play_events(event_list=event_list)

        def update_image_intensity_event(self):
            self.widget.filter_params.copy_from(self.filter_params)
            # print(f"parameters {self.filter_params}")
            self.widget.paintAll()

        def event(self, evt):
            if self.event_recorder is not None:
                if evt.type() in RESIZE_EVENTS:
                    self.event_recorder.store_event(self, evt)
            return QtWidgets.QMainWindow.event(self, evt)

        def closeEvent(self, event):
            # Save event list
            if self.event_recorder is not None:
                self.event_recorder.save_screen(self.widget)
                self.event_recorder.save_events()
            event.accept()

    # create the Qt App and window
    app = QtWidgets.QApplication(sys.argv)
    if _params['play'] is not None:
        with open(_params['play']) as json_file:
            events = json.load(json_file)
    else:
        events = None

    window = TestWindow(events)
    window.show()
    app.exec()

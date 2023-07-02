
from Qt import QtGui, QtCore, QtWidgets
from ..utils.utils import get_time
import json
from .qtdump import *

class EventRecorder:

    def __init__(self, filename=None):
        self.event_list = []
        self.event_start = get_time()
        self.filename = filename
        self.widgets = dict()
        self.verbose = True

    def register_widget(self, id, name):
        self.widgets[id] = name

    def store_event(self, widget, evt):
        # event_filter = MOUSE_EVENTS + RESIZE_EVENTS + WHEEL_EVENT
        if evt.type() in MOUSE_EVENTS and evt.spontaneous():
            evt_info = QtDump.mouseevent2dict(evt)
        else:
            if evt.type() in RESIZE_EVENTS and evt.spontaneous():
                evt_info = QtDump.resizeevent2dict(evt)
            else:
                if evt.type() in WHEEL_EVENT and evt.spontaneous():
                    evt_info = QtDump.wheelevent2dict(evt)
                else:
                    return
        evt_time = get_time() - self.event_start
        evt_info = {'info': evt_info, 'time': evt_time}
        if id(widget) in self.widgets:
            evt_info['widget_name'] = self.widgets[id(widget)]
        if self.verbose:
            print(f"recording {evt_info}")
        self.event_list.append(evt_info)

    def save_screen(self, widget):
        image_hash = QtDump.get_screen_hash(widget)
        print(f"image hash ${image_hash}")
        evt_time = get_time() - self.event_start
        evt_info = {'info': {'type':'save_screen', 'hash':image_hash}, 'time': evt_time}
        if id(widget) in self.widgets:
            evt_info['widget_name'] = self.widgets[id(widget)]
        self.event_list.append(evt_info)

    def save_events(self, filename=None):
        if filename is None:
            filename = self.filename
        if filename is not None:
            with open(filename, 'w') as fp:
                json.dump({'events': self.event_list}, fp)
        else:
            print("Error: EventRecorder.save_events() no filename given !")

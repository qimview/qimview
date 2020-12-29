from Qt import QtGui, QtCore, QtWidgets
from tests.qtdump import *

class EventPlayer:

    def __init__(self):
        self.widgets = dict()

    def register_widget(self, name, widget):
        self.widgets[name] = widget

    def play_events(self, event_list=None, start_delay=100):
        if len(event_list) > 0:
            # Send events using Qtimer
            print("EventPlayer: start")
            QtCore.QTimer.singleShot(start_delay, lambda: self.send_event( event_list, 0))

    def send_event(self, event_list, index):
        # print(f" {event_list[index]}")
        # now create mouse event
        new_event = QtDump.dict2event(event_list[index]['info'])
        widget_name = event_list[index]['widget_name']
        if widget_name in self.widgets:
            widget = self.widgets[widget_name]
            if new_event.type() in RESIZE_EVENTS:
                # for resize, we need to actually do the resize of the control widget instead of calling the resize event
                # QtWidgets.QApplication.postEvent(widget.parent(), new_event)
                widget.resize(new_event.size())
            else:
                QtWidgets.QApplication.postEvent(widget, new_event)
        else:
            print(f"EventPlayer, widget '{widget_name}' not found")
        index = index + 1
        if index < len(event_list):
            delay = event_list[index]['time']-event_list[index-1]['time']
            QtCore.QTimer.singleShot(delay*1000,  lambda: self.send_event(event_list, index))
        else:
            print("EventPlayer: end")

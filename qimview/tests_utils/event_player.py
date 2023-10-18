from ..utils.qt_imports import QtCore, QtWidgets
from .qtdump import *

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
        # now create mouse event
        current_event = event_list[index]
        event_info    = current_event['info']
        widget_name   = current_event.get('widget_name')
        # print(f"EventPlayer {index}: {widget_name} {event_info}", end=', ')
        if widget_name in self.widgets:
            widget = self.widgets[widget_name]
            if isinstance(event_info['type'],str):
                if event_info['type'] == 'save_screen':
                    widget_hash = QtDump.get_screen_hash(widget)
                    recorded_hash = event_info['hash']
                    # print(f"hash now {widget_hash} recorded {recorded_hash}")
                    if widget_hash != recorded_hash:
                        print('Hash comparison failure !!!')
                    else:
                        print('Hash comparison success !!!')
            else:
                # --- Deal with QEvent
                qt_event = QtDump.dict2event(event_info)
                if qt_event.type() in RESIZE_EVENTS:
                    # for resize, we need to actually do the resize of the control widget instead of calling the resize event
                    # QtWidgets.QApplication.postEvent(widget.parent(), new_event)
                    # print(" resize", end=', ')
                    widget.resize(qt_event.size())
                else:
                    # print(" post event", end=', ')
                    QtWidgets.QApplication.postEvent(widget, qt_event)
        else:
            print(f"widget '{widget_name}' not found", end=', ')
        index = index + 1
        if index < len(event_list):
            # print("")
            delay = event_list[index]['time']-event_list[index-1]['time']
            QtCore.QTimer.singleShot(delay*1000,  lambda: self.send_event(event_list, index))
        else:
            print(" end")


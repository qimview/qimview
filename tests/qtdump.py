
from Qt import QtGui, QtCore, QtWidgets

MOUSE_EVENTS = [
            # QtCore.QEvent.Type.KeyPress,
            QtCore.QEvent.Type.MouseButtonPress,
            QtCore.QEvent.Type.MouseButtonRelease,
            QtCore.QEvent.Type.MouseMove,
        ]

RESIZE_EVENTS = [ QtCore.QEvent.Type.Resize ]


class QtDump:

    # ----- Convert QMouseEvent -----
    @staticmethod
    def mouseevent2dict(evt):
        evt_type = evt.type()
        button = evt.button()
        buttons = evt.buttons()
        pos = evt.localPos()
        modifiers = evt.modifiers()
        # Transform each information
        info = {
            'type': evt_type.__int__(),
            'button': button.__int__(),
            'buttons': buttons.__int__(),
            'pos': [pos.x(), pos.y()],
            'modifiers': modifiers.__int__()
        }
        return info

    @staticmethod
    def dict2mouseevent(info):
        pos = QtCore.QPointF(info['pos'][0], info['pos'][1])
        mouse_event = QtGui.QMouseEvent(
            QtCore.QEvent.Type(info['type']),
            pos,
            QtCore.Qt.MouseButton(info['button']),
            QtCore.Qt.MouseButtons(info['buttons']),
            QtCore.Qt.KeyboardModifiers(info['modifiers']))
        return mouse_event

    # ----- Convert QSize -----
    @staticmethod
    def size2dict(size):
        width = size.width()
        height = size.height()
        # Transform each information
        info = {
            'width': width,
            'height': height,
        }
        return info

    @staticmethod
    def dict2size(info):
        return QtCore.QSize(info['width'], info['height'])

    # ----- Convert QResizeEvent -----
    @staticmethod
    def resizeevent2dict(evt):
        evt_type = evt.type()
        # Transform each information
        info = {
            'type': evt_type.__int__(),
            'size':    QtDump.size2dict(evt.size()),
            'oldSize': QtDump.size2dict(evt.oldSize()),
        }
        return info

    @staticmethod
    def dict2resizeevent(info):
        size = QtDump.dict2size(info['size'])
        oldSize = QtDump.dict2size(info['oldSize'])
        resize_event = QtGui.QResizeEvent(size, oldSize)
        return resize_event

    # ----- Convert dict to event -----
    @staticmethod
    def dict2event(info):
        if info['type'] in MOUSE_EVENTS:
            return QtDump.dict2mouseevent(info)
        else:
            if info['type'] in RESIZE_EVENTS:
                return QtDump.dict2resizeevent(info)
            else:
                print("Error: event not available")

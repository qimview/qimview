
from Qt import QtGui, QtCore, QtWidgets
from _md5 import md5
import numpy as np

# Adapt to different versions
if "MouseButtonPress" in QtCore.QEvent.__dict__:
    qevent_types = QtCore.QEvent
else:
    qevent_types = QtCore.QEvent.Type

MOUSE_EVENTS = [
            # qevent_typese.KeyPress,
            qevent_types.MouseButtonPress,
            qevent_types.MouseButtonRelease,
            qevent_types.MouseMove,
        ]

RESIZE_EVENTS = [ qevent_types.Resize ]

WHEEL_EVENT = [ qevent_types.Wheel]

class QtDump:

    # ----- Qt enum value -----
    @staticmethod
    def enum_value(v):
        return v.__int__()

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
            'type':     QtDump.enum_value(evt_type),
            'button':   QtDump.enum_value(button),
            'buttons':  QtDump.enum_value(buttons),
            'pos': [pos.x(), pos.y()],
            'modifiers': QtDump.enum_value(modifiers)
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

    # ----- Convert QPointF -----
    @staticmethod
    def qpointf2dict(pt):
        x = pt.x()
        y = pt.y()
        # Transform each information
        info = {
            'x': x,
            'y': y,
        }
        return info

    @staticmethod
    def dict2qpointf(info):
        return QtCore.QPointF(info['x'], info['y'])

    # ----- Convert QPoint -----
    @staticmethod
    def qpoint2dict(pt):
        x = pt.x()
        y = pt.y()
        # Transform each information
        info = {
            'x': x,
            'y': y,
        }
        return info

    @staticmethod
    def dict2qpoint(info):
        return QtCore.QPoint(info['x'], info['y'])

    # ----- Convert QWheelEvent -----
    @staticmethod
    def wheelevent2dict(evt):
        evt_type = evt.type()
        # Transform each information
        info = {
            'type':       evt_type.__int__(),
            'pos':        QtDump.qpointf2dict(evt.position()),
            'globalPos':  QtDump.qpointf2dict(evt.globalPosition()),
            'pixelDelta': QtDump.qpoint2dict (evt.pixelDelta()),
            'angleDelta': QtDump.qpoint2dict (evt.angleDelta()),
            'buttons':    QtDump.enum_value(evt.buttons()),
            'modifiers':  QtDump.enum_value(evt.modifiers()),
            'phase':      QtDump.enum_value(evt.phase()),
            'inverted':   evt.inverted(),
        }
        return info

    @staticmethod
    def dict2wheelevent(info):
        pos        = QtDump.dict2qpointf(         info['pos'])
        globalPos  = QtDump.dict2qpointf(         info['globalPos'])
        pixelDelta = QtDump.dict2qpoint(          info['pixelDelta'])
        angleDelta = QtDump.dict2qpoint(          info['angleDelta'])
        buttons    = QtCore.Qt.MouseButtons(      info['buttons'])
        modifiers  = QtCore.Qt.KeyboardModifiers( info['modifiers'])
        phase      = QtCore.Qt.ScrollPhase(       info['phase'])
        inverted   = info['inverted']
        event = QtGui.QWheelEvent(pos, globalPos, pixelDelta, angleDelta, buttons, modifiers, phase, inverted)
        return event

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
            # print(f"mouse_event {info['type']}")
            return QtDump.dict2mouseevent(info)
        else:
            if info['type'] in RESIZE_EVENTS:
                return QtDump.dict2resizeevent(info)
            else:
                if info['type'] in WHEEL_EVENT:
                    return QtDump.dict2wheelevent(info)
                else:
                    print("Error: event not available")

    # ----- widget snapshot hash -----
    @staticmethod
    def get_screen_hash(widget):
        image = QtGui.QImage(widget.size(), QtGui.QImage.Format.Format_RGB32)
        widget.render(image)
        # image.save("_pixmap.png")
        # get hash from image data pixmap.toImage().bits()
        image_bits = image.constBits()
        arr = np.array(image_bits).reshape(image.height(), image.width(), 4)
        image_hash = md5(arr).hexdigest()
        return image_hash

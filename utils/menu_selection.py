
from .qt_imports import *

class MenuSelection:
    def __init__(self, _title, _menu, _dict, _default, _callback=None):
        """[summary]

        Args:
            _title ([type]): [description]
            _menu ([type]): [description]
            _dict ([type]): [description]
            _default ([type]): [description]
            _callback ([type], optional): [description]. Defaults to None.
        """        
        self._options = _dict
        self._default = _default
        self._current = _default
        self._menu    = _menu
        self._menu.addSection(_title)
        self._group = QtWidgets.QActionGroup(_menu)
        self._actions = {}
        self._callback = _callback
        for option in self._options:
            action = QtWidgets.QAction(option,  self._menu, checkable=True)
            action.setActionGroup(self._group)
            self._menu.addAction(action)
            self._actions[option] = action
        self._group.triggered.connect(self.update_selection)
        self._actions[self._current].setChecked(True)

    def update_selection(self):
        print(f"MenuSelection update_selection()")
        # find selected option
        for option in self._options:
            if self._actions[option].isChecked():
                self._current = option
                print(f" selection is {option}")
        if self._callback is not None:
            self._callback()
    
    def get_selection(self):
        return self._current
    
    def get_selection_value(self):
        return self._options[self._current]
    
    def set_selection(self, selection):
        self._current = selection
        self._actions[self._current].setChecked(True)

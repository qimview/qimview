""" Simple TabDialog class """

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QDialog,
    QTabWidget,
    QDialogButtonBox,
    QTextBrowser,
    QSizePolicy
)
from PySide6 import QtCore
from typing import Optional

class TabDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None, title: str = "TabDialog") -> None:
        super().__init__(parent)
        # self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.tab_widget = QTabWidget()
        self.tab_widget.setUsesScrollButtons(False)
        # self.tab_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        button_box = QDialogButtonBox( QDialogButtonBox.StandardButton.Ok )
        button_box.accepted.connect(self.accept)
        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.tab_widget)
        self.main_layout.addWidget(button_box)
        self.setLayout(self.main_layout)
        self.setWindowTitle(title)

    def add_tab(self, widget: QWidget, title:str)->None:
        self.tab_widget.addTab(widget, title)

    def add_markdown_tab(self, title:str, text: str)->None:
        text_browser = QTextBrowser()
        text_browser.setOpenExternalLinks(True)
        # # Avoid scroll bars for the moment
        # textedit.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # text_browser.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        text_browser.setReadOnly(True)
        text_browser.setMarkdown(text)
        self.tab_widget.addTab(text_browser,title)
        # self.tab_widget.updateGeometry()

if __name__ == "__main__":
    from PySide6.QtWidgets import (
        QApplication,
        QTextEdit,
    )
    import sys
    # Usage example
    app = QApplication(sys.argv)
    tab_dialog = TabDialog()
    text1 = QTextEdit()
    text1.setReadOnly(True)
    text1.setMarkdown(" markdown text here  \n"
                      "second line")
    tab_dialog.add_tab(text1,"text1")

    text2 = QTextEdit()
    text2.setReadOnly(True)
    text2.setMarkdown("## section  \n"
                      " markdown text here  \n"
                      "second line")
    tab_dialog.add_tab(text2,"text2")
    tab_dialog.show()
    sys.exit(app.exec())

import sys
from PySide6 import QtCore, QtWidgets

from .main_window import MainWindow


class SelectableLabels(QtCore.QObject):
    """
    Give every QLabel that shows text an I-beam cursor and text that the mouse can select. Install it on the QApplication object, which passes each event of each object to the filter. Qt polishes a widget one time before it first becomes visible, so a label that a dialog builds later also gets the flag. Qt sets the focus policy of a selectable label to ClickFocus, so a click on a label moves the keyboard focus away from an input field.
    """

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.Polish and isinstance(watched, QtWidgets.QLabel) and watched.pixmap().isNull():
            watched.setTextInteractionFlags(watched.textInteractionFlags() | QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            watched.setCursor(QtCore.Qt.CursorShape.IBeamCursor)
        return False


def main() -> int:
    qt_application = QtWidgets.QApplication(sys.argv)
    qt_application.installEventFilter(SelectableLabels(qt_application))
    main_window = MainWindow()
    main_window.resize(1200, 800)
    main_window.show()
    return qt_application.exec()

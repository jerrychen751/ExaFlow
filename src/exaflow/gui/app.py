import sys
from PySide6 import QtCore, QtWidgets

from .main_window import MainWindow


class SelectableLabels(QtCore.QObject):
    """
    Let QLabels which show text become selectable (cursor becomes I-beam instead of pointer upon hover).
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

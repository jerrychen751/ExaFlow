import sys
from PySide6 import QtWidgets

from .main_window import MainWindow


def main() -> int:
    qt_application = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.resize(1200, 800)
    main_window.show()
    return qt_application.exec()



import sys
import os
import time
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from gui import ArnisApp, SplashScreen

os.environ["QT_LOGGING_RULES"] = "qt5*.debug=false"

if __name__ == "__main__":
    import numpy as np
    from PyQt5.QtCore import QMetaType
    try:
        QMetaType.register(np.ndarray)
    except:
        pass

    app = QApplication(sys.argv)

    splash = SplashScreen()
    splash.show()
    app.processEvents()

    splash.update_progress(10, "Loading models...")
    app.processEvents()
    time.sleep(1)

    splash.update_progress(40, "Initializing UI...")
    app.processEvents()
    time.sleep(0.5)

    splash.update_progress(70, "Setting up cameras...")
    app.processEvents()
    time.sleep(0.5)

    splash.update_progress(90, "Almost ready...")
    app.processEvents()
    window = ArnisApp()

    splash.update_progress(100, "Ready!")
    app.processEvents()
    time.sleep(0.5)

    splash.close()
    window.show()

    sys.exit(app.exec_())

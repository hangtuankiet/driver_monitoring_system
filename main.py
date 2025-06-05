import tkinter as tk
from src.gui import DriverMonitoringGUI
from src.utils import create_storage_directories

def main():
    create_storage_directories()
    root = tk.Tk()
    DriverMonitoringGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
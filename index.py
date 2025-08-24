import tkinter as tk
from tkinter import filedialog, Listbox, Scrollbar, Frame
from rag import build_index
import threading

class IndexBuilderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Index Builder")
        self.root.geometry("600x400")

        self.file_paths = set()

        # Top frame for buttons
        top_frame = Frame(self.root)
        top_frame.pack(pady=10)

        self.select_button = tk.Button(top_frame, text="Select Files", command=self.select_files)
        self.select_button.pack(side=tk.LEFT, padx=5)

        self.build_button = tk.Button(top_frame, text="Build Index", command=self.start_indexing_thread)
        self.build_button.pack(side=tk.LEFT, padx=5)

        # Frame for listbox
        list_frame = Frame(self.root)
        list_frame.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        self.listbox = Listbox(list_frame)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = Scrollbar(list_frame, orient=tk.VERTICAL)
        scrollbar.config(command=self.listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.listbox.config(yscrollcommand=scrollbar.set)

        # Status label
        self.status_label = tk.Label(self.root, text="", fg="blue")
        self.status_label.pack(pady=5)

    def select_files(self):
        files = filedialog.askopenfilenames(
            title="Select files to index",
            filetypes=[
                ("All files", "*.*"),
                ("Text files", "*.txt")
            ]
        )
        if files:
            self.file_paths.update(files)
            self.listbox.delete(0, tk.END)
            for file in sorted(list(self.file_paths)):
                self.listbox.insert(tk.END, file)
            self.status_label.config(text=f"{len(self.file_paths)} files selected.")

    def start_indexing_thread(self):
        if not self.file_paths:
            self.status_label.config(text="Error: No files selected!", fg="red")
            return
        
        self.build_button.config(state=tk.DISABLED)
        self.select_button.config(state=tk.DISABLED)
        self.status_label.config(text="Indexing in progress...", fg="blue")

        # Run indexing in a separate thread to avoid freezing the GUI
        thread = threading.Thread(target=self.run_indexing)
        thread.start()

    def run_indexing(self):
        try:
            build_index(list(self.file_paths))
            self.root.after(0, self.on_indexing_complete)
        except Exception as e:
            self.root.after(0, self.on_indexing_error, e)

    def on_indexing_complete(self):
        self.status_label.config(text="Indexing complete!", fg="green")
        self.build_button.config(state=tk.NORMAL)
        self.select_button.config(state=tk.NORMAL)

    def on_indexing_error(self, error):
        self.status_label.config(text=f"Error during indexing: {error}", fg="red")
        self.build_button.config(state=tk.NORMAL)
        self.select_button.config(state=tk.NORMAL)


if __name__ == "__main__":
    root = tk.Tk()
    app = IndexBuilderApp(root)
    root.mainloop()
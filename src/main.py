import os
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk  # Для работы с изображениями

import matplotlib
matplotlib.use('TkAgg')

from file_loader import load_file_and_show_table
from plotter import Plotter
from table_handler import TableHandler


class MainApp:
    """
    Класс основного приложения для обработки данных и визуализации.
    """

    def __init__(self, root):
        """
        Инициализация главного окна приложения.

        :param root: Корневое окно Tkinter.
        """
        self.root = root
        self.root.title("Умный обработчик данных")
        self.root.geometry("600x500")
        self.root.resizable(False, False)

        self.data = None  # Хранение загруженных данных

        # Установка иконки приложения
        self._set_app_icon()

        # Установка темы интерфейса
        self._set_theme()

        # Создание интерфейса
        self._create_interface()

    def _set_app_icon(self):
        """Устанавливает иконку приложения."""
        icon_path = os.path.join(os.path.dirname(__file__), "icons", "app_icon.png")
        if os.path.exists(icon_path):
            try:
                icon_image = Image.open(icon_path)
                icon_image = icon_image.resize((64, 64), Image.LANCZOS)
                icon_photo = ImageTk.PhotoImage(icon_image)
                self.root.iconphoto(False, icon_photo)
                self.icon_photo = icon_photo  # Сохраняем ссылку, чтобы изображение не удалилось сборщиком мусора
            except Exception as e:
                messagebox.showwarning("Предупреждение", f"Не удалось загрузить иконку приложения: {e}")
        else:
            messagebox.showwarning("Предупреждение", f"Файл иконки приложения не найден: {icon_path}")

    def _set_theme(self):
        """Устанавливает тему интерфейса."""
        self.style = ttk.Style()
        self.style.theme_use('clam')  # Доступные темы: 'clam', 'alt', 'default', 'classic'

    def _create_interface(self):
        """Создаёт основной интерфейс приложения."""
        # Основная рамка с отступами
        main_frame = ttk.Frame(self.root, padding="20 20 20 20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Меню приложения
        self._create_menu()

        # Используем Notebook для вкладок
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Создание вкладок
        self._create_tabs(notebook)

        # Кнопка выхода в нижней части окна
        self._create_exit_button(main_frame)

    def _create_menu(self):
        """Создаёт меню приложения."""
        menu = tk.Menu(self.root)
        self.root.config(menu=menu)

        # Файл -> Загрузить данные, Выход
        file_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="Файл", menu=file_menu)
        file_menu.add_command(label="Загрузить и отобразить данные", command=self.load_data)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.root.quit)

        # Графики
        graph_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="Графики", menu=graph_menu)
        graph_menu.add_command(label="Построить график", command=self.open_plotter)

        # Таблицы
        table_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="Таблицы", menu=table_menu)
        table_menu.add_command(label="Обработать таблицу", command=self.open_table_handler)

    def _create_tabs(self, notebook):
        """Создаёт вкладки приложения."""
        # Вкладка Загрузка
        load_tab = ttk.Frame(notebook, padding="10 10 10 10")
        notebook.add(load_tab, text="Загрузка данных")

        # Вкладка Графики
        plot_tab = ttk.Frame(notebook, padding="10 10 10 10")
        notebook.add(plot_tab, text="Графики")

        # Вкладка Таблицы
        table_tab = ttk.Frame(notebook, padding="10 10 10 10")
        notebook.add(table_tab, text="Таблицы")

        # Загрузка иконок для кнопок
        load_icon = self._load_icon("load_icon.png")
        plot_icon = self._load_icon("plot_icon.png")
        table_icon = self._load_icon("table_icon.png")

        # Кнопки на вкладке Загрузка
        load_button = ttk.Button(
            load_tab,
            text="Загрузить и отобразить данные",
            command=self.load_data,
            image=load_icon,
            compound=tk.LEFT
        )
        if load_icon:
            load_button.image = load_icon  # Сохраняем ссылку
        load_button.pack(fill=tk.X, pady=10)

        # Кнопки на вкладке Графики
        plot_button = ttk.Button(
            plot_tab,
            text="Построить график",
            command=self.open_plotter,
            image=plot_icon,
            compound=tk.LEFT
        )
        if plot_icon:
            plot_button.image = plot_icon
        plot_button.pack(fill=tk.X, pady=10)

        # Кнопки на вкладке Таблицы
        table_button = ttk.Button(
            table_tab,
            text="Обработать таблицу",
            command=self.open_table_handler,
            image=table_icon,
            compound=tk.LEFT
        )
        if table_icon:
            table_button.image = table_icon
        table_button.pack(fill=tk.X, pady=10)

    def _create_exit_button(self, parent):
        """Создаёт кнопку выхода."""
        exit_icon = self._load_icon("exit_icon.png")
        exit_button = ttk.Button(
            parent,
            text="Выход",
            command=self.root.quit,
            image=exit_icon,
            compound=tk.LEFT
        )
        if exit_icon:
            exit_button.image = exit_icon
        exit_button.pack(side=tk.BOTTOM, pady=10)

    def _load_icon(self, icon_name):
        """
        Загружает иконку из папки icons.

        :param icon_name: Имя файла иконки.
        :return: Объект PhotoImage с иконкой или None при ошибке.
        """
        icon_path = os.path.join(os.path.dirname(__file__), "icons", icon_name)
        if os.path.exists(icon_path):
            try:
                icon_image = Image.open(icon_path)
                icon_image = icon_image.resize((20, 20), Image.LANCZOS)
                return ImageTk.PhotoImage(icon_image)
            except Exception as e:
                messagebox.showwarning("Предупреждение", f"Не удалось загрузить иконку '{icon_name}': {e}")
                return None
        else:
            messagebox.showwarning("Предупреждение", f"Файл иконки не найден: {icon_path}")
            return None

    def load_data(self):
        """Загружает данные и отображает таблицу в новом окне."""
        self.data = load_file_and_show_table()
        if self.data is not None:
            messagebox.showinfo("Успех", "Данные успешно загружены.")
        else:
            messagebox.showerror("Ошибка", "Не удалось загрузить данные.")

    def open_plotter(self):
        """Открывает окно построения графика, если данные загружены."""
        if self.data is not None:
            plot_window = tk.Toplevel(self.root)
            plot_window.title("Построение графика")
            plot_window.geometry("800x600")
            # Установка иконки для окна графика
            plot_icon = self._load_icon("plot_icon.png")
            if plot_icon:
                plot_window.iconphoto(False, plot_icon)
            Plotter(plot_window, self.data)
        else:
            messagebox.showwarning("Внимание", "Пожалуйста, загрузите данные перед построением графика.")

    def open_table_handler(self):
        """Открывает окно для обработки таблицы."""
        if self.data is not None:
            table_window = tk.Toplevel(self.root)
            table_window.title("Обработка таблицы")
            table_window.geometry("900x700")
            # Установка иконки для окна обработки таблицы
            table_icon = self._load_icon("table_icon.png")
            if table_icon:
                table_window.iconphoto(False, table_icon)
            TableHandler(table_window, self.data)
        else:
            messagebox.showwarning("Внимание", "Пожалуйста, загрузите данные перед обработкой таблицы.")


if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()

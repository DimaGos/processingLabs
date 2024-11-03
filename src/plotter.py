import tkinter as tk
from tkinter import ttk, messagebox, colorchooser, filedialog
import matplotlib.pyplot as plt
import seaborn as sns  # Подключаем Seaborn
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.special import factorial  # Для функции факториала в распределении Пуассона

# Настройки matplotlib без использования LaTeX
import matplotlib
matplotlib.rcParams['text.usetex'] = False  # Отключаем использование LaTeX
matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # Устанавливаем шрифт с поддержкой кириллицы

class Plotter:
    def __init__(self, master, data):
        self.master = master
        self.master.title("Построение графика")
        self.data = data.reset_index(drop=True)  # Сброс индексов
        self.theoretical_functions = []  # Список теоретических функций
        self.outliers_removed = False  # Флаг удаления выбросов
        self.outliers = set()  # Набор индексов выбросов
        self.selected_points = set()  # Набор выбранных пользователем точек

        try:
            if self.data is not None and isinstance(self.data, pd.DataFrame):
                print("Данные успешно получены в Plotter.")
                print(self.data.head())  # Отладка: вывод первых строк данных
            else:
                messagebox.showerror("Ошибка", "Данные не были переданы в Plotter или имеют некорректный формат.")
                self.master.destroy()
                return

            self.create_widgets()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка при инициализации Plotter: {e}")
            print(f"Ошибка в Plotter __init__: {e}")
            self.master.destroy()

    def create_widgets(self):
        try:
            # Создаем контейнер для настроек с прокруткой
            settings_container = tk.Frame(self.master)
            settings_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=10, pady=10)

            # Создаем Canvas для прокрутки
            self.settings_canvas = tk.Canvas(settings_container)
            self.settings_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            # Добавляем скроллбар
            scrollbar = ttk.Scrollbar(settings_container, orient=tk.VERTICAL, command=self.settings_canvas.yview)
            scrollbar.pack(side=tk.RIGHT, fill='y')

            # Настраиваем Canvas
            self.settings_canvas.configure(yscrollcommand=scrollbar.set)
            self.settings_canvas.bind('<Configure>', lambda e: self.settings_canvas.configure(scrollregion=self.settings_canvas.bbox('all')))

            # Создаем внутренний фрейм для размещения виджетов настроек
            settings_frame = tk.Frame(self.settings_canvas)
            self.settings_canvas.create_window((0, 0), window=settings_frame, anchor='nw')

            # Обработка прокрутки колесиком мыши
            def _on_mousewheel(event):
                self.settings_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

            self.settings_canvas.bind_all('<MouseWheel>', _on_mousewheel)

            # Выбор столбцов для осей X и Y
            self.x_label_widget = tk.Label(settings_frame, text="Выберите столбец для оси X:")
            self.x_label_widget.pack(pady=5)
            self.x_column = ttk.Combobox(settings_frame, values=list(self.data.columns))
            self.x_column.pack(pady=5)

            self.y_label_widget = tk.Label(settings_frame, text="Выберите столбец для оси Y:")
            self.y_label_widget.pack(pady=5)
            self.y_column = ttk.Combobox(settings_frame, values=list(self.data.columns))
            self.y_column.pack(pady=5)

            # Опции логарифмирования
            self.log_x_var = tk.BooleanVar()
            self.log_y_var = tk.BooleanVar()
            self.log_x_check = tk.Checkbutton(settings_frame, text="Логарифмировать X", variable=self.log_x_var)
            self.log_x_check.pack(pady=2)
            self.log_y_check = tk.Checkbutton(settings_frame, text="Логарифмировать Y", variable=self.log_y_var)
            self.log_y_check.pack(pady=2)

            # Выбор метода аппроксимации
            tk.Label(settings_frame, text="Выберите метод аппроксимации:").pack(pady=5)
            self.approx_method = tk.StringVar()
            self.approx_method.set('linear')
            methods = [("Линейная аппроксимация", 'linear'),
                       ("Полиномиальная аппроксимация", 'polyfit'),
                       ("Аппроксимация Пуассона", 'poisson'),
                       ("Аппроксимация Гаусса", 'gauss'),
                       ("Без аппроксимации", 'none')]
            for text, method in methods:
                tk.Radiobutton(settings_frame, text=text, variable=self.approx_method, value=method, command=self.update_widgets).pack(anchor=tk.W)

            # Ввод степени аппроксимации
            self.degree_label = tk.Label(settings_frame, text="Степень аппроксимации:")
            self.degree_label.pack(pady=5)
            self.degree_entry = tk.Entry(settings_frame)
            self.degree_entry.insert(0, "2")
            self.degree_entry.pack(pady=5)

            # Названия осей
            tk.Label(settings_frame, text="Название оси X:").pack(pady=5)
            self.x_label_entry = tk.Entry(settings_frame)
            self.x_label_entry.pack(pady=5)
            tk.Label(settings_frame, text="Название оси Y:").pack(pady=5)
            self.y_label_entry = tk.Entry(settings_frame)
            self.y_label_entry.pack(pady=5)

            # Добавление теоретических функций
            tk.Label(settings_frame, text="Теоретические функции:").pack(pady=5)
            self.function_entry = tk.Entry(settings_frame)
            self.function_entry.pack(pady=5)
            tk.Label(settings_frame, text="Используйте 'x' как переменную. Пример: sin(x)").pack(pady=2)

            # Секция с примерами функций
            examples_frame = tk.LabelFrame(settings_frame, text="Примеры функций")
            examples_frame.pack(pady=5, fill=tk.BOTH, expand=True)
            examples_text = (
                "sin(x)\n"
                "cos(x)\n"
                "exp(-x)\n"
                "0.125 * exp(-((x - 10.4) / (2 * 3.23**2)))\n"
                "log(x)\n"
                "x ** 2 + 3 * x + 5"
            )
            tk.Label(examples_frame, text=examples_text, justify=tk.LEFT).pack()

            # Кнопки для добавления и удаления функций
            function_buttons_frame = tk.Frame(settings_frame)
            function_buttons_frame.pack(pady=5)
            tk.Button(function_buttons_frame, text="Добавить функцию", command=self.add_function).pack(side=tk.LEFT, padx=5)
            tk.Button(function_buttons_frame, text="Удалить функцию", command=self.delete_function).pack(side=tk.LEFT, padx=5)

            self.function_listbox = tk.Listbox(settings_frame)
            self.function_listbox.pack(pady=5)

            # Выбор типа графика
            tk.Label(settings_frame, text="Выберите тип графика:").pack(pady=5)
            self.plot_type = tk.StringVar()
            self.plot_type.set('scatter')
            plot_types = [("Точечный график", 'scatter'),
                          ("Линейный график", 'line'),
                          ("Гистограмма", 'histogram'),
                          ("Столбчатая диаграмма", 'bar')]
            for text, value in plot_types:
                tk.Radiobutton(settings_frame, text=text, variable=self.plot_type, value=value).pack(anchor=tk.W)

            # Выбор цвета графика
            tk.Label(settings_frame, text="Цвет графика:").pack(pady=5)
            self.plot_color = '#0000FF'  # Цвет по умолчанию: синий

            def choose_plot_color():
                color = colorchooser.askcolor(title="Выберите цвет графика")[1]
                if color:
                    self.plot_color = color
                    self.color_button.config(bg=color)

            self.color_button = tk.Button(settings_frame, text="Выбрать цвет", command=choose_plot_color, bg=self.plot_color)
            self.color_button.pack(pady=5)

            # Кнопки управления
            tk.Button(settings_frame, text="Построить график", command=self.plot_graph).pack(pady=10)
            tk.Button(settings_frame, text="Показать коэффициенты", command=self.show_coefficients).pack(pady=5)
            tk.Button(settings_frame, text="Сохранить график в PDF", command=self.save_graph).pack(pady=5)
            tk.Button(settings_frame, text="Работа с выбросами", command=self.outlier_menu).pack(pady=5)

            # Фрейм для графика
            self.figure = plt.Figure(figsize=(6, 5))
            self.ax = self.figure.add_subplot(111)
            self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
            self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

            # Переменные для хранения данных
            self.x = None
            self.y = None
            self.scatter = None  # Для хранения графика точек

            # Связывание событий
            self.canvas.mpl_connect('button_press_event', self.on_click)

            # Флаг для режима выбора точек
            self.selecting_points = False

            # Переменные для хранения коэффициентов и уравнения
            self.coeffs_str = ''
            self.equation = ''
            self.r_squared = None

            # Обработка изменения метода аппроксимации
            self.approx_method.trace('w', self.update_widgets)
            self.update_widgets()  # Инициализация виджетов в соответствии с выбранным методом

        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка при создании виджетов: {e}")
            print(f"Ошибка в create_widgets: {e}")
            self.master.destroy()

    def update_widgets(self, *args):
        method = self.approx_method.get()
        if method == 'polyfit':
            self.degree_label.config(state='normal')
            self.degree_entry.config(state='normal')
        else:
            self.degree_label.config(state='disabled')
            self.degree_entry.config(state='disabled')

        if method == 'poisson':
            # Скрываем выбор столбца Y и логарифмирование Y
            self.y_label_widget.pack_forget()
            self.y_column.pack_forget()
            self.log_y_check.pack_forget()
            self.y_label_entry.pack_forget()
        else:
            # Показываем выбор столбца Y и логарифмирование Y
            self.y_label_widget.pack(pady=5)
            self.y_column.pack(pady=5)
            self.log_y_check.pack(pady=2)
            self.y_label_entry.pack(pady=5)

    def add_function(self):
        func_text = self.function_entry.get()
        if func_text:
            color = colorchooser.askcolor(title="Выберите цвет для функции")[1]
            if color:
                self.theoretical_functions.append({'func': func_text, 'color': color})
                self.function_listbox.insert(tk.END, func_text)
                self.function_entry.delete(0, tk.END)
            else:
                messagebox.showwarning("Предупреждение", "Выберите цвет перед добавлением функции.")
        else:
            messagebox.showwarning("Предупреждение", "Введите функцию перед добавлением.")

    def delete_function(self):
        selected_index = self.function_listbox.curselection()
        if selected_index:
            index = selected_index[0]
            self.function_listbox.delete(index)
            del self.theoretical_functions[index]
        else:
            messagebox.showwarning("Предупреждение", "Выберите функцию для удаления.")

    def outlier_menu(self):
        # Создаем окно для выбора метода работы с выбросами
        menu_window = tk.Toplevel(self.master)
        menu_window.title("Работа с выбросами")

        tk.Label(menu_window, text="Выберите метод работы с выбросами:").pack(pady=10)

        tk.Button(menu_window, text="Автоматическое обнаружение выбросов",
                  command=lambda: [menu_window.destroy(), self.detect_outliers()]).pack(pady=5)
        tk.Button(menu_window, text="Ручной выбор выбросов",
                  command=lambda: [menu_window.destroy(), self.enable_point_selection()]).pack(pady=5)
        tk.Button(menu_window, text="Завершить выбор точек",
                  command=lambda: [menu_window.destroy(), self.disable_point_selection()]).pack(pady=5)
        tk.Button(menu_window, text="Удалить выбранные выбросы",
                  command=lambda: [menu_window.destroy(), self.remove_selected_outliers()]).pack(pady=5)
        tk.Button(menu_window, text="Сбросить выбросы",
                  command=lambda: [menu_window.destroy(), self.reset_outliers()]).pack(pady=5)

    def disable_point_selection(self):
        self.selecting_points = False
        messagebox.showinfo("Режим выбора", "Режим выбора точек отключен.")

    def detect_outliers(self):
        try:
            x_col = self.x_column.get()
            y_col = self.y_column.get()

            method = self.approx_method.get()
            if method == 'poisson':
                y_col = None  # Игнорируем y_col для Пуассона

            if x_col not in self.data.columns and (y_col not in self.data.columns or y_col is not None):
                messagebox.showerror("Ошибка", "Выберите корректные столбцы для осей.")
                return

            data = self.get_filtered_data()

            if method == 'poisson':
                x = data[x_col].astype(float)
                if self.log_x_var.get():
                    x = np.log(x)
                X = x.values.reshape(-1, 1)
            else:
                x = data[x_col].astype(float)
                y = data[y_col].astype(float)

                if self.log_x_var.get():
                    x = np.log(x)
                if self.log_y_var.get():
                    y = np.log(y)

                X = np.column_stack((x, y))

            # Используем метод интерквартильного размаха (IQR)
            self.outliers_mask = self.detect_outliers_iqr(X)

            # Если выбросы найдены
            if np.any(self.outliers_mask):
                # Отобразить выбросы на графике
                self.outliers = set(data.index[self.outliers_mask])
                self.plot_graph()
                messagebox.showinfo("Обнаружены выбросы", "Выбросы автоматически обнаружены и отмечены на графике.")
            else:
                messagebox.showinfo("Выбросы не обнаружены", "Выбросы не обнаружены в данных.")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось обнаружить выбросы: {e}")

    def detect_outliers_iqr(self, data):
        # Метод интерквартильного размаха (IQR)
        Q1 = np.percentile(data, 25, axis=0)
        Q3 = np.percentile(data, 75, axis=0)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((data < lower_bound) | (data > upper_bound)).any(axis=1)
        return outliers

    def enable_point_selection(self):
        if self.x is None:
            messagebox.showwarning("Предупреждение", "Сначала постройте график.")
            return
        self.selecting_points = True
        messagebox.showinfo("Режим выбора", "Кликните на точках, которые хотите отметить как выбросы.\nНажмите 'Работа с выбросами' -> 'Завершить выбор точек' для выхода из режима.")

    def on_click(self, event):
        if not self.selecting_points:
            return

        if event.inaxes != self.ax:
            return

        # Получаем координаты клика
        x_click = event.xdata
        y_click = event.ydata

        # Находим ближайшую точку
        if self.y is not None:
            distances = np.hypot(self.x - x_click, self.y - y_click)
        else:
            distances = np.abs(self.x - x_click)
        min_index = np.argmin(distances)
        # Устанавливаем порог расстояния для выбора точки
        threshold = 0.05 * (self.ax.get_xlim()[1] - self.ax.get_xlim()[0])

        if distances[min_index] < threshold:
            # Получаем индекс точки в исходных данных
            point_index = self.data_filtered.index[min_index]

            # Переключаем состояние точки (выбрана/не выбрана)
            if point_index in self.selected_points:
                self.selected_points.remove(point_index)
            else:
                self.selected_points.add(point_index)
            self.plot_graph()

    def remove_selected_outliers(self):
        if not self.selected_points and not self.outliers:
            messagebox.showwarning("Предупреждение", "Не выбрано ни одной точки для удаления.")
            return
        # Объединяем выбранные точки и автоматически обнаруженные выбросы
        indices = sorted(self.selected_points.union(self.outliers))
        if not indices:
            messagebox.showwarning("Предупреждение", "Не выбрано ни одной точки для удаления.")
            return

        # Удаляем выбранные точки из данных
        self.outliers_removed = True
        self.filtered_data = self.data.drop(indices).reset_index(drop=True)
        self.selected_points.clear()
        self.outliers.clear()
        messagebox.showinfo("Выбросы удалены", "Выбранные выбросы удалены. Нажмите 'Построить график' для обновления.")
        self.selecting_points = False

    def reset_outliers(self):
        # Сбрасываем все выбросы и выбор точек
        self.outliers_removed = False
        self.selected_points.clear()
        self.outliers.clear()
        self.selecting_points = False
        messagebox.showinfo("Сброс выбросов", "Все выбросы и выбор точек сброшены.")
        self.plot_graph()

    def save_graph(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf"), ("PNG files", "*.png")])
        if file_path:
            self.figure.savefig(file_path)
            messagebox.showinfo("Сохранение", f"График успешно сохранён в {file_path}")

    def get_filtered_data(self):
        """Получает данные с учетом удаления выбросов и исключения значений None/NaN."""
        if self.outliers_removed:
            data = self.filtered_data.copy()
        else:
            data = self.data.copy()

        # Исключаем строки, где x или y являются None или NaN
        x_col = self.x_column.get()
        y_col = self.y_column.get()

        method = self.approx_method.get()
        if method == 'poisson':
            if x_col in data.columns:
                data = data[data[x_col].notna()]
            data.reset_index(drop=True, inplace=True)
            return data

        if x_col in data.columns:
            data = data[data[x_col].notna()]
        if y_col in data.columns:
            data = data[data[y_col].notna()]

        data.reset_index(drop=True, inplace=True)
        return data

    def plot_graph(self):
        try:
            x_col = self.x_column.get()
            y_col = self.y_column.get()
            method = self.approx_method.get()

            if x_col not in self.data.columns:
                messagebox.showerror("Ошибка", "Выберите корректный столбец для оси X.")
                return

            data = self.get_filtered_data()

            if data.empty:
                messagebox.showerror("Ошибка", "Данные для построения графика отсутствуют после фильтрации.")
                return

            # Приводим данные к числовому типу и обрабатываем ошибки
            try:
                x = data[x_col].astype(float)
            except ValueError:
                messagebox.showerror("Ошибка", f"Столбец {x_col} содержит некорректные данные для оси X.")
                return

            if self.log_x_var.get():
                x = np.log(x)

            self.x = x.values
            x_label = self.x_label_entry.get() or x_col

            self.data_filtered = data  # Сохраняем фильтрованные данные для использования при выборе точек

            plot_type = self.plot_type.get()

            self.ax.clear()

            if method == 'poisson':
                self.y = None
                y_label = 'Частота'

                # Построение гистограммы данных
                counts, bins, patches = self.ax.hist(self.x, bins=range(int(min(self.x)), int(max(self.x)) + 2), align='left', color=self.plot_color, alpha=0.6, label='Данные')

                # Обработка аппроксимации
                self.perform_approximation(method)

                # Построение теоретических функций
                self.plot_theoretical_functions()

                # Настройка осей
                self.ax.set_xlabel(x_label)
                self.ax.set_ylabel(y_label)
                self.ax.set_title(f"Гистограмма {x_label}")

            else:
                if y_col not in self.data.columns:
                    messagebox.showerror("Ошибка", "Выберите корректный столбец для оси Y.")
                    return
                try:
                    y = data[y_col].astype(float)
                except ValueError:
                    messagebox.showerror("Ошибка", f"Столбец {y_col} содержит некорректные данные для оси Y.")
                    return

                if self.log_y_var.get():
                    y = np.log(y)

                self.y = y.values
                y_label = self.y_label_entry.get() or y_col

                # Отображение данных
                if plot_type == 'scatter':
                    self.scatter = self.ax.scatter(self.x, self.y, label='Данные', color=self.plot_color)
                elif plot_type == 'line':
                    self.scatter, = self.ax.plot(self.x, self.y, label='Данные', color=self.plot_color)
                elif plot_type == 'histogram':
                    sns.histplot(self.x, bins=20, kde=False, ax=self.ax, color=self.plot_color, stat='probability', label='Гистограмма')
                elif plot_type == 'bar':
                    sns.barplot(x=self.x, y=self.y, ax=self.ax, color=self.plot_color)
                else:
                    messagebox.showerror("Ошибка", f"Неизвестный тип графика: {plot_type}")
                    return

                # Если есть выбросы, выделяем их
                if plot_type == 'scatter':
                    for idx in self.selected_points.union(self.outliers):
                        if idx in data.index:
                            point_idx = data.index.get_loc(idx)
                            self.ax.scatter(self.x[point_idx], self.y[point_idx], color='red', s=100)

                # Обработка аппроксимации
                if method != 'none':
                    self.perform_approximation(method)

                # Построение теоретических функций
                self.plot_theoretical_functions()

                # Настройка осей
                self.ax.set_xlabel(x_label)
                self.ax.set_ylabel(y_label)
                self.ax.set_title(f"График {y_label} от {x_label}")

            self.ax.legend()
            self.ax.grid(True)
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка при построении графика: {e}")
            print(f"Ошибка в plot_graph: {e}")

    def plot_theoretical_functions(self):
        x_min, x_max = self.ax.get_xlim()
        x_func = np.linspace(x_min, x_max, 500)

        # Разрешенные имена для eval
        allowed_names = {
            'x': x_func,
            'np': np,
            'e': np.e,
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan,
            'exp': np.exp,
            'log': np.log,
            'sqrt': np.sqrt,
            'pi': np.pi,
            'arcsin': np.arcsin,
            'arccos': np.arccos,
            'arctan': np.arctan,
            'sinh': np.sinh,
            'cosh': np.cosh,
            'tanh': np.tanh,
        }

        for func_item in self.theoretical_functions:
            func_text = func_item['func']
            color = func_item['color']
            try:
                # Проверка на безопасность функции
                y_func = eval(func_text, {"__builtins__": None}, allowed_names)
                if not isinstance(y_func, np.ndarray):
                    raise ValueError("Функция должна возвращать массив значений.")
                self.ax.plot(x_func, y_func, label=f'Функция: {func_text}', color=color)
            except Exception as e:
                messagebox.showerror("Ошибка", f"Некорректная функция '{func_text}': {e}")
                return

    def perform_approximation(self, method):
        try:
            # Инициализируем строки коэффициентов и уравнения
            self.coeffs_str = ''
            self.equation = ''
            self.r_squared = None

            # Учитываем удаленные выбросы
            if self.outliers_removed:
                x_fit = self.x
                y_fit = self.y if self.y is not None else None
            else:
                mask = np.ones(len(self.x), dtype=bool)
                indices = [self.data_filtered.index.get_loc(idx) for idx in self.selected_points.union(self.outliers) if idx in self.data_filtered.index]
                mask[indices] = False
                x_fit = self.x[mask]
                y_fit = self.y[mask] if self.y is not None else None

            if method == 'linear':
                if len(x_fit) < 2:
                    messagebox.showerror("Ошибка", "Недостаточно точек для линейной аппроксимации.")
                    return

                # Выполняем линейную аппроксимацию
                coeffs = np.polyfit(x_fit, y_fit, 1)
                y_fit_line = np.polyval(coeffs, x_fit)

                # Вычисление R^2
                ss_res = np.sum((y_fit - y_fit_line) ** 2)
                ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                self.r_squared = r_squared

                # Построение аппроксимации
                x_plot = np.linspace(np.min(x_fit), np.max(x_fit), 500)
                y_plot = np.polyval(coeffs, x_plot)
                self.ax.plot(x_plot, y_plot, label='Линейная аппроксимация', color='green')

                # Формирование формулы для отображения на графике
                equation = f"y = {coeffs[0]:.3g}·x + {coeffs[1]:.3g}"

                # Подготовка коэффициентов для отображения
                coeffs_str = f"Коэффициент 0 (наклон): {coeffs[0]:.4f}\nКоэффициент 1 (свободный член): {coeffs[1]:.4f}"

                # Сохраняем коэффициенты и уравнение
                self.coeffs_str = coeffs_str
                self.equation = equation

                # Отображение формулы и R^2 на графике
                self.ax.text(0.05, 0.95, equation, transform=self.ax.transAxes,
                             fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
                r_squared_text = f"R² = {r_squared:.4f}"
                self.ax.text(0.05, 0.85, r_squared_text, transform=self.ax.transAxes,
                             fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

            elif method == 'polyfit':
                degree_str = self.degree_entry.get()
                try:
                    degree = int(degree_str)
                except ValueError:
                    messagebox.showerror("Ошибка", "Введите корректную степень аппроксимации.")
                    return

                if degree <= 0:
                    messagebox.showerror("Ошибка", "Степень аппроксимации должна быть положительной.")
                    return

                if len(x_fit) < degree + 1:
                    messagebox.showerror("Ошибка", "Недостаточно точек для выбранной степени аппроксимации.")
                    return

                # Выполняем полиномиальную аппроксимацию
                coeffs = np.polyfit(x_fit, y_fit, degree)
                x_plot = np.linspace(np.min(x_fit), np.max(x_fit), 500)
                y_plot = np.polyval(coeffs, x_plot)

                # Вычисление предсказанных значений на обучающих точках
                y_fit_line = np.polyval(coeffs, x_fit)

                # Вычисление R^2
                ss_res = np.sum((y_fit - y_fit_line) ** 2)
                ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                self.r_squared = r_squared

                # Построение аппроксимации
                self.ax.plot(x_plot, y_plot, label=f'Полиномиальная аппроксимация (степень {degree})', color='red')

                # Формирование формулы для отображения на графике
                coeffs_text_parts = []
                degree_int = len(coeffs) - 1
                for i, coeff in enumerate(coeffs):
                    power = degree_int - i
                    coeff_str = f"{abs(coeff):.3g}"
                    if coeff < 0:
                        sign = " - "
                    elif i == 0:
                        sign = ""
                    else:
                        sign = " + "
                    # Форматирование члена полинома
                    if power == 0:
                        term = f"{sign}{coeff_str}"
                    elif power == 1:
                        term = f"{sign}{coeff_str}·x"
                    else:
                        term = f"{sign}{coeff_str}·x^{power}"
                    coeffs_text_parts.append(term)

                coeffs_text = "".join(coeffs_text_parts)
                equation = f"y = {coeffs_text}"

                # Подготовка коэффициентов для отображения
                coeffs_str = '\n'.join([f'Коэффициент при x^{degree_int - i}: {coeff:.4f}' for i, coeff in enumerate(coeffs)])

                # Сохраняем коэффициенты и уравнение
                self.coeffs_str = coeffs_str
                self.equation = equation

                # Отображение формулы и R^2 на графике
                self.ax.text(0.05, 0.95, equation, transform=self.ax.transAxes,
                             fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
                r_squared_text = f"R² = {r_squared:.4f}"
                self.ax.text(0.05, 0.85, r_squared_text, transform=self.ax.transAxes,
                             fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

            elif method == 'poisson':
                if len(x_fit) < 1:
                    messagebox.showerror("Ошибка", "Недостаточно точек для аппроксимации Пуассона.")
                    return

                # Предполагаем, что x - это дискретные значения (целые числа)
                x_fit_int = x_fit.astype(int)

                # Оцениваем параметр λ (среднее значение)
                lambda_estimate = np.mean(x_fit_int)
                self.r_squared = None  # R² не применим для Пуассоновской аппроксимации

                # Создаем функцию распределения Пуассона
                def poisson_pmf(k, lamb):
                    return (lamb ** k) * np.exp(-lamb) / factorial(k)

                # Построение аппроксимации
                x_plot = np.arange(np.min(x_fit_int), np.max(x_fit_int) + 1)
                y_plot = poisson_pmf(x_plot, lambda_estimate) * len(x_fit_int)  # Масштабируем по количеству точек

                self.ax.plot(x_plot, y_plot, 'o-', label='Аппроксимация Пуассона', color='purple')

                # Формирование формулы для отображения на графике
                equation = f"P(k; λ) = (λ^k * e^-λ) / k!"
                coeffs_str = f"Оценка λ: {lambda_estimate:.4f}"

                # Сохраняем коэффициенты и уравнение
                self.coeffs_str = coeffs_str
                self.equation = equation

                # Отображение параметра λ на графике
                self.ax.text(0.05, 0.95, f"λ = {lambda_estimate:.3g}", transform=self.ax.transAxes,
                             fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

            elif method == 'gauss':
                if len(x_fit) < 3:
                    messagebox.showerror("Ошибка", "Недостаточно точек для аппроксимации Гаусса.")
                    return

                # Определяем функцию Гаусса
                def gauss_function(x, A, mu, sigma):
                    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

                # Начальные приближения параметров
                initial_guess = [max(y_fit), x_fit[np.argmax(y_fit)], np.std(x_fit)]

                # Аппроксимация данных функцией Гаусса
                try:
                    popt, pcov = curve_fit(gauss_function, x_fit, y_fit, p0=initial_guess)
                except RuntimeError:
                    messagebox.showerror("Ошибка", "Не удалось подобрать параметры для аппроксимации Гаусса.")
                    return

                # Предсказанные значения
                y_fit_line = gauss_function(x_fit, *popt)

                # Вычисление R^2
                ss_res = np.sum((y_fit - y_fit_line) ** 2)
                ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                self.r_squared = r_squared

                # Построение аппроксимации
                x_plot = np.linspace(np.min(x_fit), np.max(x_fit), 500)
                y_plot = gauss_function(x_plot, *popt)
                self.ax.plot(x_plot, y_plot, label='Аппроксимация Гаусса', color='orange')

                # Формирование формулы для отображения на графике
                equation = f"y = {popt[0]:.3g}·exp(-((x - {popt[1]:.3g})² / (2·{popt[2]:.3g}²)))"

                # Подготовка коэффициентов для отображения
                coeffs_str = f"A (амплитуда): {popt[0]:.4f}\nμ (среднее): {popt[1]:.4f}\nσ (стандартное отклонение): {popt[2]:.4f}"

                # Сохраняем коэффициенты и уравнение
                self.coeffs_str = coeffs_str
                self.equation = equation

                # Отображение формулы и R^2 на графике
                self.ax.text(0.05, 0.95, equation, transform=self.ax.transAxes,
                             fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
                r_squared_text = f"R² = {r_squared:.4f}"
                self.ax.text(0.05, 0.85, r_squared_text, transform=self.ax.transAxes,
                             fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

            else:
                messagebox.showerror("Ошибка", f"Неизвестный метод аппроксимации: {method}")
                return

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при аппроксимации: {e}")
            print(f"Ошибка в perform_approximation: {e}")

    def show_coefficients(self):
        if self.coeffs_str:
            coeffs_window = tk.Toplevel(self.master)
            coeffs_window.title("Параметры аппроксимации")
            tk.Label(coeffs_window, text="Параметры аппроксимации:", font=('Arial', 12, 'bold')).pack(pady=10)
            coeffs_text_widget = tk.Text(coeffs_window, width=50, height=10)
            coeffs_text_widget.insert(tk.END, self.coeffs_str)
            coeffs_text_widget.config(state='disabled')  # Делаем текст недоступным для редактирования
            coeffs_text_widget.pack(pady=5)

            # Добавление кнопки "Копировать"
            def copy_coeffs():
                self.master.clipboard_clear()
                self.master.clipboard_append(self.coeffs_str)
                messagebox.showinfo("Копирование", "Коэффициенты скопированы в буфер обмена.")

            tk.Button(coeffs_window, text="Копировать", command=copy_coeffs).pack(pady=5)
            tk.Button(coeffs_window, text="Закрыть", command=coeffs_window.destroy).pack(pady=5)
        else:
            messagebox.showwarning("Предупреждение", "Коэффициенты аппроксимации отсутствуют. Сначала выполните аппроксимацию.")

import tkinter as tk
from tkinter import ttk, messagebox, colorchooser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import lagrange
from scipy.optimize import curve_fit
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings

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
            settings_frame = tk.Frame(self.master)
            settings_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

            # Выбор столбцов для осей X и Y
            tk.Label(settings_frame, text="Выберите столбец для оси X:").pack(pady=5)
            self.x_column = ttk.Combobox(settings_frame, values=list(self.data.columns))
            self.x_column.pack(pady=5)

            tk.Label(settings_frame, text="Выберите столбец для оси Y:").pack(pady=5)
            self.y_column = ttk.Combobox(settings_frame, values=list(self.data.columns))
            self.y_column.pack(pady=5)

            # Опции логарифмирования
            self.log_x_var = tk.BooleanVar()
            self.log_y_var = tk.BooleanVar()
            tk.Checkbutton(settings_frame, text="Логарифмировать X", variable=self.log_x_var).pack(pady=2)
            tk.Checkbutton(settings_frame, text="Логарифмировать Y", variable=self.log_y_var).pack(pady=2)

            # Выбор метода аппроксимации
            tk.Label(settings_frame, text="Выберите метод аппроксимации:").pack(pady=5)
            self.approx_method = tk.StringVar()
            self.approx_method.set('linear')
            methods = [("Линейная аппроксимация", 'linear'),
                       ("Полиномиальная аппроксимация", 'polyfit'),
                       ("Аппроксимация Лагранжа", 'lagrange'),
                       ("Аппроксимация Гаусса", 'gauss'),
                       ("Без аппроксимации", 'none')]
            for text, method in methods:
                tk.Radiobutton(settings_frame, text=text, variable=self.approx_method, value=method).pack(anchor=tk.W)

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
            tk.Label(settings_frame, text="Используйте 'x' как переменную. Пример: np.sin(x)").pack(pady=2)
            tk.Button(settings_frame, text="Добавить функцию", command=self.add_function).pack(pady=5)

            self.function_listbox = tk.Listbox(settings_frame)
            self.function_listbox.pack(pady=5)

            # Кнопки управления
            tk.Button(settings_frame, text="Построить график", command=self.plot_graph).pack(pady=10)
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

            # Обработка изменения метода аппроксимации
            self.approx_method.trace('w', self.on_approx_method_change)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка при создании виджетов: {e}")
            print(f"Ошибка в create_widgets: {e}")
            self.master.destroy()

    def on_approx_method_change(self, *args):
        method = self.approx_method.get()
        if method == 'polyfit':
            self.degree_label.config(state='normal')
            self.degree_entry.config(state='normal')
        else:
            self.degree_label.config(state='disabled')
            self.degree_entry.config(state='disabled')

    def add_function(self):
        func_text = self.function_entry.get()
        if func_text:
            color = colorchooser.askcolor(title="Выберите цвет для функции")[1]
            self.theoretical_functions.append({'func': func_text, 'color': color})
            self.function_listbox.insert(tk.END, func_text)
            self.function_entry.delete(0, tk.END)
        else:
            messagebox.showwarning("Пустое поле", "Введите функцию перед добавлением.")

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

            if x_col not in self.data.columns or y_col not in self.data.columns:
                messagebox.showerror("Ошибка", "Выберите корректные столбцы для осей X и Y.")
                return

            x = self.data[x_col].astype(float)
            y = self.data[y_col].astype(float)

            if self.log_x_var.get():
                x = np.log(x)
            if self.log_y_var.get():
                y = np.log(y)

            # Объединяем x и y в один массив для обнаружения выбросов
            X = np.column_stack((x, y))

            # Используем метод интерквартильного размаха (IQR)
            self.outliers_mask = self.detect_outliers_iqr(X)

            # Если выбросы найдены
            if np.any(self.outliers_mask):
                # Отобразить выбросы на графике
                self.outliers = set(np.where(self.outliers_mask)[0])
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
        if self.x is None or self.y is None:
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
        distances = np.hypot(self.x - x_click, self.y - y_click)
        min_index = np.argmin(distances)
        # Устанавливаем порог расстояния для выбора точки
        threshold = 0.05 * (self.ax.get_xlim()[1] - self.ax.get_xlim()[0])

        if distances[min_index] < threshold:
            # Переключаем состояние точки (выбрана/не выбрана)
            if min_index in self.selected_points:
                self.selected_points.remove(min_index)
                self.scatter._facecolors[min_index, :] = (0, 0, 1, 1)  # Синий цвет
            else:
                self.selected_points.add(min_index)
                self.scatter._facecolors[min_index, :] = (1, 0, 0, 1)  # Красный цвет
            self.canvas.draw()

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
        import tkinter.filedialog as fd
        file_path = fd.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
        if file_path:
            self.figure.savefig(file_path)
            messagebox.showinfo("Сохранение", f"График успешно сохранён в {file_path}")

    def plot_graph(self):
        try:
            x_col = self.x_column.get()
            y_col = self.y_column.get()

            if x_col not in self.data.columns or y_col not in self.data.columns:
                messagebox.showerror("Ошибка", "Выберите корректные столбцы для осей X и Y.")
                return

            if self.outliers_removed:
                # Используем данные без выбросов
                data = self.filtered_data
            else:
                data = self.data

            x = data[x_col].astype(float)
            y = data[y_col].astype(float)

            if self.log_x_var.get():
                x = np.log(x)
            if self.log_y_var.get():
                y = np.log(y)

            self.x = x.values
            self.y = y.values

            x_label = self.x_label_entry.get() or x_col
            y_label = self.y_label_entry.get() or y_col

            self.ax.clear()

            # Отображение данных
            self.scatter = self.ax.scatter(self.x, self.y, label='Данные', color='blue')

            # Если есть выбранные выбросы, выделяем их
            for idx in self.selected_points:
                if idx < len(self.scatter._facecolors):
                    self.scatter._facecolors[idx, :] = (1, 0, 0, 1)  # Красный цвет

            # Если есть автоматически обнаруженные выбросы, выделяем их
            for idx in self.outliers:
                if idx < len(self.scatter._facecolors):
                    self.scatter._facecolors[idx, :] = (1, 0, 0, 1)  # Красный цвет

            method = self.approx_method.get()
            if method == 'linear':
                # Учитываем удаленные выбросы
                if self.outliers_removed:
                    x_fit = self.x
                    y_fit = self.y
                else:
                    mask = np.ones(len(self.x), dtype=bool)
                    for idx in self.selected_points.union(self.outliers):
                        if idx < len(mask):
                            mask[idx] = False
                    x_fit = self.x[mask]
                    y_fit = self.y[mask]

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

                # Построение аппроксимации
                x_plot = np.linspace(np.min(x_fit), np.max(x_fit), 500)
                y_plot = np.polyval(coeffs, x_plot)
                self.ax.plot(x_plot, y_plot, label='Линейная аппроксимация', color='green')

                # Формирование формулы для отображения на графике
                equation = f"$y = {coeffs[0]:.3g}·x + {coeffs[1]:.3g}$"

                try:
                    self.ax.text(0.05, 0.95, equation, transform=self.ax.transAxes,
                                 fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
                except Exception as e:
                    print(f"Ошибка при отображении формулы: {e}")
                    messagebox.showerror("Ошибка", f"Не удалось отобразить формулу на графике: {e}")

                # Отображение коэффициента детерминации R^2
                r_squared_text = f"$R^2 = {r_squared:.4f}$"
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

                # Учитываем удаленные выбросы
                if self.outliers_removed:
                    x_fit = self.x
                    y_fit = self.y
                else:
                    mask = np.ones(len(self.x), dtype=bool)
                    for idx in self.selected_points.union(self.outliers):
                        if idx < len(mask):
                            mask[idx] = False
                    x_fit = self.x[mask]
                    y_fit = self.y[mask]

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

                # Построение аппроксимации
                self.ax.plot(x_plot, y_plot, label=f'Полиномиальная аппроксимация (степень {degree})', color='red')

                # Формирование формулы для отображения на графике
                coeffs_text_parts = []
                degree_int = len(coeffs) - 1
                for i, coeff in enumerate(coeffs):
                    power = degree_int - i
                    coeff_str = f"{coeff:.3g}"
                    if i > 0:
                        if coeff >= 0:
                            coeff_str = f"+ {coeff_str}"
                        else:
                            coeff_str = f"- {abs(coeff):.3g}"
                    # Форматирование члена полинома
                    if power == 0:
                        term = f"{coeff_str}"
                    elif power == 1:
                        term = f"{coeff_str}·x"
                    else:
                        term = f"{coeff_str}·x^{power}"
                    coeffs_text_parts.append(term)

                coeffs_text = " ".join(coeffs_text_parts)
                equation = f"$y = {coeffs_text}$"

                try:
                    self.ax.text(0.05, 0.95, equation, transform=self.ax.transAxes,
                                 fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
                except Exception as e:
                    print(f"Ошибка при отображении формулы: {e}")
                    messagebox.showerror("Ошибка", f"Не удалось отобразить формулу на графике: {e}")

                # Отображение коэффициента детерминации R^2
                r_squared_text = f"$R^2 = {r_squared:.4f}$"
                self.ax.text(0.05, 0.85, r_squared_text, transform=self.ax.transAxes,
                             fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

            elif method == 'lagrange':
                # Учитываем удаленные выбросы
                if self.outliers_removed:
                    x_fit = self.x
                    y_fit = self.y
                else:
                    mask = np.ones(len(self.x), dtype=bool)
                    for idx in self.selected_points.union(self.outliers):
                        if idx < len(mask):
                            mask[idx] = False
                    x_fit = self.x[mask]
                    y_fit = self.y[mask]

                if len(x_fit) < 2:
                    messagebox.showerror("Ошибка", "Недостаточно точек для аппроксимации Лагранжа.")
                    return

                poly = lagrange(x_fit, y_fit)
                x_new = np.linspace(np.min(x_fit), np.max(x_fit), 500)
                y_new = poly(x_new)

                # Вычисление предсказанных значений на обучающих точках
                y_pred = poly(x_fit)

                # Вычисление R^2
                ss_res = np.sum((y_fit - y_pred) ** 2)
                ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)

                self.ax.plot(x_new, y_new, label='Аппроксимация Лагранжа', color='purple')

                # Формирование формулы для отображения на графике
                equation = f"Полином Лагранжа (степень {len(poly.c) - 1})"

                try:
                    self.ax.text(0.05, 0.95, equation, transform=self.ax.transAxes,
                                 fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
                except Exception as e:
                    print(f"Ошибка при отображении формулы: {e}")
                    messagebox.showerror("Ошибка", f"Не удалось отобразить формулу на графике: {e}")

                # Отображение коэффициента детерминации R^2
                r_squared_text = f"$R^2 = {r_squared:.4f}$"
                self.ax.text(0.05, 0.85, r_squared_text, transform=self.ax.transAxes,
                             fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

            elif method == 'gauss':
                # Учитываем удаленные выбросы
                if self.outliers_removed:
                    x_fit = self.x
                    y_fit = self.y
                else:
                    mask = np.ones(len(self.x), dtype=bool)
                    for idx in self.selected_points.union(self.outliers):
                        if idx < len(mask):
                            mask[idx] = False
                    x_fit = self.x[mask]
                    y_fit = self.y[mask]

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

                # Построение аппроксимации
                x_plot = np.linspace(np.min(x_fit), np.max(x_fit), 500)
                y_plot = gauss_function(x_plot, *popt)
                self.ax.plot(x_plot, y_plot, label='Аппроксимация Гаусса', color='orange')

                # Формирование формулы для отображения на графике
                equation = f"$y = {popt[0]:.3g}·e^{{-((x - {popt[1]:.3g})^2 / (2·{popt[2]:.3g}^2))}}$"

                try:
                    self.ax.text(0.05, 0.95, equation, transform=self.ax.transAxes,
                                 fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
                except Exception as e:
                    print(f"Ошибка при отображении формулы: {e}")
                    messagebox.showerror("Ошибка", f"Не удалось отобразить формулу на графике: {e}")

                # Отображение коэффициента детерминации R^2
                r_squared_text = f"$R^2 = {r_squared:.4f}$"
                self.ax.text(0.05, 0.85, r_squared_text, transform=self.ax.transAxes,
                             fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

            # Построение теоретических функций
            for func_item in self.theoretical_functions:
                func_text = func_item['func']
                color = func_item['color']
                x_func = np.linspace(np.min(self.x), np.max(self.x), 500)
                try:
                    allowed_names = {'np': np, 'x': x_func}
                    y_func = eval(func_text, {"__builtins__": None}, allowed_names)
                    self.ax.plot(x_func, y_func, label=f'Функция: {func_text}', color=color)
                except Exception as e:
                    messagebox.showerror("Ошибка", f"Некорректная функция '{func_text}': {e}")
                    return

            # Настройка графика
            self.ax.set_xlabel(x_label)
            self.ax.set_ylabel(y_label)
            self.ax.set_title(f"График {y_label} от {x_label}")
            self.ax.legend()
            self.ax.grid(True)

            # Отображение графика
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка при построении графика: {e}")
            print(f"Ошибка в plot_graph: {e}")

    def add_function(self):
        func_text = self.function_entry.get()
        if func_text:
            color = colorchooser.askcolor(title="Выберите цвет для функции")[1]
            if color:
                self.theoretical_functions.append({'func': func_text, 'color': color})
                self.function_listbox.insert(tk.END, func_text)
                self.function_entry.delete(0, tk.END)
            else:
                messagebox.showwarning("Пустое поле", "Выберите цвет перед добавлением функции.")
        else:
            messagebox.showwarning("Пустое поле", "Введите функцию перед добавлением.")

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

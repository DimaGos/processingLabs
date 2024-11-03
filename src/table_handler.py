from tkinter import ttk, Toplevel, messagebox, filedialog, simpledialog
import tkinter as tk
import pandas as pd
import numpy as np
import os
from collections import deque  # Для реализации функции undo

# Предполагаем, что plotter.py находится в той же директории и содержит класс Plotter
from plotter import Plotter  # Импортируем класс Plotter для открытия окна построения графика

class TableHandler:
    def __init__(self, master, data):
        self.master = master
        self.master.title("Обработка таблицы")
        self.master.geometry("1200x800")
        self.master.resizable(True, True)

        self.data = data.copy()           # Текущие данные (возможно отфильтрованные)
        self.filtered_data = None         # Данные после фильтрации

        # Стек для хранения истории изменений (для undo)
        self.undo_stack = deque(maxlen=20)  # Ограничиваем размер стека для экономии памяти
        self.save_state()  # Сохраняем начальное состояние

        # Флаг для отслеживания, были ли данные отфильтрованы
        self.is_filtered = False

        # Создаем фрейм для таблицы и инструментов
        self.main_frame = ttk.Frame(self.master)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Создаем таблицу
        self.create_table()

        # Создаем панель инструментов
        self.create_tools()

        # Настраиваем стиль интерфейса
        self.style = ttk.Style()
        self.style.theme_use('clam')  # Вы можете выбрать другую тему по вашему вкусу

    def create_table(self):
        """Создает и отображает таблицу с данными."""
        # Создаем фрейм для таблицы и полос прокрутки
        table_frame = ttk.Frame(self.main_frame)
        table_frame.pack(fill=tk.BOTH, expand=True)

        # Создаем вертикальную и горизонтальную полосы прокрутки
        vsb = ttk.Scrollbar(table_frame, orient="vertical")
        hsb = ttk.Scrollbar(table_frame, orient="horizontal")

        # Создаем Treeview
        self.tree = ttk.Treeview(
            table_frame,
            columns=list(self.data.columns),
            show="headings",
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set
        )

        # Конфигурируем полосы прокрутки
        vsb.config(command=self.tree.yview)
        hsb.config(command=self.tree.xview)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self.tree.pack(fill="both", expand=True)

        # Устанавливаем заголовки колонок и их ширину
        for column in self.data.columns:
            self.tree.heading(column, text=column, command=lambda _col=column: self.sort_column(_col, False))
            # Автоматически устанавливаем ширину колонки на основе максимальной ширины содержимого
            try:
                max_width = max(
                    self.data[column].astype(str).map(len).max(),
                    len(column)
                ) * 10  # Примерно 10 пикселей на символ
            except:
                max_width = 100
            self.tree.column(column, width=max_width, anchor="center")

        # Заполняем таблицу данными
        self.populate_table()

        # Связываем двойной клик для редактирования ячеек
        self.tree.bind('<Double-1>', self.on_double_click)

    def populate_table(self):
        """Заполняет таблицу данными."""
        # Очищаем текущие данные в таблице
        for row in self.tree.get_children():
            self.tree.delete(row)

        # Вставляем новые данные
        for index, row in self.data.iterrows():
            values = list(row)
            # Преобразуем все значения в строки для отображения
            values = [str(item) if pd.notnull(item) else '' for item in values]
            self.tree.insert("", "end", iid=str(index), values=values)

    def create_tools(self):
        """Создает панель инструментов для обработки таблицы."""
        tools_frame = ttk.Frame(self.master)
        tools_frame.pack(fill=tk.X, padx=10, pady=5)

        # Используем Grid для размещения кнопок в несколько рядов
        row = 0
        column = 0
        padx = 5
        pady = 5

        # Кнопка для добавления строки
        add_row_button = ttk.Button(tools_frame, text="Добавить строку", command=self.add_row)
        add_row_button.grid(row=row, column=column, padx=padx, pady=pady, sticky='nsew')
        column += 1

        # Кнопка для удаления выбранных строк
        delete_row_button = ttk.Button(tools_frame, text="Удалить выбранные строки", command=self.delete_selected_rows)
        delete_row_button.grid(row=row, column=column, padx=padx, pady=pady, sticky='nsew')
        column += 1

        # Кнопка для добавления столбца
        add_col_button = ttk.Button(tools_frame, text="Добавить столбец", command=self.add_column)
        add_col_button.grid(row=row, column=column, padx=padx, pady=pady, sticky='nsew')
        column += 1

        # Кнопка для удаления выбранных столбцов
        delete_col_button = ttk.Button(tools_frame, text="Удалить выбранные столбцы", command=self.delete_selected_columns)
        delete_col_button.grid(row=row, column=column, padx=padx, pady=pady, sticky='nsew')
        column += 1

        # Кнопка для округления значений
        round_button = ttk.Button(tools_frame, text="Округлить значения", command=self.round_values)
        round_button.grid(row=row, column=column, padx=padx, pady=pady, sticky='nsew')
        column += 1

        # Кнопка для отмены последних изменений
        undo_button = ttk.Button(tools_frame, text="Отменить изменение", command=self.undo_change)
        undo_button.grid(row=row, column=column, padx=padx, pady=pady, sticky='nsew')
        column += 1

        # Переходим на новую строку
        row += 1
        column = 0

        # Кнопка для применения пользовательской функции
        apply_func_button = ttk.Button(tools_frame, text="Применить функцию", command=self.apply_custom_function)
        apply_func_button.grid(row=row, column=column, padx=padx, pady=pady, sticky='nsew')
        column += 1

        # Кнопка для вычисления статистических параметров
        stats_button = ttk.Button(tools_frame, text="Вычислить статистику", command=self.calculate_statistics)
        stats_button.grid(row=row, column=column, padx=padx, pady=pady, sticky='nsew')
        column += 1

        # Кнопка для фильтрации данных
        filter_button = ttk.Button(tools_frame, text="Фильтровать данные", command=self.filter_data)
        filter_button.grid(row=row, column=column, padx=padx, pady=pady, sticky='nsew')
        column += 1

        # Кнопка для сброса фильтра
        self.reset_filter_button = ttk.Button(tools_frame, text="Сбросить фильтр", command=self.reset_filter)
        self.reset_filter_button.grid(row=row, column=column, padx=padx, pady=pady, sticky='nsew')
        self.reset_filter_button.config(state=tk.DISABLED)  # Изначально кнопка отключена
        column += 1

        # Кнопка для сохранения изменений
        save_button = ttk.Button(tools_frame, text="Сохранить изменения", command=self.save_changes)
        save_button.grid(row=row, column=column, padx=padx, pady=pady, sticky='nsew')
        column += 1

        # Переходим на новую строку
        row += 1
        column = 0

        # Кнопка для построения графика
        plot_data_button = ttk.Button(tools_frame, text="Построить график", command=self.open_plotter)
        plot_data_button.grid(row=row, column=column, padx=padx, pady=pady, sticky='nsew')
        column += 1

        # Расширяем колонки и строки
        for i in range(column):
            tools_frame.columnconfigure(i, weight=1)
        for i in range(row + 1):
            tools_frame.rowconfigure(i, weight=1)

    def open_plotter(self):
        """Открывает окно построения графика с текущими данными."""
        try:
            plotter_root = tk.Toplevel(self.master)
            plotter_app = Plotter(plotter_root, self.data.copy())
            # Не вызываем plotter_root.mainloop(), так как основной цикл уже запущен
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть окно построения графика: {e}")

    def on_double_click(self, event):
        """Обрабатывает двойной клик для редактирования ячеек."""
        item = self.tree.identify_row(event.y)
        column = self.tree.identify_column(event.x)
        if not item or not column:
            return

        x, y, width, height = self.tree.bbox(item, column)
        column_index = int(column.replace('#', '')) - 1
        column_name = self.data.columns[column_index]

        value = self.tree.set(item, column_name)

        # Создаем Entry виджет для редактирования
        self.edit_box = ttk.Entry(self.tree)
        self.edit_box.place(x=x, y=y, width=width, height=height)
        self.edit_box.insert(0, value)
        self.edit_box.focus()

        # Связываем события для завершения редактирования
        self.edit_box.bind("<Return>", lambda e: self.save_edit(item, column_name))
        self.edit_box.bind("<FocusOut>", lambda e: self.cancel_edit())

    def save_edit(self, item, column_name):
        """Сохраняет изменения из Entry виджета в таблицу и DataFrame."""
        new_value = self.edit_box.get()
        self.tree.set(item, column_name, new_value)
        # Обновляем DataFrame
        try:
            # Пытаемся привести к числовому типу, если возможно
            self.data.at[int(item), column_name] = pd.to_numeric(new_value, errors='ignore')
        except Exception:
            self.data.at[int(item), column_name] = new_value
        self.edit_box.destroy()
        self.save_state()

    def cancel_edit(self):
        """Отменяет редактирование."""
        self.edit_box.destroy()

    def add_row(self):
        """Добавляет новую строку в таблицу и DataFrame."""
        new_index = len(self.data)
        empty_row = {col: '' for col in self.data.columns}
        # Используем pd.concat для добавления новой строки
        new_row_df = pd.DataFrame([empty_row])
        self.data = pd.concat([self.data, new_row_df], ignore_index=True)
        values = list(empty_row.values())
        self.tree.insert("", "end", iid=str(new_index), values=values)
        self.save_state()

    def delete_selected_rows(self):
        """Удаляет выбранные строки из таблицы и DataFrame."""
        selected_items = self.tree.selection()
        if not selected_items:
            messagebox.showwarning("Предупреждение", "Выберите строки для удаления.")
            return
        confirm = messagebox.askyesno("Подтверждение", "Вы уверены, что хотите удалить выбранные строки?")
        if not confirm:
            return
        for item in selected_items:
            self.tree.delete(item)
            self.data = self.data.drop(int(item))
        self.data.reset_index(drop=True, inplace=True)
        self.populate_table()
        self.save_state()

    def add_column(self):
        """Добавляет новый столбец в таблицу и DataFrame."""
        new_col = simpledialog.askstring("Добавить столбец", "Введите название нового столбца:")
        if not new_col:
            return
        if new_col in self.data.columns:
            messagebox.showerror("Ошибка", "Столбец с таким названием уже существует.")
            return
        self.data[new_col] = ''
        self.update_treeview_columns()
        messagebox.showinfo("Успех", f"Столбец '{new_col}' успешно добавлен.")
        self.save_state()

    def delete_selected_columns(self):
        """Удаляет выбранные столбцы из таблицы и DataFrame."""
        selected_columns = self.get_selected_columns()
        if not selected_columns:
            messagebox.showwarning("Предупреждение", "Выберите столбцы для удаления.")
            return
        confirm = messagebox.askyesno("Подтверждение", "Вы уверены, что хотите удалить выбранные столбцы?")
        if not confirm:
            return
        self.data.drop(columns=selected_columns, inplace=True)
        self.update_treeview_columns()
        messagebox.showinfo("Успех", "Выбранные столбцы успешно удалены.")
        self.save_state()

    def get_selected_columns(self):
        """Возвращает список выбранных столбцов."""
        selected = []
        # Открываем диалог для выбора столбцов
        columns = self.data.columns.tolist()
        select_window = Toplevel(self.master)
        select_window.title("Выбор столбцов для удаления")
        select_window.geometry("300x400")
        select_window.transient(self.master)  # Устанавливаем родительское окно
        select_window.grab_set()

        ttk.Label(select_window, text="Выберите столбцы для удаления:", font=('Arial', 12)).pack(pady=10)

        selected_vars = {col: tk.BooleanVar() for col in columns}
        for col in columns:
            ttk.Checkbutton(select_window, text=col, variable=selected_vars[col]).pack(anchor=tk.W, padx=20)

        def confirm_selection():
            for col, var in selected_vars.items():
                if var.get():
                    selected.append(col)
            select_window.destroy()

        ttk.Button(select_window, text="Удалить", command=confirm_selection).pack(pady=10)

        self.master.wait_window(select_window)
        return selected

    def round_values(self):
        """Округляет значения в выбранных столбцах до заданного количества знаков."""
        columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if not columns:
            messagebox.showwarning("Предупреждение", "Нет числовых столбцов для округления.")
            return

        # Открываем диалог для выбора столбцов и количества знаков
        round_window = Toplevel(self.master)
        round_window.title("Округление значений")
        round_window.geometry("400x400")
        round_window.transient(self.master)  # Устанавливаем родительское окно
        round_window.grab_set()

        ttk.Label(round_window, text="Выберите столбцы для округления:", font=('Arial', 12)).pack(pady=10)

        selected_vars = {col: tk.BooleanVar() for col in columns}
        for col in columns:
            ttk.Checkbutton(round_window, text=col, variable=selected_vars[col]).pack(anchor=tk.W, padx=20)

        ttk.Label(round_window, text="Количество знаков после запятой:", font=('Arial', 12)).pack(pady=10)
        decimals_entry = ttk.Entry(round_window)
        decimals_entry.pack(pady=5)

        def apply_rounding():
            try:
                decimals = int(decimals_entry.get())
                if decimals < 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Ошибка", "Введите корректное целое неотрицательное число для количества знаков.")
                return

            selected_cols = [col for col, var in selected_vars.items() if var.get()]
            if not selected_cols:
                messagebox.showwarning("Предупреждение", "Выберите хотя бы один столбец для округления.")
                return

            self.data[selected_cols] = self.data[selected_cols].round(decimals)
            self.populate_table()
            self.save_state()
            messagebox.showinfo("Успех", "Значения успешно округлены.")
            round_window.destroy()

        ttk.Button(round_window, text="Округлить", command=apply_rounding).pack(pady=10)

    def undo_change(self):
        """Отменяет последнее изменение."""
        if len(self.undo_stack) > 1:
            # Удаляем текущее состояние
            self.undo_stack.pop()
            # Восстанавливаем предыдущее состояние
            self.data = self.undo_stack[-1].copy()
            self.update_treeview_columns()
            self.populate_table()
            messagebox.showinfo("Отмена", "Последнее изменение отменено.")
        else:
            messagebox.showwarning("Предупреждение", "Нет действий для отмены.")

    def save_state(self):
        """Сохраняет текущее состояние данных для возможности отмены изменений."""
        self.undo_stack.append(self.data.copy())

    def apply_custom_function(self):
        """Позволяет пользователю применить собственную функцию к данным."""
        func_window = Toplevel(self.master)
        func_window.title("Применить пользовательскую функцию")
        func_window.geometry("600x500")
        func_window.transient(self.master)  # Устанавливаем родительское окно
        func_window.grab_set()

        ttk.Label(func_window, text="Введите код для обработки данных:", font=('Arial', 12)).pack(pady=10)
        ttk.Label(func_window, text="Используйте переменную 'data' для обращения к таблице.", wraplength=580).pack(pady=5)
        ttk.Label(func_window, text="Доступные модули: numpy (np), pandas (pd).", wraplength=580).pack(pady=2)

        # Создаем вкладки для ввода кода и примеров
        notebook = ttk.Notebook(func_window)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Вкладка для ввода кода
        code_frame = ttk.Frame(notebook)
        notebook.add(code_frame, text="Код")

        func_entry = tk.Text(code_frame, height=20, width=70)
        func_entry.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        # Вкладка с примерами
        examples_frame = ttk.Frame(notebook)
        notebook.add(examples_frame, text="Примеры")

        examples = [
            "1. Добавить новый столбец:\n   data['V'] = data['x'] ** 2",
            "2. Изменить существующий столбец:\n   data['A'] = np.log(data['A'])",
            "3. Удалить строки с пропущенными значениями:\n   data.dropna(inplace=True)",
            "4. Фильтровать данные:\n   data = data[data['A'] > 5]",
            "5. Группировка данных:\n   data = data.groupby('Category').sum().reset_index()",
            "6. Применение функции к столбцу:\n   data['A'] = data['A'].apply(lambda x: x**2)",
            "7. Сортировка данных:\n   data.sort_values('A', inplace=True)",
            "8. Объединение столбцов:\n   data['FullName'] = data['FirstName'] + ' ' + data['LastName']",
            "9. Создание индикатора на основе условия:\n   data['IsHigh'] = np.where(data['A'] > 10, 1, 0)",
            "10. Вычисление кумулятивной суммы:\n    data['CumSum'] = data['A'].cumsum()"
        ]
        examples_text = '\n\n'.join(examples)
        examples_label = tk.Text(examples_frame, height=20, width=70)
        examples_label.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        examples_label.insert(tk.END, examples_text)
        examples_label.config(state=tk.DISABLED)

        def apply_function():
            expr = func_entry.get("1.0", tk.END).strip()
            if not expr:
                messagebox.showwarning("Предупреждение", "Введите код для обработки данных.")
                return
            try:
                # Выполняем выражение в контексте DataFrame
                exec(expr, {"data": self.data, "np": np, "pd": pd})
                self.update_treeview_columns()
                self.populate_table()
                self.save_state()
                messagebox.showinfo("Успех", "Функция успешно применена.")
                func_window.destroy()
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось выполнить функцию: {e}")

        ttk.Button(func_window, text="Применить", command=apply_function).pack(pady=10)

    def calculate_statistics(self):
        """Вычисляет статистические параметры и отображает их в отдельной таблице."""
        stats_window = Toplevel(self.master)
        stats_window.title("Вычислить статистику")
        stats_window.geometry("600x800")
        stats_window.transient(self.master)  # Устанавливаем родительское окно
        stats_window.grab_set()

        ttk.Label(stats_window, text="Выберите статистические параметры для вычисления:", font=('Arial', 12)).pack(pady=10)

        # Список доступных статистических функций
        stats_functions = {
            'Среднее (Mean)': 'mean',
            'Медиана (Median)': 'median',
            'Стандартное отклонение (Std)': 'std',
            'Дисперсия (Variance)': 'var',
            'Сумма (Sum)': 'sum',
            'Минимум (Min)': 'min',
            'Максимум (Max)': 'max',
            'Размах (Range)': lambda x: x.max() - x.min(),
            'Коэффициент асимметрии (Skewness)': 'skew',
            'Куртозис (Kurtosis)': 'kurt',
            '25-й процентиль': lambda x: x.quantile(0.25),
            '50-й процентиль (Медиана)': lambda x: x.quantile(0.50),
            '75-й процентиль': lambda x: x.quantile(0.75),
            'Мода (Mode)': lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
            'Количество ненулевых значений (Count Non-Null)': 'count',
            'Количество уникальных значений (Unique Count)': lambda x: x.nunique(),
        }

        # Чекбоксы для выбора статистических параметров
        stats_vars = {}
        for stat_name in stats_functions.keys():
            var = tk.BooleanVar()
            stats_vars[stat_name] = var
            ttk.Checkbutton(stats_window, text=stat_name, variable=var).pack(anchor=tk.W, padx=20)

        # Выбор столбцов для статистики
        ttk.Label(stats_window, text="Выберите столбцы для вычисления статистики:", font=('Arial', 12)).pack(pady=10)

        numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_columns:
            messagebox.showwarning("Предупреждение", "В таблице нет числовых столбцов для вычисления статистики.")
            stats_window.destroy()
            return

        column_vars = {}
        for col in numeric_columns:
            var = tk.BooleanVar()
            column_vars[col] = var
            ttk.Checkbutton(stats_window, text=col, variable=var).pack(anchor=tk.W, padx=40)

        # Добавляем опцию добавить результаты в таблицу
        self.add_to_table_var = tk.BooleanVar()
        ttk.Checkbutton(stats_window, text="Добавить результаты в таблицу", variable=self.add_to_table_var).pack(pady=10)

        def perform_stats():
            # Получаем выбранные статистики
            selected_stats = {stat_name: func for stat_name, func in stats_functions.items() if stats_vars[stat_name].get()}
            if not selected_stats:
                messagebox.showwarning("Предупреждение", "Выберите хотя бы один статистический параметр.")
                return

            # Получаем выбранные столбцы
            selected_cols = [col for col, var in column_vars.items() if var.get()]
            if not selected_cols:
                messagebox.showwarning("Предупреждение", "Выберите хотя бы один столбец для вычисления статистики.")
                return

            # Создаем DataFrame для результатов
            results_df = pd.DataFrame(index=selected_stats.keys(), columns=selected_cols)

            # Вычисляем статистики
            for stat_name, func in selected_stats.items():
                for col in selected_cols:
                    try:
                        if callable(func):
                            result = func(self.data[col])
                        else:
                            result = getattr(self.data[col], func)()
                        results_df.at[stat_name, col] = result
                    except Exception as e:
                        results_df.at[stat_name, col] = np.nan

            # Создаем новое окно для отображения статистики
            self.show_statistics_results(results_df)
            self.stats_df = results_df  # Сохраняем DataFrame как атрибут класса

            # Если выбрано добавление результатов в таблицу
            if self.add_to_table_var.get():
                self.add_statistics_to_table(results_df)

            messagebox.showinfo("Успех", "Статистические параметры успешно вычислены.")
            stats_window.destroy()

        ttk.Button(stats_window, text="Вычислить", command=perform_stats).pack(pady=10)

    def show_statistics_results(self, results_df):
        """Отображает результаты статистики в новом окне."""
        self.stats_df = results_df  # Сохраняем DataFrame как атрибут класса

        result_window = Toplevel(self.master)
        result_window.title("Результаты статистики")
        result_window.geometry("800x600")
        result_window.resizable(True, True)
        result_window.transient(self.master)  # Устанавливаем родительское окно
        result_window.grab_set()

        # Создаем Treeview для отображения результатов
        tree = ttk.Treeview(result_window, columns=["Статистика"] + list(results_df.columns), show="headings")
        vsb = ttk.Scrollbar(result_window, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(result_window, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        tree.pack(fill="both", expand=True)

        tree.heading("Статистика", text="Статистика")
        tree.column("Статистика", width=200, anchor="center")

        for col in results_df.columns:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor="center")

        # Заполняем Treeview результатами
        for stat_name in results_df.index:
            values = [stat_name] + [
                round(results_df.at[stat_name, col], 4) if pd.notnull(results_df.at[stat_name, col]) else 'NaN' for col
                in results_df.columns]
            tree.insert("", "end", values=values)

        # Добавляем кнопку для сохранения статистики
        save_stats_button = ttk.Button(result_window, text="Сохранить статистику", command=self.save_statistics)
        save_stats_button.pack(pady=10)

    def add_statistics_to_table(self, results_df):
        """Добавляет результаты статистики в таблицу."""
        # Транспонируем DataFrame для удобства
        transposed_df = results_df.T
        # Добавляем новый столбец 'Статистика' для названия статистики
        transposed_df.insert(0, 'Статистика', transposed_df.index)
        # Добавляем новые строки в self.data с использованием pd.concat
        new_rows = transposed_df.reset_index(drop=True)
        self.data = pd.concat([self.data, new_rows], ignore_index=True)
        self.update_treeview_columns()
        self.populate_table()
        self.save_state()

    def save_statistics(self):
        """Сохраняет результаты статистики в файл."""
        try:
            stats_df = self.stats_df  # Используем сохраненный DataFrame

            # Запросить место сохранения
            save_path = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[
                    ("Excel Files", "*.xlsx"),
                    ("CSV Files", "*.csv"),
                    ("Text Files", "*.txt"),
                    ("All Files", "*.*")]
            )
            if not save_path:
                return  # Пользователь отменил сохранение

            # Сохранить DataFrame в выбранный файл
            file_extension = os.path.splitext(save_path)[1].lower()
            if file_extension == '.xlsx':
                stats_df.to_excel(save_path, index=True, engine='openpyxl')
            elif file_extension == '.csv':
                stats_df.to_csv(save_path, index=True)
            elif file_extension == '.txt':
                stats_df.to_csv(save_path, index=True, sep='\t')
            else:
                # Если расширение неизвестно, предлагаем сохранить как Excel
                stats_df.to_excel(save_path, index=True, engine='openpyxl')

            messagebox.showinfo("Успех", f"Статистические данные успешно сохранены в {save_path}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить статистические данные: {e}")

    def filter_data(self):
        """Позволяет пользователю фильтровать данные по заданному условию."""
        filter_window = Toplevel(self.master)
        filter_window.title("Фильтрация данных")
        filter_window.geometry("500x300")
        filter_window.transient(self.master)  # Устанавливаем родительское окно
        filter_window.grab_set()

        ttk.Label(filter_window, text="Введите условие фильтрации:", font=('Arial', 12)).pack(pady=10)
        ttk.Label(filter_window, text="Например: y > 7 and x < 5", wraplength=480).pack(pady=5)
        ttk.Label(filter_window, text="Доступные переменные: названия столбцов данных.", wraplength=480).pack(pady=5)

        condition_entry = ttk.Entry(filter_window, width=60)
        condition_entry.pack(pady=10)

        def apply_filter():
            condition = condition_entry.get().strip()
            if not condition:
                messagebox.showwarning("Предупреждение", "Введите условие фильтрации.")
                return
            try:
                # Используем метод query для фильтрации данных
                filtered_data = self.data.query(condition)
                if filtered_data.empty:
                    messagebox.showinfo("Результат", "Фильтрация не вернула никаких результатов.")
                    return
                # Сохраняем отфильтрованные данные
                self.filtered_data = self.data.copy()  # Сохраняем текущее состояние для сброса
                self.data = filtered_data.reset_index(drop=True)
                self.is_filtered = True
                self.reset_filter_button.config(state=tk.NORMAL)
                self.populate_table()
                self.save_state()
                messagebox.showinfo("Успех", "Фильтрация успешно применена.")
                filter_window.destroy()
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось применить фильтр: {e}")

        ttk.Button(filter_window, text="Применить фильтр", command=apply_filter).pack(pady=10)

    def reset_filter(self):
        """Возвращает данные до фильтрации."""
        if self.is_filtered:
            self.data = self.filtered_data.copy()
            self.is_filtered = False
            self.reset_filter_button.config(state=tk.DISABLED)
            self.update_treeview_columns()
            self.populate_table()
            self.save_state()
            messagebox.showinfo("Сброс фильтра", "Фильтр успешно сброшен. Отображены данные до фильтрации.")
        else:
            messagebox.showwarning("Предупреждение", "Фильтр не был применен.")

    def sort_column(self, col, reverse):
        """Сортирует столбец и обновляет таблицу."""
        try:
            # Пытаемся привести столбец к числовому типу
            self.data[col] = pd.to_numeric(self.data[col], errors='ignore')
            self.data.sort_values(by=col, ascending=not reverse, inplace=True)
            self.data.reset_index(drop=True, inplace=True)
            self.populate_table()
            self.save_state()
            # Переключаем порядок сортировки для следующего раза
            self.tree.heading(col, command=lambda _col=col: self.sort_column(_col, not reverse))
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось отсортировать столбец: {e}")

    def save_changes(self):
        """Сохраняет изменения в данные и обновляет таблицу."""
        try:
            # Открываем диалог для выбора места сохранения
            save_path = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[
                    ("Excel Files", "*.xlsx"),
                    ("CSV Files", "*.csv"),
                    ("Text Files", "*.txt"),
                    ("All Files", "*.*")]
            )
            if not save_path:
                return  # Пользователь отменил сохранение

            file_extension = os.path.splitext(save_path)[1].lower()
            if file_extension == '.xlsx':
                self.data.to_excel(save_path, index=False, engine='openpyxl')
            elif file_extension == '.csv':
                self.data.to_csv(save_path, index=False)
            elif file_extension == '.txt':
                self.data.to_csv(save_path, index=False, sep='\t')
            else:
                # Если расширение неизвестно, предложим сохранить как Excel
                self.data.to_excel(save_path, index=False, engine='openpyxl')

            messagebox.showinfo("Успех", f"Данные успешно сохранены в {save_path}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {e}")

    def update_treeview_columns(self):
        """Обновляет колонки Treeview после изменений в DataFrame."""
        self.tree['columns'] = list(self.data.columns)
        # Удаляем все существующие заголовки
        for col in self.tree["columns"]:
            self.tree.heading(col, text=col, command=lambda _col=col: self.sort_column(_col, False))
            # Автоматически устанавливаем ширину колонки на основе максимальной ширины содержимого
            try:
                max_width = max(
                    self.data[col].astype(str).map(len).max(),
                    len(col)
                ) * 10  # Примерно 10 пикселей на символ
            except:
                max_width = 100
            self.tree.column(col, width=max_width, anchor="center")
        self.populate_table()

    def get_data(self):
        """Возвращает текущий DataFrame."""
        return self.data.copy()
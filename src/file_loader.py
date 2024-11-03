import tkinter as tk
from tkinter import messagebox, ttk, Toplevel, simpledialog
import pandas as pd
import easygui  # Используем easygui вместо filedialog для выбора файла

def load_file_and_show_table():
    filepath = easygui.fileopenbox(
        title="Выберите файл данных",
        filetypes=["*.csv", "*.xlsx", "*.xls", "*.txt"]
    )

    if not filepath:
        messagebox.showinfo("Информация", "Файл не был выбран.")
        return None

    try:
        file_extension = filepath.split('.')[-1].lower()

        if file_extension == 'csv':
            data = pd.read_csv(filepath, sep=None, engine='python')
        elif file_extension in ['xlsx', 'xls']:
            data = pd.read_excel(filepath, engine='openpyxl' if file_extension == 'xlsx' else 'xlrd')
        elif file_extension == 'txt':
            # Определяем количество столбцов из первой строки
            with open(filepath, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                if not first_line:
                    messagebox.showerror("Ошибка", "Файл пустой.")
                    return None
                num_columns = len(first_line.strip().split())
                column_names = [f"Column {i+1}" for i in range(num_columns)]
            data = pd.read_csv(
                filepath,
                sep=r'\s+|\t+|,',  # Разделители: пробелы, табуляция или запятая
                engine='python',
                header=None,  # Нет заголовка
                names=column_names,
                decimal=','
            )
        else:
            messagebox.showwarning("Ошибка", "Неподдерживаемый формат файла.")
            return None

        data = data.where(pd.notnull(data), None)  # Заполняем пустые значения None

        # Проверка, что data.columns не пустые
        if len(data.columns) == 0:
            messagebox.showerror("Ошибка", "Файл не содержит столбцов.")
            return None

        # Убедимся, что все названия столбцов являются строками и не пустые
        data.columns = [str(col).strip() if str(col).strip() else f"Column {i+1}" for i, col in enumerate(data.columns)]

        # Проверка на уникальность названий столбцов
        if len(data.columns) != len(set(data.columns)):
            messagebox.showwarning("Предупреждение", "Названия столбцов не уникальны. Переименуйте столбцы.")
            renamed_data = rename_columns_window(data)
            if renamed_data is None:
                # Пользователь отменил переименование столбцов
                return None
        else:
            # Предлагаем переименовать столбцы для уверенности
            renamed_data = rename_columns_window(data)
            if renamed_data is None:
                # Пользователь отменил переименование столбцов
                return None

        # Проверка, что после переименования названия столбцов корректны
        if renamed_data is None or renamed_data.empty:
            messagebox.showerror("Ошибка", "Не удалось переименовать столбцы.")
            return None

        show_table_window(renamed_data)  # Отображаем таблицу
        return renamed_data

    except Exception as e:
        messagebox.showerror("Ошибка", f"Ошибка при загрузке файла: {e}")
        return None

def rename_columns_window(data):
    """Открывает окно для переименования столбцов и возвращает обновлённые данные."""
    rename_window = Toplevel()
    rename_window.title("Переименование столбцов")
    rename_window.geometry("600x600")
    rename_window.grab_set()
    rename_window.transient()
    rename_window.focus_set()

    ttk.Label(rename_window, text="Измените названия столбцов:", font=('Arial', 12)).pack(pady=10)

    # Создаем фрейм для размещения полей ввода
    input_frame = ttk.Frame(rename_window)
    input_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

    # Добавляем прокрутку, если столбцов много
    canvas = tk.Canvas(input_frame)
    scrollbar = ttk.Scrollbar(input_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    entries = {}
    for col in data.columns:
        frame = ttk.Frame(scrollable_frame)
        frame.pack(pady=5, padx=5, fill=tk.X)

        ttk.Label(frame, text=col, width=20).pack(side=tk.LEFT)
        new_name_var = tk.StringVar(value=col)
        entry = ttk.Entry(frame, textvariable=new_name_var)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        entries[col] = new_name_var

    # Объект для хранения возвращаемых данных
    return_data = [None]

    def apply_renaming():
        new_names = [var.get().strip() for var in entries.values()]
        # Проверка на уникальность
        if len(new_names) != len(set(new_names)):
            messagebox.showerror("Ошибка", "Названия столбцов должны быть уникальными.")
            return
        # Проверка на отсутствие пустых названий
        if any(not name for name in new_names):
            messagebox.showerror("Ошибка", "Названия столбцов не могут быть пустыми.")
            return
        try:
            data.columns = new_names
            return_data[0] = data
            rename_window.destroy()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось изменить названия столбцов: {e}")

    def cancel():
        rename_window.destroy()
        return_data[0] = None

    # Создаем фрейм для кнопок
    button_frame = ttk.Frame(rename_window)
    button_frame.pack(pady=20)

    apply_button = ttk.Button(button_frame, text="Применить", command=apply_renaming)
    apply_button.pack(side=tk.LEFT, padx=10)

    cancel_button = ttk.Button(button_frame, text="Отмена", command=cancel)
    cancel_button.pack(side=tk.LEFT, padx=10)

    rename_window.wait_window()

    return return_data[0]

def show_table_window(data):
    """Открывает новое окно и отображает данные в виде таблицы с возможностью редактирования."""
    if data.empty:
        messagebox.showinfo("Информация", "В таблице нет данных для отображения.")
        return

    table_window = Toplevel()
    table_window.title("Просмотр и редактирование данных")
    table_window.geometry("1000x700")
    table_window.resizable(True, True)

    # Фрейм для кнопок управления
    control_frame = ttk.Frame(table_window)
    control_frame.pack(side=tk.TOP, fill=tk.X)

    delete_row_button = ttk.Button(control_frame, text="Удалить выбранные строки", command=lambda: delete_rows(tree, data))
    delete_row_button.pack(side=tk.LEFT, padx=5, pady=5)

    delete_col_button = ttk.Button(control_frame, text="Удалить выбранный столбец", command=lambda: delete_column(tree, data))
    delete_col_button.pack(side=tk.LEFT, padx=5, pady=5)

    save_button = ttk.Button(control_frame, text="Сохранить изменения", command=lambda: save_changes(data))
    save_button.pack(side=tk.LEFT, padx=5, pady=5)

    # Создаем Treeview
    tree = ttk.Treeview(table_window, columns=list(data.columns), show="headings", selectmode='extended')
    tree.pack(fill="both", expand=True)

    # Прокрутка
    vsb = ttk.Scrollbar(tree, orient="vertical", command=tree.yview)
    hsb = ttk.Scrollbar(tree, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

    vsb.pack(side="right", fill="y")
    hsb.pack(side="bottom", fill="x")

    for column in data.columns:
        tree.heading(column, text=column, command=lambda c=column: treeview_sort_column(tree, c, False))
        tree.column(column, width=100, anchor="center")

    # Заполняем данными
    for index, row in data.iterrows():
        values = list(row)
        values = [str(item) if item is not None else '' for item in values]
        tree.insert("", "end", iid=str(index), values=values)

    # Обработка двойного клика для редактирования ячейки
    tree.bind('<Double-1>', lambda event: edit_cell(event, tree, data))

def edit_cell(event, tree, data):
    """Редактирование значения ячейки при двойном клике."""
    item = tree.identify('item', event.x, event.y)
    column = tree.identify_column(event.x)
    if not item or not column:
        return

    x, y, width, height = tree.bbox(item, column)
    value = tree.item(item, 'values')[int(column[1:]) - 1]

    # Создаем Entry для редактирования
    entry = tk.Entry(tree)
    entry.place(x=x, y=y, width=width, height=height)
    entry.insert(0, value)
    entry.focus_set()

    def on_focus_out(event):
        new_value = entry.get()
        entry.destroy()
        # Обновляем значение в Treeview
        values = list(tree.item(item, 'values'))
        values[int(column[1:]) - 1] = new_value
        tree.item(item, values=values)
        # Обновляем значение в DataFrame
        row_index = int(item)
        col_name = tree.heading(column)['text']
        data.at[row_index, col_name] = new_value if new_value != '' else None

    entry.bind('<Return>', lambda e: on_focus_out(e))
    entry.bind('<FocusOut>', lambda e: on_focus_out(e))

def delete_rows(tree, data):
    """Удаляет выбранные строки из таблицы и DataFrame."""
    selected_items = tree.selection()
    if not selected_items:
        messagebox.showwarning("Предупреждение", "Пожалуйста, выберите строки для удаления.")
        return

    for item in selected_items:
        row_index = int(item)
        tree.delete(item)
        data.drop(row_index, inplace=True)

    data.reset_index(drop=True, inplace=True)

    # Обновляем идентификаторы элементов в Treeview
    tree.delete(*tree.get_children())
    for index, row in data.iterrows():
        values = list(row)
        values = [str(item) if item is not None else '' for item in values]
        tree.insert("", "end", iid=str(index), values=values)

def delete_column(tree, data):
    """Удаляет выбранный столбец из таблицы и DataFrame."""
    # Открываем диалоговое окно для выбора столбца
    columns = list(data.columns)
    if not columns:
        messagebox.showwarning("Предупреждение", "Нет столбцов для удаления.")
        return

    col_to_delete = simpledialog.askstring("Удалить столбец", f"Введите название столбца для удаления:\nДоступные столбцы: {', '.join(columns)}")
    if col_to_delete not in columns:
        messagebox.showerror("Ошибка", "Некорректное название столбца.")
        return

    # Удаляем столбец из DataFrame
    data.drop(columns=[col_to_delete], inplace=True)

    # Обновляем Treeview
    tree['columns'] = list(data.columns)
    for col in tree['columns']:
        tree.heading(col, text=col)
        tree.column(col, width=100, anchor="center")

    # Обновляем данные в Treeview
    tree.delete(*tree.get_children())
    for index, row in data.iterrows():
        values = list(row)
        values = [str(item) if item is not None else '' for item in values]
        tree.insert("", "end", iid=str(index), values=values)

def save_changes(data):
    """Сохраняет изменения в DataFrame."""
    messagebox.showinfo("Сохранение", "Изменения сохранены.")

def treeview_sort_column(tv, col, reverse):
    """Сортирует столбец при нажатии на заголовок."""
    data_list = [(tv.set(k, col), k) for k in tv.get_children('')]
    data_list.sort(reverse=reverse)

    for index, (val, k) in enumerate(data_list):
        tv.move(k, '', index)

    tv.heading(col, command=lambda: treeview_sort_column(tv, col, not reverse))

# Для запуска интерфейса
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    load_file_and_show_table()
    root.mainloop()
import pandas as pd

# Definir los nombres de las columnas base
column_names = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']

# Definir los nombres de los archivos de entrada
input_files = ['lp1.txt', 'lp2.txt', 'lp3.txt', 'lp4.txt', 'lp5.txt']  # Lista de archivos a procesar

# Ruta del archivo de salida
output_excel_path = 'Data_four_classes.xlsx'

# Inicializar una lista para almacenar los DataFrames de cada archivo
all_data = []

# Definir cuántos grupos de mediciones quieres almacenar por fila (15 en este caso)
num_groups = 15

# Procesar cada archivo de entrada
for input_file_path in input_files:
    # Inicializar variables para almacenar los datos
    data = []
    labels = []
    original_types = []  # Esta lista almacenará el tipo original de cada línea
    current_label = None
    current_row = []
    line1 = None

    # Leer y procesar el archivo de texto
    with open(input_file_path, 'r') as file:
        for line in file:
            line = line.strip()
            # Verificar si la línea contiene una etiqueta de clase
            if line in ['normal', 'ok']:
                current_label = 'No Error'
                line1 = line
            elif line in ['obstruction', 'bottom_obstruction']:
                current_label = 'obstruction'
                line1 = line
            elif line in ['fr_collision', 'front_col', 'right_col', 'left_col','lost',
                          'back_col', 'collision', 'collision_in_part', 'collision_in_tool', 'bottom_collision']:
                current_label = 'severe_collision'
                line1 = line
            elif line in ['moved', 'slightly_moved']:
                current_label = 'mild_collision'
                line1 = line
            elif line in []:
                current_label = None
                line1 = line
            else:
                # Intentar convertir la línea en valores numéricos
                if current_label and line1:
                    try:
                        values = list(map(float, line.split()))
                        if len(values) == 6:
                            current_row.extend(values)  # Añadir las mediciones a la fila actual
                            if len(current_row) == 6 * num_groups:  # Verificar si se han alcanzado las 15 mediciones
                                # Añadir columnas correspondientes a las clases
                                current_row.append(1 if current_label == 'No Error' else 0)
                                current_row.append(1 if current_label == 'obstruction' else 0)
                                current_row.append(1 if current_label == 'severe_collision' else 0)
                                current_row.append(1 if current_label == 'mild_collision' else 0)
                                data.append(current_row)  # Guardar la fila completa
                                current_row = []  # Reiniciar la fila
                    except ValueError:
                        # Ignorar líneas que no contienen datos numéricos válidos
                        continue

    # Crear los nombres de las columnas dinámicamente
    dynamic_columns = []
    for i in range(1, num_groups + 1):
        for col in column_names:
            dynamic_columns.append(f'{col}{i}')

    # Añadir las columnas de clase ('No Error', 'obstruction', 'col_robot_transfer', 'col_arm_motion')
    dynamic_columns.extend(['No Error', 'obstruction', 'severe_collision', 'mild_collision'])

    # Crear un DataFrame con los datos
    df = pd.DataFrame(data, columns=dynamic_columns)

    # Añadir el DataFrame a la lista
    all_data.append(df)

# Concatenar todos los DataFrames
combined_data = pd.concat(all_data, ignore_index=True)

# Guardar el DataFrame combinado en un archivo Excel
combined_data.to_excel(output_excel_path, index=False)

print(f"El archivo procesado ha sido guardado como '{output_excel_path}'.")
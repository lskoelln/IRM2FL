import os

def generate_folder(folder, level=0):
    strings = folder.split('/')
    strings = [item for item in strings if item != '']

    if (level > 0):
        print("Wrong value for level in generate_folder. Return without action.")
        return

    if len(strings) <= abs(level):
        return

    for n in range(len(strings) + level):
        new_folder = '/'.join(strings[:n + 1])
        if not os.path.exists(new_folder): os.makedirs(new_folder)
import os

i_dir = "/home/asr/projects/tatar-tts-2/VITS/vits/Almaz"

def create_path_map(source_dir):
    path_map = {}
    for root, dirs,files in os.walk(source_dir):
        for file in files:
            if file.endswith(".wav"):
                path_map[file] = os.path.join(root, file)
    return path_map

path_map = create_path_map('/home/asr/projects/tatar-tts-2/VITS/vits/Almaz')
print(path_map)

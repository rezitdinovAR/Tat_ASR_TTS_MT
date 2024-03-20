# Load the data from the csv file
import pandas as pd
import os
import random

random.seed(42)

dataset_name = "AA44_base"
data = pd.read_csv(f"/home/ips/projects/tatar-tts-2/VITS/vits/filelists/{dataset_name}.csv", sep='\t')
data = data.drop(['Unnamed: 0'], axis=1)
print(data.head())

# Support for DataFrames
def split_file_list(orig_data: pd.DataFrame, train_ratio=None, test_samples=None, max_samples=None):
    # Shuffle the data
    data = orig_data.sample(frac=1).reset_index(drop=True)

    if max_samples is not None:
        data = data[:max_samples]

    if test_samples is not None:
        train_set = data[:-test_samples]
        test_set = data[-test_samples:]
    elif train_ratio is not None:
        train_set_size = int(len(data) * train_ratio)
        train_set = data[:train_set_size]
        test_set = data[train_set_size:]

    else:
        raise ValueError("Either 'train_ratio' or 'test_samples' should be provided.")

    return train_set, test_set


# Example usage
train_data, val_data = split_file_list(data, test_samples=240)

i_dir = "/home/ips/projects/tatar-tts-2/VITS/vits/Almaz2"
o_file_train = f"/home/ips/projects/tatar-tts-2/VITS/vits/filelists/{dataset_name}_audio_sid_text_train_filelist.txt"
o_file_val = f"/home/ips/projects/tatar-tts-2/VITS/vits/filelists/{dataset_name}_audio_sid_text_test_filelist.txt"

link_name = "DUMMY4"

def create_path_map(source_dir):
    path_map = {}
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".wav"):
                path_map[file] = os.path.join(root, file)
    return path_map


def save_file_list(data, out_file_path, source_dir, path_map, link_name, cleaned_text=False):
    with open(out_file_path, "w") as file:
        for row in data.itertuples():
            uttid = f"{row.uttid}.wav"
            path = path_map[uttid].replace(source_dir, link_name)
            spkidx = row.spkidx
            info = row.text if not cleaned_text else row.phonemes

            file.write(f"{path}|{spkidx}|{info}\n")
            # Print every nth sample
            if row.Index % 5000 == 0:
                print(f"{row.Index}: {path}|{spkidx}|{info}")

    print(f"Saved to '{out_file_path}' ({len(data)} samples).")


def save_files(data, out_file_path, source_dir, path_map, link_name):
    save_file_list(train_data, out_file_path, source_dir, path_map, link_name)
    if "phonemes" in data.columns:
        out_file_path = out_file_path.replace(".txt", ".txt.cleaned")
        save_file_list(data, out_file_path, source_dir,
                       path_map, link_name, cleaned_text=True)
        
path_map = create_path_map(i_dir)


save_files(train_data, o_file_train, i_dir, path_map, link_name)
save_files(val_data, o_file_val, i_dir, path_map, link_name)

# Create symlink to the dataset
os.system(f"ln -s {i_dir} {link_name}")

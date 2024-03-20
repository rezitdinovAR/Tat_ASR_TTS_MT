from pydub import AudioSegment
import os

def change_file_16(input_filepath, output_filepath):
    # Read the wav file
    sound = AudioSegment.from_wav(input_filepath)

    # Set the sample width to 2 bytes (16-bit)
    sound = sound.set_sample_width(2)

    # Export the sound to a new wav file
    sound.export(output_filepath, format='wav')

for dirpath, dirnames, filenames in os.walk('/home/ips/projects/tatar-tts-2/VITS/vits/Almaz2'):
    count = 0
    for filename in filenames:
        if filename.endswith('.wav'):
            #print(f"/home/ips/projects/tatar-tts-2/VITS/vits/Almaz2/{filename}")
            change_file_16(f"/home/ips/projects/tatar-tts-2/VITS/vits/Almaz2/{filename}", f"/home/ips/projects/tatar-tts-2/VITS/vits/Almaz/{filename}")
            count += 1
    print(count)

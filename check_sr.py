import os
import sys
import wave

_, folder_path = sys.argv

for file_name in os.listdir(folder_path):
	if file_name.endswith('.wav'):
		with wave.open(folder_path + '/' + file_name, "rb") as wave_file:
			frame_rate = wave_file.getframerate()
			# if frame_rate != 16000:
			print(frame_rate)
	else:
		print(file_name, 'is not wav')

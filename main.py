import pyaudio
import wave
import struct
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os
from scipy.fftpack import fft
from scipy.io import wavfile

chunk = 1024*4  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 1
fs = 44100  # Record at 44100 samples per second
seconds = 8
filename = r"C:\Users\Monali Fernando\Desktop\wave files\output.wav"
# sr, data = wavfile.read(r"C:\Users\Monali Fernando\Desktop\wave files\output.wav")

p = pyaudio.PyAudio()  # Create an interface to PortAudio

print('Recording frequency...')

stream = p.open(
                format=sample_format,
                channels=channels,
                rate=fs,
                frames_per_buffer=chunk,
                input=True,
                output=True)

frames = []  # Initialize array to store frames
# Save the recorded data as a WAV file
fig, ax = plt.subplots()

                # x= np.arange(0, 2*chunk, 2)
                # line, = ax.plot(x, np.random.rand(chunk))
                # ax.set_ylim(0,255)
                # ax.set_ylim(0,chunk)
                #
                # while True:
                #     data = stream.read(chunk)
                #     data_int= np.array(struct.unpack(str(2*chunk) +'B',data),dtype='b')[::2]+127
                #     line.set_ydata(data_int)
                #     fig.canvas.draw()
                #     fig.canvas.flush_events()

# Store data in chunks for 3 seconds
for i in range(0, int(fs / chunk * seconds)):
    data = stream.read(chunk)
    frames.append(data)
    len(data)
    data_int = struct.unpack(str(2*chunk)+ 'B', data)

print('Finished recording\nOutput wave file will be saved!')
stream.stop_stream()
stream.close()
p.terminate()

plt.xlabel('Time (Seconds)')
plt.ylabel('Amplitude')
plt.title("Captured Signal")
            # signal = np.arange(10, 550, 3);
            # # getting the amplitude of the signal
            # signalAmplitude = np.sin(signal)
            # # plotting the signal
            # ax.plot(signal, signalAmplitude, color='blue')
            # # plt.plot(data_int, '-')

wav =wave.open(r"C:\Users\Monali Fernando\Desktop\wave files\output.wav")
#---
raw=wav.readframes(-1)
raw=np.frombuffer(raw,"Int16")
sampleRate = wav.getframerate()

if wav.getnchannels() == 2:
    print("file is not supported")
    sys.exit(0)
Time = np.linspace(0,len(raw)/sampleRate, num=len(raw))
plt.title("Baby cry signal")
plt.plot(Time, raw, color="blue")
plt.ylabel("Amplitude")
plt.savefig(r"C:\Users\Monali Fernando\Desktop\Cry spectrum images\output")
plt.show()
#----

wf = wave.open(filename, 'wb')
wf.setnchannels(channels)
wf.setsampwidth(p.get_sample_size(sample_format))
wf.setframerate(fs)
wf.writeframes(b''.join(frames))
wf.close()

#****************************************************************
#finding similarity

original = cv2.imread(r"D:\Research progress\Cry freq\hunger.png")
original2 = cv2.imread(r"D:\Research progress\Cry freq\discom.png")

#cv2.imshow("Original image ",original)

sift = cv2.xfeatures2d.SIFT_create()
kp_1, desc_1 = sift.detectAndCompute(original, None)
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Load all the images
all_images_to_compare = []
titles = []

for f in glob.iglob(r"C:\Users\Monali Fernando\Desktop\Cry spectrum images\*"):
        image = cv2.imread(f)
        titles.append(f)
        all_images_to_compare.append(image)

        for image_to_compare, title in zip(all_images_to_compare, titles):
            # Check if 2 images are equals
            if original.shape == image_to_compare.shape:
                print("The images have same size and channels")
                difference = cv2.subtract(original, image_to_compare)
                b, g, r = cv2.split(difference)
                if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
                    print("Similarity: 100% (equal size and channels)")


            # Check for similarities between the 2 images
            kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)
            matches = flann.knnMatch(desc_1, desc_2, k=2)
            good_points = []
            for m, n in matches:
                if m.distance > 0.6 * n.distance:
                    good_points.append(m)
            number_keypoints = 0
            if len(kp_1) >= len(kp_2):
                number_keypoints = len(kp_1)
            else:
                number_keypoints = len(kp_2)
            print("Image: " + title)
            percentage_similarity = len(good_points) / number_keypoints * 100
            # print(len(good_points) / number_keypoints)
            # if (len(good_points) / number_keypoints * 100 >= 60)
                 # print("The baby feels hungry..")
            # else
                 # print("The baby feels discomfort..")
            not_similar =  100 - (len(good_points) / number_keypoints * 100)
            print("Similarity: " + str(int(percentage_similarity)) + "% | Differences: "+ str(int(not_similar)) + "%\n")

cv2.waitKey(0)
cv2.destroyAllWindows()


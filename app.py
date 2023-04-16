import streamlit as st
from scipy.io.wavfile import read
import io
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torchaudio
import sounddevice as sd
from scipy.io.wavfile import write


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, 3)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.conv4_bn = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        #print(self.pool.shape)
        self.fc1 = nn.Linear(64 * 1 * 11, 128)
        self.fc2 = nn.Linear(128, 4)
        self.dropOut = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = self.dropOut(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = self.dropOut(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.dropOut(x)
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = self.dropOut(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = self.dropOut(x)
        x = F.max_pool2d(x, 2, 2)
        #print(x.shape)
        #print(x.shape)
        x = x.view(-1, 64 * 1 * 11)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device('cpu')
model = Net()
#model.to(device)
model.load_state_dict(torch.load('tone_cnn_2.pth'))
model.to(device)
#model.eval()


st.set_page_config(
page_title = "Mandarin Tone Classifier",
page_icon = ":pencil:",
)


hide_streamlit_style = """
                       <style>
                       #MainMenu {visibility: hidden;}
                       footer {visibility: hidden;}
                       </style>
                       """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("Mandarin Tone Classifier App")


if st.button('Record'):
    fs = 44100  # Sample rate
    seconds = 1   # Duration of recording
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    write('output.mp3', fs, myrecording)  # Save as WAV file
    audio_file = open('output.mp3', 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/mp3')
    waveform, rate = torchaudio.load("output.mp3")
    train_audio_transforms = nn.Sequential( torchaudio.transforms.MFCC(
    sample_rate = rate, n_mfcc=60) )

    spec = train_audio_transforms(waveform)
    #audio_bytes = audio_recorder()
    #spec= spec.cuda()
    p2d = (0,220-spec.size(2))
    spec = F.pad(spec, p2d, "constant", 0).unsqueeze(0)
    preds = model(spec)
    st.write(f"Preds: {preds}")
    st.write(f"Tone: {int(torch.argmax(preds, dim = 1))+1}")






    #st.audio(audio_bytes, format="audio/wav")

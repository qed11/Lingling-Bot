## from youtube to mp3 or wac
from youtube_dl import YoutubeDL
from moviepy.editor import *
import os

## from youtube to mp4 to mp3
def download_one_audio(url):
    # download mp4 in the os directory and return the title as well as duration
    os.chdir("/Users/sarinaxi/Desktop/Lingling-Bot/Downloads/Videos/")
    ydl_opts = {
        'format': 'mp4',
        'outtmpl': '%(title)s.mp4',
    }

    audio_loader = YoutubeDL()
    with YoutubeDL(ydl_opts) as ydl:
        vid = ydl.download([url])
        info = ydl.extract_info(url, download = False)

    title = info['title']
    duration = info["duration"]
    title = title.replace(":", " -")
    video = VideoFileClip(title + ".mp4")
    os.chdir("/Users/sarinaxi/Desktop/Lingling-Bot/Downloads/Audios/")
    video.audio.write_audiofile(title + ".mp3")

    return title, duration

# URL = input('Enter youtube url:')
# title, duration = download_one_audio(URL)

## read csv file with youtube links at once
import pandas as pd
import numpy as np

def download_csv_audio(csv_path):
    # directly read csv file formatted with title "link" and a column of youtube links
    # Note you can directly export excel and sheets files into csv, an example Book1.csv is provided
    os.chdir("/Users/sarinaxi/Desktop/Lingling-Bot/CSV/")
    col_list = ["link"]
    df = pd.read_csv(csv_path, usecols=col_list)
    dict = {}
    links = df["link"]
    os.chdir("/Users/sarinaxi/Desktop/Lingling-Bot/Downloads/Videos/")
    for i in links:
       title, duration = download_one_audio(i)

download_csv_audio("Book1.csv")

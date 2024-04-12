import boto3
import streamlit as st
from contextlib import closing
import os
import sys
import subprocess
from tempfile import gettempdir

# Session = boto3.Session(
#         aws_access_key_id = st.secrets['aws_access_key_id'],
#         aws_secret_access_key = st.secrets['aws_secret_access_key'],
#         region_name = "us-east-1"
#     )

import os
# from pathlib import Path
from openai import OpenAI


def synthesize_speech(text):
    client = OpenAI(api_key="#####")
    response = client.audio.speech.create(
      model="tts-1-hd",
      voice="alloy",
      input=text
    )
    # Directly access the audio data
    audio_data = response.content
    output = os.path.join(gettempdir(), "speech.mp3")
    # Write the audio data to a file
    with open(output, "wb") as file:
        file.write(audio_data)
    return output


    


def synthesize_speech_2(text):
    client = OpenAI(api_key="#####")
    response = client.audio.speech.create(
      model="tts-1",
      voice="alloy",
      input=text
    )
    if "AudioStream" in response:
        # Note: Closing the stream is important because the service throttles on the
        # number of parallel connections. Here we are using contextlib.closing to
        # ensure the close method of the stream object will be called automatically
        # at the end of the with statement's scope.
        with closing(response["AudioStream"]) as stream:
            output = os.path.join(gettempdir(), "speech.mp3")

            try:
                # Open a file for writing the output as a binary stream
                with open(output, "wb") as file:
                    file.write(stream.read())
            except IOError as error:
                # Could not write to file, exit gracefully
                print(error)
                sys.exit(-1)
    else:
        # The response didn't contain audio data, exit gracefully
        print("Could not stream audio")
        sys.exit(-1)
    '''
    # Play the audio using the platform's default player
    if sys.platform == "win32":
        os.startfile(output)
    else:
        # The following works on macOS and Linux. (Darwin = mac, xdg-open = linux).
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, output])'''
    return output
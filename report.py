import re
import soundfile as sf
import os
mysp=__import__("my-voice-analysis")
import sys
import librosa

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import random

# Configuration
REPO_PATH = r"C:\Users\Arpan Kumar\Downloads\GPTInterviewer-main (3)\GPTInterviewer-main\GPTInterviewer-main\my-voice-analysis"  # Replace with the actual path to the repo
AUDIO_FILE = "fullaudio.wav" 
AUDIO_FILE_PATH = r"C:\Users\Arpan Kumar\Downloads\GPTInterviewer-main (3)\GPTInterviewer-main\GPTInterviewer-main\temp\fullaudio.wav"
CHUNK_DURATION_SECONDS = 5
MYSP_LIBRARY_PATH = "path/to/mysp/library"  # If needed

class OutputCatcher:
    def __init__(self):
        self.last_output = []

    def write(self, data: str):
        self.last_output.append(data)
        if self.last_output.__len__() == 11:
            self.last_output.pop(0)
        sys.__stdout__.write(data)
    
    def flush(self):  # Add the flush method
        sys.__stdout__.flush()  # Delegate flushing to the original stdout

catcher = OutputCatcher()

def resample_audio(y, orig_sr, target_sr=44000):
  
  # Resample the audio to the target sample rate
  y_resampled = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)

  return y_resampled, target_sr

def split_audio_and_process_chunks(audio_file_path, chunk_duration_seconds, mysp_library_path):
    """Splits audio, saves chunks in tempfiles, and runs mysp.mysptotal on each."""
    data, samplerate = sf.read(audio_file_path)
    data1 = [x[0] for x in data]
    data1, samplerate = resample_audio(np.asarray(data1), samplerate)
    data1  = data1.tolist()
    # data2 = [x[0] for x in data]
    chunk_size = chunk_duration_seconds * samplerate
    total_time = len(data1) / samplerate
    queue = []
    speech_rate = []
    pitch = []
    posteriori_probability_score = -1
    for i in range(0, len(data1), chunk_size):
        end_limit = min(i+chunk_size, len(data1)-1)
        chunk_data = data1[i:end_limit]
        queue.append(chunk_data)
        try:
            with open(os.path.join(REPO_PATH, "temp.wav"), "w") as temp_file:
                sf.write(temp_file.name, chunk_data, samplerate)
                file_title = os.path.split(temp_file.name)[-1]
                file_title = os.path.splitext(file_title)[0]
                file_path = os.path.split(temp_file.name)[:-1]
                file_path = os.path.join(*file_path)

                mysp.myspsr(file_title, file_path)
                last_ten_outputs = catcher.last_output[-10:]

                result_idx = last_ten_outputs.index("rate_of_speech=") + 2
                result = last_ten_outputs[result_idx]
                result = int(result) * 60
                speech_rate.append(result)
                # print(result)
        except ValueError:
                speech_rate.append(np.mean(speech_rate) if speech_rate else 0)


    for i in range(0, len(data1), chunk_size):
        end_limit = min(i+chunk_size, len(data1)-1)
        chunk_data = data1[i:end_limit]
        with open(os.path.join(REPO_PATH, "temp.wav"), "w") as temp_file:
            try:
                sf.write(temp_file.name, chunk_data, samplerate)
                file_title = os.path.split(temp_file.name)[-1]
                file_title = os.path.splitext(file_title)[0]
                file_path = os.path.split(temp_file.name)[:-1]
                file_path = os.path.join(*file_path)

                mysp.myspf0mean(file_title, file_path)
                last_ten_outputs = catcher.last_output[-10:]

                result_idx = last_ten_outputs.index("f0_mean=") + 2
                result = last_ten_outputs[result_idx]
                result = float(result)
                pitch.append(result)
            except ValueError:
                pitch.append(np.mean(pitch) if pitch else 0)

    with open(os.path.join(REPO_PATH, "temp.wav"), "w") as temp_file:
        try:
            sf.write(temp_file.name, data1, samplerate)
            file_title = os.path.split(temp_file.name)[-1]
            file_title = os.path.splitext(file_title)[0]
            file_path = os.path.split(temp_file.name)[:-1]
            file_path = os.path.join(*file_path)

            mysp.mysppron(file_title, file_path)
            last_ten_outputs = catcher.last_output[-10:]
            print(last_ten_outputs)

            for line in last_ten_outputs:
                if "Pronunciation_posteriori_probability_score_percentage=" in line:
                    # Extract the score using a regular expression
                    match = re.search(r'= :(\d+\.\d+)', line)
                    if match:
                        posteriori_probability_score = float(match.group(1))
                        break  # Stop searching once you find the score
                    else:
                        print("Score format not recognized")
                        posteriori_probability_score = -1
                        break
            else:  # 'else' clause of the for loop executes if no break occurred
                print("Score not found")
                posteriori_probability_score = -1
        except ValueError:
            print("Score not found")
            posteriori_probability_score = -1
    return speech_rate, pitch, total_time, posteriori_probability_score

def create_range_array(start_value, array_size, step=1):
    return [start_value + i * step for i in range(array_size)]

def generate_report():
    sys.stdout = catcher


    # --- Main Execution ---
    speech_rate, pitch, time, posteriori_probability_score = \
        split_audio_and_process_chunks(AUDIO_FILE_PATH, CHUNK_DURATION_SECONDS, MYSP_LIBRARY_PATH) 

    # Streamlit title and layout
    st.title("Your Rehearsal Report (Preview)")

    col1, col2, col3 = st.columns(3)

    filler_words = ["umm", "hmm", "oh"]
    sensitive_phrases = []

    time_series  = [5*i for i in range(10)]

    # speech_rate = [random.randint(160, 340) for _ in range(10)]
    speech_rate_mean = np.round(np.mean(speech_rate), 2)
    speech_rate_std = np.round(np.std(speech_rate), 2)
    speech_rate_range = np.max(speech_rate) - np.min(speech_rate)
    upper_speech_rate = 360
    lower_speech_rate = 180

    # pitch = [random.randint(1800, 2300) for _ in range(10)]
    pitch_mean = np.round(np.mean(pitch), 2)
    pitch_std = np.round(np.std(pitch), 2)
    pitch_range = np.max(pitch) - np.min(pitch)

    # Summary section (col1)
    with col1:
        st.header("Summary")
        st.write("Good job rehearsing! Keep up the hard work.")
        st.header(f"{int(time//60)}:{int(time%60)}")
        st.caption("Total  \ntime spent (in min)")
        
        # st.write(f"Pace: {speech_rate_mean} syllables/min")

        # Fillers section
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        # st.write("")
        # st.write("")
        st.header("Pronunciation Score")
        st.header(f"{posteriori_probability_score:0.2f}%")
        # st.caption(f"")

        # Learn More buttons
        st.button("Learn More")

    # Pace and Pitch sections (col2)
    with col2:
        st.header("Speech Rate")
        st.write(f"Average: {speech_rate_mean} syllables/min")
        st.write(f"Variation: ±{speech_rate_std} syllables/min")
        fig, ax = plt.subplots()
        ax.plot(create_range_array(1, len(speech_rate)), speech_rate, linewidth=2, color="green") 
        ax.axhspan(lower_speech_rate, upper_speech_rate, color='#b9feb9', alpha=0.75, lw=0)
        ax.set_title("Speech Rate Variation Over Time")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Words per Minute")
        ax.set_ylim(
            min(min(speech_rate) - .5 * speech_rate_range, lower_speech_rate - .5 * speech_rate_range),
            max(max(speech_rate) + .5 * speech_rate_range, upper_speech_rate + .5 * speech_rate_range),
        )
        st.pyplot(fig)

        st.write("")
        st.write("")
        st.header("Pace")
        if speech_rate_mean >= lower_speech_rate and speech_rate_mean <= upper_speech_rate:
            st.success("Your pace is just right! Keep it up!")
        elif speech_rate_mean < lower_speech_rate:
            st.error("You are speaking too slowly.")
        else:
            st.error("You need to speak a bit slow to be clear.")

        # Pace chart (using Matplotlib)
        fig, ax = plt.subplots()
        colors = ['#F5F5F5', '#00FF40', 'white']
        ax.pie([540 - speech_rate_mean, speech_rate_mean, 540], colors=colors)
        ax.add_artist(plt.Circle((0, 0), 0.8, color='white'))
        ax.text(0, 0.3, f"{speech_rate_mean:0.1f}", 
            ha='center',  va='center', fontsize=40, fontweight="heavy")
        ax.text(0, 0.3, f"\n\n\n\nsyllables/min", 
            ha='center',  va='center', fontsize=10)
        
        # Labels for 0, 180, 360, 540
        radius = 1.1  # Radius for placing the labels slightly outside the chart
        for angle in [0, 180, 360, 540]:
            theta = np.deg2rad(angle)  # Convert to radians, offset for start angle
            x = - radius * np.cos(theta // 3 + np.deg2rad(5))
            y = radius * np.sin(theta // 3 + np.deg2rad(5))
            ax.annotate(str(angle), xy=(x, y), va='center', ha='center', fontsize=12)

        st.pyplot(fig)

    with col3:
        st.header("Pitch")
        # st.write(f"Mean: {pitch_mean} Hz")
        # st.write(f"Variation: ±{pitch_std} Hz")
        st.write(f"Low pitch variation will make your audience lose interest. Try increasing the tone\
                for your key points")
        # Pitch chart (using Matplotlib)
        fig, ax = plt.subplots()
        ax.plot(create_range_array(1, len(pitch)), pitch, linewidth=2)
        ax.axhspan(min(pitch), max(pitch), color='#AFDBF5', alpha=0.75, lw=0)
        ax.set_title("Pitch Variation Over Time")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Pitch (Hz)")
        ax.set_ylim(
            min(pitch) - 1.5 * pitch_range,
            max(pitch) + 1.5 * pitch_range,
        )
        st.pyplot(fig)

        # # Sensitive Phrases section
        # st.header("Sensitive Phrases")
        # if sensitive_phrases == []:
        #     st.success(
        #         "No sensitive phrases found. Great job using inclusive speech!",
        #         icon="✅",
        #     )
        # else:
        #     st.write(
        #         "Avoid using insensitive language. Don't use words like:"
        #     )
        #     all_sensitive_phrases = ""
        #     for i, phrase in enumerate(sensitive_phrases):
        #         all_sensitive_phrases += phrase
        #         if i < len(sensitive_phrases) - 2:
        #             all_sensitive_phrases += ", "
        #         if i == len(sensitive_phrases) - 2 and len(sensitive_phrases) >= 2:
        #             all_sensitive_phrases += " and "
        #     st.error(all_sensitive_phrases)
        # st.button("Learn More", key="last")

    # Rehearse Again button
    st.button("Rehearse Again")
import cv2
import numpy as np
from pydub import AudioSegment
import moviepy.editor as mp
import pandas as pd

def add_sound_to_video(video_path, audio_path, csv_path):
    # Charger la vidéo
    video = mp.VideoFileClip(video_path)
    
    # Charger l'audio
    sound = AudioSegment.from_file(audio_path)
    
    # Lire le fichier CSV contenant les données de confiance
    confidence_data = pd.read_csv(csv_path)['confidence'].values
    
    # Normaliser les valeurs de confiance entre 0 et 100
    normalized_volumes = (confidence_data * 100).astype(int)
    
    # Appliquer le volume à chaque frame du son
    fps = video.fps
    total_frames = int(video.duration * fps)
    
    result_sound = AudioSegment.silent(duration=len(sound))
    
    for i in range(total_frames):
        start_time = i / fps
        end_time = (i + 1) / fps
        
        # Trouver la valeur de confiance pour cette frame
        confidence_index = min(int(i / total_frames * len(confidence_data)), len(confidence_data) - 1)
        volume = normalized_volumes[confidence_index]
        
        sound_frame = sound[start_time*1000:end_time*1000]
        sound_frame_with_volume = sound_frame.apply_gain(-volume)
        
        result_sound = result_sound.overlay(sound_frame_with_volume, gain_during_overlay=-volume)
    
    # Ajouter le son à la vidéo
    final_video = video.set_audio(result_sound)
    
    # Enregistrer la vidéo finale
    final_video.write_videofile("output.mp4")

# Exemple d'utilisation
video_path = "input_video.mp4"
audio_path = "background_music.mp3"
csv_path = "confidence_data.csv"

add_sound_to_video(video_path, audio_path, csv_path)

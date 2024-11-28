import numpy as np
import pyaudio
import socket

UDP_IP = '127.0.0.1'  # Même IP que l'envoyeur
UDP_PORT = 12345      # Même port que l'envoyeur

# Création du socket UDP
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"En attente de messages UDP sur {UDP_IP}:{UDP_PORT}")

while True:
    data, addr = sock.recvfrom(1024)  # Réception d'un message (taille max : 1024 octets)
    print(f"Message reçu : {data.decode()} depuis {addr}")


# Charge la variable x1
try:
    with open("x1_variable.pkl", "rb") as f:
        x1 = pickle.load(f)
    print(f"Variable x1 récupérée : {x1}")
except FileNotFoundError:
    print("Le fichier contenant la variable x1 n'a pas été trouvé.")

def generate_sine_wave(frequency, duration, volume=0.5):
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = np.sin(frequency * t * 2 * np.pi) * volume
    return wave.astype(np.float32)

def adjust_volume(wave, volume_factor):
    """
    Ajuste le volume de l'onde sinusoïdale.
    :param wave: Onde sinusoïdale générée
    :param volume_factor: Facteur d'ajustement du volume (entre 0 et 1)
    :return: Onde sinusoïdale avec le volume ajusté
    """
    max_amplitude = np.max(np.abs(wave))
    adjusted_max_amplitude = max_amplitude * volume_factor
    adjusted_wave = wave * (adjusted_max_amplitude / max_amplitude)
    return adjusted_wave.astype(np.float32)

def play_note(frequency, duration, adjusted_wave=None):
    p = pyaudio.PyAudio()
    
    if adjusted_wave is None:
        # Générer l'onde sinusoïdale avec le volume ajusté directement ici
        adjusted_wave = generate_sine_wave(frequency, duration, volume=0.8)
    
    stream = p.open(format=p.get_format_from_width(2),
                    channels=1,
                    rate=44100,
                    output=True,
                    frames_per_buffer=1024)
    
    # Convertir les valeurs flottantes en entiers 16 bits
    audio_data = (adjusted_wave * 32767).astype(np.int16)
    
    # Écrire les données audio dans le flux
    stream.write(audio_data.tobytes())
    
    stream.stop_stream()
    stream.close()
    p.terminate()

def main():
    total_duration = 10
    num_variations = 20
    
    frequencies = np.random.uniform(200, 800, num_variations)
    segment_duration = total_duration / num_variations
    
    for freq in frequencies:
        # Ajuster dynamiquement le volume pour chaque note
        volume_factor = np.random.uniform(0.5, 1.5)  # Volume aléatoire entre 50% et 150%
        
        # Générer l'onde sinusoïdale avec le volume ajusté directement ici
        adjusted_wave = generate_sine_wave(freq, segment_duration, volume=volume_factor)
        
        play_note(freq, segment_duration, adjusted_wave)
    
    print("Fin du morceau.")

if __name__ == "__main__":
    main()

import numpy as np
from midiutil import MIDIFile
import matplotlib.pyplot as plt

def generate_ou_notes(n_steps, theta, mu, sigma, start_value):
    dt = 1
    notes = [start_value]

    for _ in range(n_steps - 1):
        dW = np.random.normal(0, np.sqrt(dt))
        dX = theta * (notes[-1] - mu) * dt + sigma + dW
        notes.append(notes[-1] + dX)

    return np.array(notes)

def map_to_scale(notes, scale):
    return [min(scale, key=lambda x: abs(x - note)) for note in notes]

def save_to_midi(notes, file_name, duration=1):
    midi = MIDIFile(1)
    midi.addTempo(0, 0, 120)

    for i, note in enumerate(notes):
        midi.addNote(0, 0, int(note), i * duration, duration, 100)

    with open(file_name, 'wb') as f:
        midi.writeFile(f)
    print(f"Midi file saved to {file_name}")

n_steps = 50
theta = 0.5
mu = 60
sigma = 10
start_value = 60

ou_notes = generate_ou_notes(n_steps, theta, mu, sigma, start_value)

c_major_scale = [60, 62, 64, 65, 67,69, 71, 72]

mapped_notes = map_to_scale(ou_notes, c_major_scale)

save_to_midi(mapped_notes, 'music.mid')

plt.figure(figsize=(10, 5))
plt.plot(ou_notes, label="OU Process (Raw Notes)", alpha=0.7)
plt.scatter(range(len(mapped_notes)), mapped_notes, color='red', label="Mapped Notes (C Major Scale)")
plt.title("Ornstein-Uhlenbeck Process for Music Generation")
plt.xlabel("Time Step")
plt.ylabel("MIDI Note Number")
plt.legend()
plt.show()
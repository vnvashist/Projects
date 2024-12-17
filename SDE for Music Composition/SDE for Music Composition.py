import numpy as np
from midiutil import MIDIFile
import matplotlib.pyplot as plt


def generate_ou_notes(n_steps, theta, mu, sigma, start_value):
    dt = 1
    notes = [start_value]
    for _ in range(n_steps - 1):
        dW = np.random.normal(0, np.sqrt(dt))
        dX = -theta * (notes[-1] - mu) * dt + sigma * dW
        notes.append(notes[-1] + dX)
    return np.array(notes)


def map_to_scale(notes, scale):
    return [min(scale, key=lambda x: abs(x - note)) for note in notes]


def save_to_midi(melody, harmony, file_name, durations):
    midi = MIDIFile(2)  # Two tracks: melody and harmony
    midi.addTempo(0, 0, 180)
    midi.addTempo(1, 0, 180)

    # Cumulative time for melody and harmony
    time_melody = 0
    time_harmony = 0

    for m_note, h_note, duration in zip(melody, harmony, durations):
        # Melody note
        midi.addNote(0, 0, int(m_note), time_melody, duration, np.random.randint(60, 100))
        time_melody += duration  # Increment time for next melody note

        # Harmony note
        midi.addNote(1, 0, int(h_note), time_harmony, duration, np.random.randint(50, 90))
        time_harmony += duration  # Increment time for next harmony note

    with open(file_name, "wb") as f:
        midi.writeFile(f)
    print(f"MIDI file saved as {file_name}")


# Parameters
n_steps = 100
theta, mu, sigma = 0.5, 60, 20
start_value = 60

# Generate melody and harmony
melody_ou = generate_ou_notes(n_steps, theta, mu, sigma, start_value)
harmony_ou = generate_ou_notes(n_steps, theta, mu - 12, sigma, start_value - 12)

# Map to scales
c_major_scale = [60, 62, 64, 65, 67, 69, 71, 72]
a_minor_scale = [57, 59, 60, 61, 62, 64, 65, 67, 69]
c_lydian_scale = [60, 62, 64, 66, 67, 69, 71, 72]
melody = map_to_scale(melody_ou, c_major_scale)
harmony = map_to_scale(harmony_ou, c_lydian_scale)

# Add rhythm
durations = np.random.choice([0.5, 1], size=n_steps)

# Save to MIDI
save_to_midi(melody, harmony, "ou_music_c_lydian.mid", durations)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(melody_ou, label="Melody OU Process", alpha=0.7)
plt.plot(harmony_ou, label="Harmony OU Process", alpha=0.7)
plt.scatter(range(len(melody)), melody, color='red', label="Mapped Melody Notes")
plt.title("OU Process for Music Generation")
plt.xlabel("Time Step")
plt.ylabel("MIDI Note Number")
plt.legend()
plt.show()

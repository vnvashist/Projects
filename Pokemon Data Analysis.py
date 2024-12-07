import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def fetch_pokemon_data(limit=1025):
    """
    Fetch Pokémon data from the PokéAPI.
    Args:
        limit (int): Number of Pokemon to fetch (1025 for total # of current Pokemon).
    Returns:
        DataFrame: Pokemon data
    """
    base_url = "https://pokeapi.co/api/v2/pokemon"
    pokemon_data = []

    for i in range(1, limit + 1):
        response = requests.get(f"{base_url}/{i}")
        if response.status_code == 200:
            data = response.json()

            # Relevant fields
            pokemon = {
                "id": data["id"],
                "name": data["name"],
                "height": data["height"],
                "weight": data["weight"],
                "base_experience": data["base_experience"],
                "types": [t["type"]["name"] for t in data["types"]],
                "abilities": [a["ability"]["name"] for a in data["abilities"]],
                "stats": {s["stat"]["name"]: s["base_stat"] for s in data["stats"]},
            }
            pokemon_data.append(pokemon)
        else:
            print(f"Failed to fetch data for Pokémon ID: {i}, Response code: {response.status_code}")

    return pd.DataFrame(pokemon_data)

pokemon_df = fetch_pokemon_data()

# Normalize 'types' into separate columns
types_df = pokemon_df["types"].apply(pd.Series).rename(columns=lambda x: f"type_{x + 1}")
pokemon_df = pd.concat([pokemon_df, types_df], axis=1).drop(columns=["types"])

# Normalize 'abilities' into separate columns
abilities_df = pokemon_df["abilities"].apply(pd.Series).rename(columns=lambda x: f"ability_{x + 1}")
pokemon_df = pd.concat([pokemon_df, abilities_df], axis=1).drop(columns=["abilities"])

# Expand 'stats' into separate columns
stats_df = pokemon_df["stats"].apply(pd.Series)
pokemon_df = pd.concat([pokemon_df, stats_df], axis=1).drop(columns=["stats"])

pokemon_df.fillna({"type_2": "None", "ability_2": "None", "ability_3": "None"}, inplace=True)

# One-hot encode categorical columns
encoded_df = pd.get_dummies(pokemon_df, columns=["type_1", "type_2", "ability_1", "ability_2", "ability_3"])

from sklearn.preprocessing import MinMaxScaler

# Select numeric columns to normalize
numeric_cols = ["height", "weight", "base_experience", "hp", "attack", "defense",
                "special-attack", "special-defense", "speed"]

# Apply Min-Max Scaling
scaler = MinMaxScaler()
pokemon_df[numeric_cols] = scaler.fit_transform(pokemon_df[numeric_cols])

pokemon_df.to_csv('pokemon_data.csv', index=False)

pokemon_data = pokemon_df.copy()

type_columns = [col for col in pokemon_data.columns if "type_1" in col]

if "type_1" in pokemon_data.columns:
    # Set up color mapping for primary types
    unique_types = pokemon_data["type_1"].unique()
    color_map = {ptype: plt.cm.tab10(i / len(unique_types)) for i, ptype in enumerate(unique_types)}

    # Create the scatter plot
    plt.figure(figsize=(14, 10))
    for p_type in unique_types:
        type_data = pokemon_data[pokemon_data["type_1"] == p_type]
        # Scatter plot for the current type
        plt.scatter(
            type_data["attack"],
            type_data["speed"],
            color=color_map[p_type],
            label=p_type,
            alpha=0.7
        )
        # Fit and plot regression line
        coefficients = np.polyfit(type_data["attack"], type_data["speed"], deg=1)
        poly_eq = np.poly1d(coefficients)
        x_vals = np.linspace(type_data["attack"].min(), type_data["attack"].max(), 100)
        plt.plot(
            x_vals,
            poly_eq(x_vals),
            color=color_map[p_type],
            linestyle="--",
            linewidth=2,
            label=f"{p_type} fit"
        )

    # Add titles and labels
    plt.title("Speed vs. Attack by Pokémon Primary Type with Best Fit Lines")
    plt.xlabel("Attack (Normalized)")
    plt.ylabel("Speed (Normalized)")
    plt.legend(title="Primary Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
else:
    print("The column 'type_1' does not exist in the dataset. Please check the data.")



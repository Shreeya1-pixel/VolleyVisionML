import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_volleyball_dataset(num_players=100, num_matches_per_player=20):
    """
    Generate a comprehensive volleyball dataset with realistic player statistics
    """
    np.random.seed(42)
    random.seed(42)
    
    # Player names and positions
    first_names = ['Emma', 'Sophia', 'Olivia', 'Ava', 'Isabella', 'Mia', 'Charlotte', 'Amelia', 
                   'Harper', 'Evelyn', 'Abigail', 'Emily', 'Elizabeth', 'Sofia', 'Madison', 'Avery',
                   'Ella', 'Scarlett', 'Grace', 'Chloe', 'Victoria', 'Riley', 'Aria', 'Lily',
                   'Aubrey', 'Zoey', 'Penelope', 'Layla', 'Nora', 'Lily', 'Camila', 'Adalyn',
                   'Alex', 'Jordan', 'Taylor', 'Casey', 'Morgan', 'Riley', 'Avery', 'Quinn',
                   'Parker', 'Dakota', 'Reese', 'Blake', 'Hayden', 'Kendall', 'Skylar', 'Rowan']
    
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis',
                  'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson',
                  'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin', 'Lee', 'Perez', 'Thompson',
                  'White', 'Harris', 'Sanchez', 'Clark', 'Ramirez', 'Lewis', 'Robinson', 'Walker',
                  'Young', 'Allen', 'King', 'Wright', 'Scott', 'Torres', 'Nguyen', 'Hill', 'Flores',
                  'Green', 'Adams', 'Nelson', 'Baker', 'Hall', 'Rivera', 'Campbell', 'Mitchell']
    
    positions = ['Setter', 'Outside Hitter', 'Middle Blocker', 'Opposite Hitter', 'Libero', 'Defensive Specialist']
    
    # Generate player data
    players_data = []
    
    for player_id in range(num_players):
        # Player characteristics
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
        position = random.choice(positions)
        age = random.randint(16, 25)
        height = random.uniform(160, 190)  # cm
        experience_years = random.randint(1, 8)
        
        # Base skill levels based on position and experience
        base_spike_accuracy = 0.65 + (experience_years * 0.02) + random.uniform(-0.1, 0.1)
        base_block_efficiency = 0.55 + (experience_years * 0.015) + random.uniform(-0.08, 0.08)
        base_serve_accuracy = 0.70 + (experience_years * 0.025) + random.uniform(-0.12, 0.12)
        base_dig_accuracy = 0.75 + (experience_years * 0.02) + random.uniform(-0.1, 0.1)
        
        # Position-specific adjustments
        if position == 'Setter':
            base_spike_accuracy *= 0.8
            base_serve_accuracy *= 1.1
        elif position == 'Outside Hitter':
            base_spike_accuracy *= 1.2
            base_block_efficiency *= 0.9
        elif position == 'Middle Blocker':
            base_spike_accuracy *= 1.1
            base_block_efficiency *= 1.3
        elif position == 'Libero':
            base_spike_accuracy *= 0.6
            base_dig_accuracy *= 1.3
            base_serve_accuracy *= 0.8
        
        # Generate match data for each player
        for match_num in range(num_matches_per_player):
            # Match date (last 2 years)
            match_date = datetime.now() - timedelta(days=random.randint(1, 730))
            
            # Performance with realistic variance
            spike_attempts = random.randint(15, 35)
            spike_success = int(spike_attempts * (base_spike_accuracy + random.uniform(-0.15, 0.15)))
            spike_accuracy = max(0, min(100, (spike_success / spike_attempts) * 100))
            
            block_attempts = random.randint(8, 20)
            block_success = int(block_attempts * (base_block_efficiency + random.uniform(-0.2, 0.2)))
            blocks = max(0, block_success)
            
            serve_attempts = random.randint(15, 25)
            serve_success = int(serve_attempts * (base_serve_accuracy + random.uniform(-0.2, 0.2)))
            aces = int(serve_success * random.uniform(0.1, 0.3))
            serves = f"{aces}/{serve_attempts}"
            
            dig_attempts = random.randint(20, 40)
            dig_success = int(dig_attempts * (base_dig_accuracy + random.uniform(-0.15, 0.15)))
            digs = max(0, dig_success)
            
            # Errors (inversely related to experience and skill)
            errors = random.randint(2, 8) - int(experience_years * 0.5)
            errors = max(0, errors)
            
            # Reaction time (improves with experience)
            base_reaction = 350 - (experience_years * 10) + random.uniform(-50, 50)
            reaction_time = max(200, min(500, base_reaction))
            
            # Match outcome factors
            match_duration = random.randint(60, 120)  # minutes
            team_score = random.randint(15, 25)
            opponent_score = random.randint(15, 25)
            won_match = team_score > opponent_score
            
            # Calculate performance score
            performance_score = (
                spike_accuracy * 0.3 +
                (blocks / block_attempts * 100) * 0.2 +
                (aces / serve_attempts * 100) * 0.2 +
                (digs / dig_attempts * 100) * 0.2 +
                (won_match * 10) +
                (max(0, 10 - errors) * 2)
            )
            
            # Mood/energy level (affects performance)
            energy_level = random.uniform(0.6, 1.0)
            mood_score = random.uniform(0.5, 1.0)
            
            # Apply energy/mood effects
            spike_accuracy *= energy_level
            blocks *= mood_score
            digs *= energy_level
            
            players_data.append({
                'player_id': player_id,
                'name': name,
                'position': position,
                'age': age,
                'height': height,
                'experience_years': experience_years,
                'match_date': match_date,
                'match_num': match_num + 1,
                'spike_attempts': spike_attempts,
                'spike_success': spike_success,
                'spike_accuracy': round(spike_accuracy, 2),
                'block_attempts': block_attempts,
                'blocks': blocks,
                'serve_attempts': serve_attempts,
                'aces': aces,
                'serves': serves,
                'dig_attempts': dig_attempts,
                'digs': digs,
                'errors': errors,
                'reaction_time': round(reaction_time, 1),
                'match_duration': match_duration,
                'team_score': team_score,
                'opponent_score': opponent_score,
                'won_match': won_match,
                'performance_score': round(performance_score, 2),
                'energy_level': round(energy_level, 2),
                'mood_score': round(mood_score, 2)
            })
    
    df = pd.DataFrame(players_data)
    return df

def save_dataset(df, filename="data/volleyball_dataset.csv"):
    """Save the generated dataset to CSV"""
    df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}")
    print(f"Shape: {df.shape}")
    print(f"Players: {df['player_id'].nunique()}")
    print(f"Matches: {len(df)}")
    return filename

if __name__ == "__main__":
    # Generate and save the dataset
    df = generate_volleyball_dataset(num_players=150, num_matches_per_player=25)
    save_dataset(df) 
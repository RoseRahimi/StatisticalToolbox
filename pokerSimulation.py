#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 14:47:00 2025

@author: fatimarahimi
"""
import random
from collections import Counter

# Define ranks and suits
ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
suits = ['hearts', 'diamonds', 'clubs', 'spades']

# Define the deck of 52 cards
deck = [rank + ' of ' + suit for rank in ranks for suit in suits]

# Function to deal a hand
def deal_hand():
    deck_copy = deck.copy()
    random.shuffle(deck_copy)
    
    # Deal two hole cards to the player
    hole_cards = deck_copy[:2]
    
    # Deal five community cards (flop, turn, river)
    community_cards = deck_copy[2:7]
    
    return hole_cards, community_cards

# Helper function to extract ranks from cards
def extract_ranks(cards):
    return [card.split(' ')[0] for card in cards]

# Function to evaluate the hand
def evaluate_hand(hole_cards, community_cards):
    all_cards = hole_cards + community_cards
    ranks = extract_ranks(all_cards)
    rank_count = Counter(ranks)
    
    if 2 in rank_count.values():
        return 'Pair'
    elif 3 in rank_count.values():
        return 'Three of a Kind'
    elif 4 in rank_count.values():
        return 'Four of a Kind'
    else:
        return 'High Card'

# Monte Carlo simulation for poker hands
def poker_sim(num_simulations):
    hand_frequencies = Counter()

    for _ in range(num_simulations):
        hole_cards, community_cards = deal_hand()
        hand_result = evaluate_hand(hole_cards, community_cards)
        hand_frequencies[hand_result] += 1

    return hand_frequencies

# Run the simulation for 10000 games
num_simulations = 10000
simulation_results = poker_sim(num_simulations)

# Print the results
for hand, frequency in simulation_results.items():
    print(f'{hand}: {frequency} times ({(frequency / num_simulations) * 100:.2f}%)')

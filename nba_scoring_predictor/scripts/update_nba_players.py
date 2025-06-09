#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to update NBA players data
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.nba_player_fetcher import NBAPlayerFetcher
from utils.logger import main_logger as logger

def main():
    """Main update function."""
    print("🏀 NBA Player Data Updater")
    print("=" * 40)
    
    fetcher = NBAPlayerFetcher()
    
    # Check existing data
    existing_data = fetcher.load_players_data()
    if existing_data:
        last_updated = existing_data['metadata']['last_updated']
        total_players = existing_data['metadata']['total_players']
        print(f"📊 Current data: {total_players} players (last updated: {last_updated})")
    else:
        print("📊 No existing data found")
    
    # Ask user if they want to update
    response = input("\n🔄 Update NBA players data? (y/n): ").lower().strip()
    
    if response == 'y':
        try:
            print("\n🚀 Fetching NBA players data...")
            data = fetcher.fetch_all_active_players()
            
            print(f"\n✅ Successfully updated! Fetched {data['metadata']['total_players']} players")
            print("\n📍 Position breakdown:")
            for position, players in data['players_by_position'].items():
                if players:  # Only show positions with players
                    print(f"   {position}: {len(players)} players")
            
            # Show some example players by position
            print("\n👥 Sample players by position:")
            for position in ['PG', 'SG', 'SF', 'PF', 'C']:
                position_players = data['players_by_position'][position]
                if position_players:
                    sample_names = [p['name'] for p in position_players[:3]]
                    print(f"   {position}: {', '.join(sample_names)}...")
            
        except Exception as e:
            print(f"\n❌ Error updating data: {e}")
            sys.exit(1)
    else:
        print("\n⏭️  Update cancelled")

if __name__ == "__main__":
    main()

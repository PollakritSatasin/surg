#!/usr/bin/env python3
"""
Example of using the segmentation model in a reinforcement learning environment.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from surg.models import SegmentationModel
from surg.rl_integration import (
    SegmentationObservationWrapper,
    SegmentationFeatureExtractor,
    create_rl_env_with_segmentation
)
from surg.config import load_config


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='RL integration example')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--env', type=str, default='CartPole-v1', help='Gym environment name')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to run')
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        print("Config file not found. Using defaults.")
        from surg.config.config import get_default_config
        config = get_default_config()
    
    # Initialize model
    print("Loading segmentation model...")
    device = config['inference']['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = SegmentationModel(
        num_classes=config['model']['num_classes'],
        pretrained=False,
        backbone=config['model']['backbone']
    )
    
    # Load trained weights
    model.load(args.model, device=device)
    print(f"Model loaded from: {args.model}")
    
    # Create RL environment with segmentation
    print(f"Creating environment: {args.env}")
    
    try:
        env = create_rl_env_with_segmentation(
            base_env_name=args.env,
            segmentation_model=model,
            device=device,
            confidence_threshold=config['inference']['confidence_threshold']
        )
        
        # Initialize feature extractor
        feature_extractor = SegmentationFeatureExtractor(feature_type='combined')
        
        # Run episodes
        print(f"\nRunning {args.episodes} episodes...")
        for episode in range(args.episodes):
            obs = env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            print(f"\nEpisode {episode + 1}:")
            
            while not done:
                # Extract segmentation features
                if 'boxes' in obs:
                    features = feature_extractor.extract(obs)
                    print(f"  Step {step}: Detected {obs['num_objects']} objects")
                
                # Take random action (replace with your RL policy)
                action = env.action_space.sample()
                
                # Step environment
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                step += 1
            
            print(f"  Episode reward: {episode_reward}")
        
        env.close()
        print("\nDone!")
        
    except Exception as e:
        print(f"Error running RL example: {e}")
        print("Note: This example requires an environment with image observations.")
        print("Consider using vision-based environments like Atari or custom environments.")


if __name__ == '__main__':
    main()

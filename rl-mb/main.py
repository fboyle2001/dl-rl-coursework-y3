import argparse
import agents
import torch
from typing import Optional

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="BipedalWalker-v3")
    parser.add_argument("--cuda", action="store_true", default=False)
    parser.add_argument("--agent")
    parser.add_argument("--video-freq", type=int, default=None)
    args = parser.parse_args()

    assert args.agent and args.agent in agents.AGENT_MAP.keys(), f"{args.agent} is an invalid agent type"

    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")
    agent = agents.AGENT_MAP[args.agent](args.env, device, args.video_freq)

    print(f"Agent Name: {agent._agent_name}")
    print(f"Env: {args.env}")
    print(f"Cuda Enabled: {args.cuda} (device: {device})")
    print(f"Video (every N episodes): {args.video_freq}")
    print(f"Initialised ID: {agent._id}")
    print(f"Directory: {agent._home_directory}")
    print("Starting...")

    agent.run()

if __name__ == "__main__":
    main()
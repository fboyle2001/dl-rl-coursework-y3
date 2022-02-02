import argparse
import agents
import torch
import os
import svg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="BipedalWalker-v3")
    parser.add_argument("--cuda", action="store_true", default=False)
    parser.add_argument("--agent")
    parser.add_argument("--video-freq", type=int, default=None)
    parser.add_argument("--virtual-display", action="store_true", default=False)
    args = parser.parse_args()

    agents.AGENT_MAP["TrueSVG"] = svg.SACSVGAgent

    assert args.agent and args.agent in agents.AGENT_MAP.keys(), f"{args.agent} is an invalid agent type"

    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")
    agent = agents.AGENT_MAP[args.agent](args.env, device, args.video_freq)

    print(f"Agent Name: {agent._agent_name}")
    print(f"Env: {args.env}")
    print(f"Virtual Display: {args.virutal_display}")
    print(f"Cuda Enabled: {args.cuda} (device: {device})")
    print(f"Video (every N episodes): {args.video_freq}")
    print(f"Initialised ID: {agent._id}")
    print(f"Directory: {os.path.abspath(agent._home_directory)}")

    if args.virtual_display and args.video_freq is not None and args.video_freq > 0:
        from pyvirtualdisplay import Display
        display = Display(visible=False,size=(600,600))
        display.start()

        print("Virtual Display Enabled")

    print("Starting...")
    agent.run()

if __name__ == "__main__":
    main()
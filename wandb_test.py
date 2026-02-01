import os
import time
import wandb
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_project", type=str, default="wandb-test")
    parser.add_argument("--run_name", type=str, default="wandb_connection_test")
    return parser.parse_args()


def main():
    args = parse_args()

    # 반드시 online
    os.environ["WANDB_MODE"] = "online"

    # ⚠️ login() 호출하지 않음 (v1 key는 env로만)
    wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config={
            "purpose": "wandb connectivity test",
            "env": "no_gpu_no_ddp",
        },
    )

    print("✅ wandb initialized")

    for step in range(5):
        wandb.log({
            "test/loss": 1.0 / (step + 1),
            "test/acc": step * 0.2,
        })
        print(f"logged step {step}")
        time.sleep(0.5)

    wandb.finish()
    print("✅ wandb finished successfully")


if __name__ == "__main__":
    main()

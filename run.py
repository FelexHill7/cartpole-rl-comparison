import argparse
from trainer import Trainer

ENV_NAME = 'CartPole-v1'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cmd",
        type=str,
        choices=["train", "eval", "compare"],
        help="command to execute: train / eval / compare",
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=["pg", "dqn", "ac"],
        default="pg",
        help="algorithm to use: pg / dqn / ac",
    )
    parser.add_argument(
        "--episodes", type=int, default=10, help="number of training episodes"
    )
    parser.add_argument(
        '--eval_episodes', type=int, default=5, help="number of evaluating episodes"
    )
    parser.add_argument(
        "--out_dir", type=str, default='./output', help="path to saved model for evaluation"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.98, help="discount factor"
    )
    parser.add_argument(
        "--lr", type=float, default=5e-4, help="learning rate"
    )
    parser.add_argument(
        "--critic_lr", type=float, default=5e-4, help="critic learning rate for AC"
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.1, help="epsilon for DQN"
    )
    parser.add_argument(
        "--buffer_size", type=int, default=10000, help="replay buffer size for DQN"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size for DQN"
    )
    parser.add_argument(
        '--compare_algos', choices=["pg", "dqn", "ac"], nargs='*', default=['pg', 'dqn', 'ac'], help="algorithms to compare"
    )

    args = parser.parse_args()
    defaults = {
        "pg": {"lr": 5e-4},
        "dqn": {"lr": 5e-4, "epsilon": 0.1, "buffer_size": 10000, "batch_size": 32},
        "ac": {"lr": 5e-4, "critic_lr": 5e-4},
    }

    trainer = Trainer(hyperparams_map=defaults)

    if args.cmd == "train":
        # simple default hyperparams
        config = {}
        if args.gamma:
            config["gamma"] = args.gamma
        if args.lr:
            config["lr"] = args.lr
        if args.algo == 'ac':
            if args.critic_lr:
                config["critic_lr"] = args.critic_lr
        if args.algo == "dqn":
            if args.epsilon:
                config["epsilon"] = args.epsilon
            if args.buffer_size:
                config["buffer_size"] = args.buffer_size
            if args.batch_size:
                config["batch_size"] = args.batch_size
        trainer.train(agent_key=args.algo, env_name=ENV_NAME, out_dir=args.out_dir, config=config, num_episodes=args.episodes)
    elif args.cmd == "eval":
        res = trainer.evaluate(agent_key=args.algo, env_name=ENV_NAME, model_path=args.out_dir, num_episodes=args.eval_episodes)
        print("Eval rewards:", res)
    elif args.cmd == "compare":
        trainer.compare(args.compare_algos, env_name=ENV_NAME, out_dir=args.out_dir, num_train_episodes=args.episodes, num_eval_episodes=args.eval_episodes)


if __name__ == "__main__":
    main()

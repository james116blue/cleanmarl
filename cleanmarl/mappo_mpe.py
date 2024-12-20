# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter



@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    use_ReLU: bool = True
    """use Relu deafault or Tanh"""
    use_huber_loss: bool = True
    """use huber loss or MSE"""
    huber_delta: float = 10.
    """coefficience of huber loss."""
    use_value_normalization: bool = True
    """should be returns to go normalized to decrease variation"""
    ewa_weigth: float = 0.999999
    """weight used for exponential moving average"""
    env_id: str = "simple_spread_v3"
    """the id of the environment"""
    total_timesteps: int = 100000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 128
    """the number of parallel game environments"""
    num_steps: int = 25
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 1
    """the number of mini-batches"""
    update_epochs: int = 15
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 2.0
    """coefficient of the value function"""
    max_grad_norm: float = 10.
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def layer_init(layer, bias_const=0.0):
    assert args.use_ReLU is not None
    gain = nn.init.calculate_gain(['tanh', 'relu'][args.use_ReLU])
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        active_func = [nn.Tanh(), nn.ReLU()][args.use_ReLU]
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.state_space.shape).prod(), 64)),
            active_func,
            layer_init(nn.Linear(64, 64)),
            active_func,
            layer_init(nn.Linear(64, 1)),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            active_func,
            layer_init(nn.Linear(64, 64)),
            active_func,
            layer_init(nn.Linear(64, envs.single_action_space.n)),
        )

    def get_value(self, s):
        return self.critic(s)

    def get_action_and_value(self, x, s, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(s)


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )


    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    import importlib
    import supersuit as ss
    env = importlib.import_module(f"pettingzoo.mpe.{args.env_id}").parallel_env()
    # env = ss.agent_indicator_v0(env, type_only=False)
    envs = ss.pettingzoo_env_to_vec_env_v1(env)
    envs = ss.concat_vec_envs_v1(envs, args.num_envs , num_cpus=0, base_class="gymnasium")
    envs.is_vector_env = True
    envs.single_observation_space = envs.observation_space
    envs.single_action_space = envs.action_space
    envs.state_space = env.state_space
    envs.num_agents = len(env.possible_agents)
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    #take into account that each env consists of several agents
    args.num_envs *= envs.num_agents
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    #debug
    print(f"env obs space: {envs.observation_space}")
    print(f"env state space: {envs.state_space}")
    print(f"env action space: {envs.action_space}")
    print(f"num agent per env: {envs.num_agents}")
    print(f"total num parallel agents {args.num_envs}")

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    def huber_loss(e, d):
        a = (abs(e) <= d).float()
        b = (abs(e) > d).float()
        return a * e ** 2 / 2 + b * d * (abs(e) - d / 2)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    states = torch.zeros((args.num_steps, args.num_envs) + envs.state_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_state = next_obs.reshape(args.num_envs // envs.num_agents, -1).repeat(envs.num_agents, axis=0)
    next_obs = torch.Tensor(next_obs).to(device)
    next_state = torch.Tensor(next_state).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    #reward statistics
    episode_rewards = np.zeros(args.num_envs  )
    episode_lengths = np.zeros(args.num_envs )

    if args.use_value_normalization:
        running_mean = torch.zeros(size=(1,)).to(device)
        running_mean_sq = torch.zeros(size=(1,)).to(device)
        debiasing_term = torch.zeros(size=(1,)).to(device)
        mean = running_mean / debiasing_term.clamp(min=1e-5)
        mean_sq = running_mean_sq / debiasing_term.clamp(min=1e-5)
        var = (mean_sq - mean ** 2).clamp(min=1e-2)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            episode_lengths += 1
            obs[step] = next_obs
            states[step] = next_state
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs, next_state)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_state = next_obs.reshape(args.num_envs // envs.num_agents, -1).repeat(envs.num_agents, axis=0)
            next_done = np.logical_or(terminations, truncations)
            #compute reward stat
            episode_rewards += reward # * envs.num_agents  # LAME stats   , multipply by agents number for comaprision with HARL paper
            if np.any(next_done):
                writer.add_scalar(f"charts/average_per_player_episodic_return", np.mean(episode_rewards * next_done), global_step)
                writer.add_scalar(f"charts/share_episodic_return", np.mean(envs.num_envs * episode_rewards * next_done), global_step)
                writer.add_scalar(f"charts/episodic_length", np.mean(episode_lengths * next_done), global_step)
                episode_rewards[next_done] = 0
                episode_lengths[next_done] = 0

            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_state, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_state).to(device), torch.Tensor(next_done).to(device)

        with torch.no_grad():
            next_value = agent.get_value(next_state).reshape(1, -1)
            if args.use_value_normalization:
                values = values * torch.sqrt(var) + mean
                next_value = next_value * torch.sqrt(var) + mean
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_states = states.reshape((-1,) + envs.state_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        valueclipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_states[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.use_value_normalization:
                    with torch.no_grad():
                        #update
                        running_mean = args.ewa_weigth * running_mean + (1 - args.ewa_weigth) * b_returns[mb_inds].mean()
                        running_mean_sq = args.ewa_weigth * running_mean_sq + (1 - args.ewa_weigth) * (b_returns[mb_inds] ** 2).mean()
                        debiasing_term = args.ewa_weigth * debiasing_term + (1 - args.ewa_weigth) * 1.0
                        mean = running_mean / debiasing_term.clamp(min=1e-5)
                        mean_sq = running_mean_sq / debiasing_term.clamp(min=1e-5)
                        var = (mean_sq - mean ** 2).clamp(min=1e-2)
                    #normalize
                    target_values = (b_returns[mb_inds] - mean) / torch.sqrt(var)
                else:
                    target_values = b_returns[mb_inds]
                assert target_values.requires_grad==False
                v_loss_unclipped_error = newvalue - target_values
                v_loss_unclipped =  huber_loss(v_loss_unclipped_error, args.huber_delta) if args.use_huber_loss else 0.5 * (v_loss_unclipped_error ** 2)
                if args.clip_vloss:
                    v_clipped = target_values + torch.clamp(
                        newvalue - target_values,
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped_error = v_clipped - target_values
                    v_loss_clipped = huber_loss(v_loss_clipped_error, args.huber_delta) if args.use_huber_loss else 0.5 * (v_loss_clipped_error ** 2)
                    with torch.no_grad():
                        valueclipfracs += [(v_loss_clipped > v_loss_unclipped).float().mean().item()]
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = v_loss_max.mean()
                else:
                    v_loss =  v_loss_unclipped.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/gradnorm", grad_norm.mean().item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/valueclipfrac", np.mean(valueclipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar("stats/value_normalizer_mean", mean.item(), global_step)
        writer.add_scalar("stats/value_normalizer_var", var.item(), global_step)

    envs.close()
    writer.close()
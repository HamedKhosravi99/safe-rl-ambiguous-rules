"""A9: vector-constrained forks of OSRL's BCQL and CPQ.

One cost critic PER retained reading and independent multipliers updated
on per-channel expected-cost violations — the faithful implementation of
the per-reading program max_pi J_r s.t. max_k J_{c_k} <= d, replacing the
pointwise union-cost surrogate (which optimizes E[sum_t max_k c_k] >=
max_k E[sum_t c_k]).

Batches carry costs of shape [B, K] (TransitionDataset passes an (N, K)
costs array through untouched). Parent trainers (BCQLTrainer, CPQTrainer)
work unchanged: train_one_step forwards costs to cost_critic_loss.
"""
from __future__ import annotations

import itertools
from copy import deepcopy

import torch
import torch.nn as nn

from osrl.algorithms.bcql import BCQL
from osrl.algorithms.cpq import CPQ
from osrl.common.net import (EnsembleDoubleQCritic, EnsembleQCritic,
                             LagrangianPIDController)


class VectorBCQL(BCQL):
    def __init__(self, *args, n_costs: int = 2, **kw):
        super().__init__(*args, **kw)
        self.n_costs = n_costs
        self.cost_critic = nn.ModuleList([
            EnsembleDoubleQCritic(self.state_dim, self.action_dim,
                                  self.c_hidden_sizes, nn.ReLU,
                                  num_q=self.num_qc).to(self.device)
            for _ in range(n_costs)])
        self.cost_critic_old = deepcopy(self.cost_critic)
        self.cost_critic_old.eval()
        self.controllers = [
            LagrangianPIDController(self.KP, self.KI, self.KD,
                                    self.qc_thres)
            for _ in range(n_costs)]

    def cost_critic_loss(self, observations, next_observations, actions,
                         costs, done):
        assert costs.dim() == 2 and costs.shape[1] == self.n_costs, \
            f"vector costs expected [B,{self.n_costs}], got {costs.shape}"
        total = 0.0
        stats = {}
        with torch.no_grad():
            batch_size = next_observations.shape[0]
            obs_next = torch.repeat_interleave(
                next_observations, self.sample_action_num, 0
            ).to(self.device)
            act_targ_next = self.actor_old(obs_next,
                                           self.vae.decode(obs_next))
        for k in range(self.n_costs):
            _, _, q1_list, q2_list = self.cost_critic[k].predict(
                observations, actions)
            with torch.no_grad():
                q1_t, q2_t, _, _ = self.cost_critic_old[k].predict(
                    obs_next, act_targ_next)
                q_targ = self.lmbda * torch.min(q1_t, q2_t) + \
                    (1. - self.lmbda) * torch.max(q1_t, q2_t)
                q_targ = q_targ.reshape(batch_size, -1).max(1)[0]
                backup = costs[:, k] + self.gamma * q_targ
            total = total + self.cost_critic[k].loss(backup, q1_list) + \
                self.cost_critic[k].loss(backup, q2_list)
        self.cost_critic_optim.zero_grad()
        total.backward()
        self.cost_critic_optim.step()
        stats["loss/cost_critic_loss"] = total.item()
        return total, stats

    def actor_loss(self, observations):
        for p in itertools.chain(self.critic.parameters(),
                                 self.cost_critic.parameters(),
                                 self.vae.parameters()):
            p.requires_grad = False
        actions = self.actor(observations, self.vae.decode(observations))
        q1_pi, q2_pi, _, _ = self.critic.predict(observations, actions)
        q_pi = torch.min(q1_pi, q2_pi)
        penalty = 0.0
        stats = {}
        for k in range(self.n_costs):
            qc1, qc2, _, _ = self.cost_critic[k].predict(observations,
                                                         actions)
            qc_pi = torch.min(qc1, qc2)
            with torch.no_grad():
                mult = self.controllers[k].control(qc_pi).detach()
            penalty = penalty + ((qc_pi - self.qc_thres) * mult).mean()
            stats[f"loss/lagrangian_{k}"] = float(mult.item())
        loss_actor = -q_pi.mean() + penalty
        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()
        stats["loss/actor_loss"] = loss_actor.item()
        for p in itertools.chain(self.critic.parameters(),
                                 self.cost_critic.parameters(),
                                 self.vae.parameters()):
            p.requires_grad = True
        return loss_actor, stats

    def setup_optimizers(self, actor_lr, critic_lr, vae_lr):
        self.actor_optim = torch.optim.Adam(self.actor.parameters(),
                                            lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(),
                                             lr=critic_lr)
        self.cost_critic_optim = torch.optim.Adam(
            self.cost_critic.parameters(), lr=critic_lr)
        self.vae_optim = torch.optim.Adam(self.vae.parameters(), lr=vae_lr)

    def sync_weight(self):
        self._soft_update(self.critic_old, self.critic, self.tau)
        self._soft_update(self.cost_critic_old, self.cost_critic, self.tau)
        self._soft_update(self.actor_old, self.actor, self.tau)


class VectorCPQ(CPQ):
    def __init__(self, *args, n_costs: int = 2, **kw):
        super().__init__(*args, **kw)
        self.n_costs = n_costs
        self.cost_critic = nn.ModuleList([
            EnsembleQCritic(self.state_dim, self.action_dim,
                            self.c_hidden_sizes, nn.ReLU,
                            num_q=self.num_qc).to(self.device)
            for _ in range(n_costs)])
        self.cost_critic_old = deepcopy(self.cost_critic)
        self.cost_critic_old.eval()
        self.log_alphas = torch.zeros(n_costs, device=self.device)

    def _gate(self, obs, act):
        """Product over channels of 1{qc_k <= q_thres} (no grad)."""
        g = None
        for k in range(self.n_costs):
            qc, _ = self.cost_critic_old[k].predict(obs, act)
            gk = (qc <= self.q_thres)
            g = gk if g is None else (g & gk)
        return g

    def critic_loss(self, observations, next_observations, actions,
                    rewards, done):
        _, q_list = self.critic.predict(observations, actions)
        with torch.no_grad():
            next_actions, _ = self._actor_forward(next_observations, False,
                                                  True)
            q_targ, _ = self.critic_old.predict(next_observations,
                                                next_actions)
            gate = self._gate(next_observations, next_actions)
            backup = rewards + self.gamma * (1 - done) * gate * q_targ
        loss_critic = self.critic.loss(backup, q_list)
        self.critic_optim.zero_grad()
        loss_critic.backward()
        self.critic_optim.step()
        return loss_critic, {"loss/critic_loss": loss_critic.item()}

    def cost_critic_loss(self, observations, next_observations, actions,
                         costs, done):
        assert costs.dim() == 2 and costs.shape[1] == self.n_costs, \
            f"vector costs expected [B,{self.n_costs}], got {costs.shape}"
        with torch.no_grad():
            next_actions, _ = self._actor_forward(next_observations, False,
                                                  True)
            batch_size = observations.shape[0]
            _, _, pi_dist = self.actor(observations, False, True, True)
            sampled = pi_dist.sample([self.sample_action_num]).reshape(
                self.sample_action_num * batch_size, self.action_dim)
            stacked = torch.tile(observations[None, :, :],
                                 (self.sample_action_num, 1, 1)).reshape(
                self.sample_action_num * batch_size, self.state_dim)
            _, mean, std = self.vae(stacked, sampled)
            mean = mean.reshape(self.sample_action_num, batch_size,
                                self.latent_dim)
            std = std.reshape(self.sample_action_num, batch_size,
                              self.latent_dim)
            KL = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2)
                         - std.pow(2)).mean(2)
            quant = torch.quantile(KL, 0.75)
            ood_mask = (KL >= quant)
        total = 0.0
        for k in range(self.n_costs):
            _, qc_list = self.cost_critic[k].predict(observations, actions)
            with torch.no_grad():
                qc_targ, _ = self.cost_critic_old[k].predict(
                    next_observations, next_actions)
                backup = costs[:, k] + self.gamma * qc_targ
                qc_sampled, _ = self.cost_critic_old[k].predict(stacked,
                                                                sampled)
                qc_sampled = qc_sampled.reshape(self.sample_action_num,
                                                batch_size)
                qc_ood = (ood_mask * qc_sampled).mean(0)
            alpha_k = self.log_alphas[k].exp()
            total = total + self.cost_critic[k].loss(backup, qc_list) \
                - alpha_k * (qc_ood.mean() - self.qc_thres)
            with torch.no_grad():
                self.log_alphas[k] += self.alpha_lr * alpha_k * (
                    self.qc_thres - qc_ood.mean())
        self.log_alphas.data.clamp_(min=-5.0, max=5.0)
        self.cost_critic_optim.zero_grad()
        total.backward()
        self.cost_critic_optim.step()
        return total, {"loss/cost_critic_loss": float(total.item()),
                       "loss/alpha_0":
                       float(self.log_alphas[0].exp().item())}

    def actor_loss(self, observations):
        for p in itertools.chain(self.critic.parameters(),
                                 self.cost_critic.parameters()):
            p.requires_grad = False
        actions, _ = self._actor_forward(observations, False, True)
        q_pi, _ = self.critic.predict(observations, actions)
        gate = None
        for k in range(self.n_costs):
            qc_pi, _ = self.cost_critic[k].predict(observations, actions)
            gk = (qc_pi <= self.q_thres)
            gate = gk if gate is None else (gate & gk)
        loss_actor = -(gate * q_pi).mean()
        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()
        for p in itertools.chain(self.critic.parameters(),
                                 self.cost_critic.parameters()):
            p.requires_grad = True
        return loss_actor, {"loss/actor_loss": loss_actor.item()}

    def setup_optimizers(self, actor_lr, critic_lr, alpha_lr, vae_lr):
        self.alpha_lr = alpha_lr
        self.actor_optim = torch.optim.Adam(self.actor.parameters(),
                                            lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(),
                                             lr=critic_lr)
        self.cost_critic_optim = torch.optim.Adam(
            self.cost_critic.parameters(), lr=critic_lr)
        self.vae_optim = torch.optim.Adam(self.vae.parameters(), lr=vae_lr)

    def sync_weight(self):
        self._soft_update(self.critic_old, self.critic, self.tau)
        self._soft_update(self.cost_critic_old, self.cost_critic, self.tau)
        self._soft_update(self.actor_old, self.actor, self.tau)

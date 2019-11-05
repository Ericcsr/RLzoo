"""
Proximal Policy Optimization (PPO)
----------------------------
A simple version of Proximal Policy Optimization (PPO) using single thread.
PPO is a family of first-order methods that use a few other tricks to keep new policies close to old.
PPO methods are significantly simpler to implement, and empirically seem to perform at least as well as TRPO.

Reference
---------
Proximal Policy Optimization Algorithms, Schulman et al. 2017
High Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman et al. 2016
Emergence of Locomotion Behaviours in Rich Environments, Heess et al. 2017
MorvanZhou's tutorial page: https://morvanzhou.github.io/tutorials

Prerequisites
--------------
tensorflow >=2.0.0a0
tensorflow-probability 0.6.0
tensorlayer >=2.0.0

"""
import time

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
import tensorlayer as tl

from common.utils import *
from common.distributions import make_dist
from common.value_networks import *
from common.policy_networks import *


EPS = 1e-8  # epsilon


class PPO_PENALTY(object):
    """
    PPO class
    """

    def __init__(self, net_list, optimizers_list, kl_target=0.01, lam=0.5):
        """
        :param net_list: a list of networks (value and policy) used in the algorithm, from common functions or customization
        :param optimizers_list: a list of optimizers for all networks and differentiable variables
        :param kl_target: controls bounds of policy update and adaptive lambda
        :param lam:  KL-regularization coefficient
        """
        assert len(net_list) == 2
        assert len(optimizers_list) == 2

        self.name = 'PPO_PENALTY'

        self.critic, self.actor = net_list

        assert isinstance(self.critic, ValueNetwork)
        assert isinstance(self.actor, StochasticPolicyNetwork)

        self.kl_target = kl_target
        self.lam = lam

        self.critic_opt, self.actor_opt = optimizers_list
        self.old_dist = make_dist(self.actor.action_space)

        self.batch_counter = 0
        self.buffer_s, self.buffer_a, self.buffer_r, = [], [], []
        self.batch_s, self.batch_a, self.batch_r = [], [], []

    def a_train(self, tfs, tfa, tfadv, oldpi_prob):
        """
        Update policy network
        :param tfs: state
        :param tfa: act
        :param tfadv: advantage
        :return:
        """
        tfs = np.array(tfs, np.float32)
        tfa = np.array(tfa, np.float32)
        tfadv = np.array(tfadv, np.float32)

        with tf.GradientTape() as tape:
            _ = self.actor(tfs)
            pi_prob = tf.exp(self.actor.policy_dist.logp(tfa))
            ratio = pi_prob / (oldpi_prob + EPS)

            surr = ratio * tfadv
            kl = self.old_dist.kl(self.actor.policy_dist.get_param())
            # kl = tfp.distributions.kl_divergence(oldpi, pi)
            kl_mean = tf.reduce_mean(kl)
            aloss = -(tf.reduce_mean(surr - self.lam * kl))
        a_gard = tape.gradient(aloss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(a_gard, self.actor.trainable_weights))

        return kl_mean

    def c_train(self, tfdc_r, s):
        """
        Update actor network
        :param tfdc_r: cumulative reward
        :param s: state
        :return: None
        """
        tfdc_r = np.array(tfdc_r, dtype=np.float32)
        with tf.GradientTape() as tape:
            v = self.critic(s)
            advantage = tfdc_r - v
            closs = tf.reduce_mean(tf.square(advantage))
        grad = tape.gradient(closs, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(grad, self.critic.trainable_weights))

    def cal_adv(self, tfs, tfdc_r):
        """
        Calculate advantage
        :param tfs: state
        :param tfdc_r: cumulative reward
        :return: advantage
        """
        tfdc_r = np.array(tfdc_r, dtype=np.float32)
        advantage = tfdc_r - self.critic(tfs)
        return advantage.numpy()

    def update(self, a_update_steps, c_update_steps):
        """
        Update parameter with the constraint of KL divergent
        :param s: state
        :param a: act
        :param r: reward
        :return: None
        """

        adv = self.cal_adv(self.batch_s, self.batch_r)
        # adv = (adv - adv.mean())/(adv.std()+1e-6)  # normalize advantage, sometimes helpful

        _ = self.actor(self.batch_s)
        oldpi_prob = tf.exp(self.actor.policy_dist.logp(self.batch_a))
        oldpi_prob = tf.stop_gradient(oldpi_prob)

        oldpi_param = self.actor.policy_dist.get_param()
        oldpi_param = tf.stop_gradient(oldpi_param)
        self.old_dist.set_param(oldpi_param)

        for _ in range(a_update_steps):
            kl = self.a_train(self.batch_s, self.batch_a, adv, oldpi_prob)
            if kl > 4 * self.kl_target:  # this in in google's paper
                break
        if kl < self.kl_target / 1.5:  # adaptive lambda, this is in OpenAI's paper
            self.lam /= 2
        elif kl > self.kl_target * 1.5:
            self.lam *= 2
        self.lam = np.clip(self.lam, 1e-4, 10)  # sometimes explode, this clipping is MorvanZhou's solution

        # update critic
        for _ in range(c_update_steps):
            self.c_train(self.batch_r, self.batch_s)

        self.batch_counter = 0
        self.buffer_s, self.buffer_a, self.buffer_r, = [], [], []
        self.batch_s, self.batch_a, self.batch_r = [], [], []

    def get_action(self, s):
        """
        Choose action
        :param s: state
        :return: clipped act
        """
        return self.actor([s])[0].numpy()

    def get_action_greedy(self, s):
        """
        Choose action
        :param s: state
        :return: clipped act
        """
        return self.actor([s], greedy=True)[0].numpy()

    def get_v(self, s):
        """
        Compute value
        :param s: state
        :return: value
        """
        s = s.astype(np.float32)
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.critic(s)[0, 0]

    def save_ckpt(self):
        """
        save trained weights
        :return: None
        """
        save_model(self.actor, 'actor', self.name, )
        save_model(self.critic, 'critic', self.name, )

    def load_ckpt(self):
        """
        load trained weights
        :return: None
        """
        load_model(self.actor, 'actor', self.name, )
        load_model(self.critic, 'critic', self.name, )

    def store_transition(self, s, a, r):
        self.batch_counter += 1
        self.buffer_s.append(s)
        self.buffer_a.append(a)
        self.buffer_r.append(r)

    def finish_path(self, s_, gamma):
        try:
            v_s_ = self.get_v(s_)
        except:
            v_s_ = self.get_v(s_[np.newaxis, :])   # for raw-pixel input
        discounted_r = []
        for r in self.buffer_r[::-1]:
            v_s_ = r + gamma * v_s_
            discounted_r.append(v_s_)
        discounted_r.reverse()
        bs = self.buffer_s if len(self.buffer_s[0].shape)>1 else np.vstack(self.buffer_s) # no vstack for raw-pixel input
        ba, br = np.vstack(self.buffer_a), np.array(discounted_r)[:, np.newaxis]
        self.batch_s.extend(bs)
        self.batch_a.extend(ba)
        self.batch_r.extend(br)
        self.buffer_s, self.buffer_a, self.buffer_r, = [], [], []

    def learn(self, env, train_episodes=200, test_episodes=100, max_steps=200, save_interval=10,
              gamma=0.9, mode='train', render=False, batch_size=32, a_update_steps=10, c_update_steps=10):
        """
        learn function
        :param env: learning environment
        :param train_episodes: total number of episodes for training
        :param test_episodes: total number of episodes for testing
        :param max_steps: maximum number of steps for one episode
        :param save_interval: time steps for saving
        :param gamma: reward discount factor
        :param mode: train or test
        :param render: render each step
        :param batch_size: update batch size
        :param a_update_steps: actor update iteration steps
        :param c_update_steps: critic update iteration steps
        :return: None
        """

        t0 = time.time()

        if mode == 'train':
            reward_buffer = []
            for ep in range(1, train_episodes + 1):
                s = env.reset()
                ep_rs_sum = 0
                for t in range(max_steps):  # in one episode
                    if render:
                        env.render()
                    a = self.get_action(s)

                    s_, r, done, _ = env.step(a)
                    self.store_transition(s, a, r)
                    s = s_
                    ep_rs_sum += r

                    if t == max_steps - 1 or done:
                        self.finish_path(s_, gamma)

                    # update ppo
                    if self.batch_counter % batch_size == 0:
                        if len(self.buffer_a):
                            self.finish_path(s_, gamma)
                        self.update(a_update_steps, c_update_steps)

                    if done:
                        break

                print(
                    'Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                        ep, train_episodes, ep_rs_sum,
                        time.time() - t0
                    )
                )

                reward_buffer.append(ep_rs_sum)
                if ep and not ep % save_interval:
                    self.save_ckpt()
                    plot_save_log(reward_buffer, Algorithm_name=self.name, Env_name=env.spec.id)

            self.save_ckpt()
            plot_save_log(reward_buffer, Algorithm_name=self.name, Env_name=env.spec.id)

        # test
        elif mode == 'test':
            self.load_ckpt()
            for eps in range(test_episodes):
                ep_rs_sum = 0
                state = env.reset()
                for step in range(max_steps):
                    if render:
                        env.render()
                    action = self.get_action_greedy(state)
                    state, reward, done, info = env.step(action)
                    ep_rs_sum += reward
                    if done:
                        break

                print('Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    eps, test_episodes, ep_rs_sum, time.time() - t0)
                )
        else:
            print('unknown mode type')

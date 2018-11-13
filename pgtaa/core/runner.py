import os
import time
import threading
import logging
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import OrderedDict

from pgtaa.core.agents import BaHAgent
from pgtaa.core.colorized import ColourHandler
import pgtaa.config as cfg


#TODO: add validation set (maybe k-fold cross val)


logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join(os.path.join(cfg.MODEL_DIR, "tmp"), "runner.log"),
    filemode='w',
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S')

logger = logging.getLogger(__name__)

# add stream handler
ch = ColourHandler()

# modify stream handler log format
formatter_ch = logging.Formatter('%(levelname)s - %(message)s')

ch.setFormatter(formatter_ch)

# set stream handler log level
ch.setLevel(logging.INFO)

logger.addHandler(ch)


"""
Runner file:
Environment <-> Runner <-> Agent
Some issues with multithreading -> random seeds are not thread safe.
"""


class BasicRunner(object):
    def __init__(
            self,
            epochs: int=50,
            trn_episodes: int=100,
            val_episodes: int = 100,
            test_episodes: int = 100,
            horizon: int=20,
            verbose: int=0,
            model_path: str=None,
            seed: int=92,
            inference: bool=False
    ):
        """
        Args:
            :param epochs: max epochs
            :param episodes: max episodes
            :param horizon: investment horizon
            :param verbose: console printing level
            :param model_path: path for model saves
            :param seed: seed for random number generator
            :param inference: inference or training mode
        """
        self.epochs = epochs
        self.trn_episodes = trn_episodes
        self.val_episodes = val_episodes
        self.test_episodes = test_episodes
        self.horizon = horizon
        self.verbose = verbose
        self.seed = seed
        self.inference = inference
        if self.inference:
            # for testing only one epoch is required
            self.epochs = 1
            self.episodes = self.test_episodes
        else:
            self.episodes = self.trn_episodes


        if model_path is None:
            self.model_path = os.path.join(os.getcwd(), "models", "saves")
        else:
            self.model_path = model_path

        self.start = None
        self.history = []
        self.val_history = []

        # has to be a small number in case of only negative rewards
        self.global_epoch_reward = -100

    def __str__(self):
        return str(self.__class__.__name__)

    def _run(self):
        # must be overridden in child class
        return NotImplementedError

    def _close(self):
        # must be overridden in child class
        return NotImplementedError

    def episode_finished(self, info, episode=1, thread_id=None):
        """
        Args:
            :param info: (object) list -> [reward, value]
            :param episode: (int) current episode
            :param thread_id: (int) optional thread id
        """
        if self.verbose >= 2:
            if thread_id is not None:
                print(f'Worker {thread_id} finished episode {episode + 1}. '
                      f'Reward: {info[0]}, '
                      f'Portfolio value: {info[1]}, '
                      f'Portfolio return: {info[2]}')
            else:
                print(f'Finished episode {episode + 1}. '
                      f'Reward: {info[0]}, '
                      f'Portfolio value: {info[1]}, '
                      f'Portfolio return: {info[2]}')

    def epoch_finished(self, epoch=1, thread_id=None):
        """
        Args:
            :param epoch: (int) current epoch
            :param thread_id: (int) optional thread id
        """
        if self.verbose < 1:
        #TODO: remove the print statements!
            if thread_id is not None:
                print(f'Worker {thread_id} finished epoch {epoch + 1}: '
                      f'Average reward: {np.mean(self.history[-1][:, 0], axis=0)}, '
                      f'Average return: {np.mean(self.history[-1][:, 2], axis=0)}, '
                      f'Average value {np.mean(self.history[-1][:, 1], axis=0)}')
            else:
                print(f'Finished epoch {epoch + 1}: '
                      f'Average reward: {np.mean(self.history[-1][:, 0], axis=0)}, '
                      f'Average return: {np.mean(self.history[-1][:, 2], axis=0)}, '
                      f'Average value {np.mean(self.history[-1][:, 1], axis=0)}')

        else:
            columns = ['Episode Reward', 'Portfolio Value',
                       'Portfolio Return', 'Sharpe Ratio',
                       'Portfolio Variance', 'Cumulative Costs']

            df = pd.DataFrame(self.history[-1], columns=columns,
                              index=range(1, self.episodes + 1))

            mean = pd.DataFrame(df.mean(axis=0), columns=['Average'])
            std = pd.DataFrame(df.std(axis=0), columns=['Std Deviation'])
            maximum = pd.DataFrame(df.max(axis=0), columns=['Maximum'])
            minimum = pd.DataFrame(df.min(axis=0), columns=['Minimum'])

            evaluation = pd.DataFrame(pd.concat([mean, std, maximum, minimum], axis=1))

            if thread_id is not None:
                print(f'\nWorker {thread_id} finished epoch {epoch + 1}')
            else:
                print(f'\nFinished epoch {epoch + 1}')

            print(evaluation)
            logger.debug(df)
            logger.info(f'{77 * "#"}\n{evaluation}')

        if not self.inference:
            if self.verbose < 1:
                # TODO: remove the print statements!
                if thread_id is not None:
                    print(f'Average validation reward: {np.mean(self.val_history[-1][:, 0], axis=0)}, '
                          f'Average validation return: {np.mean(self.val_history[-1][:, 2], axis=0)}, '
                          f'Average value {np.mean(self.val_history[-1][:, 1], axis=0)}')
                else:
                    print(f'Average reward: {np.mean(self.val_history[-1][:, 0], axis=0)}, '
                          f'Average return: {np.mean(self.val_history[-1][:, 2], axis=0)}, '
                          f'Average value {np.mean(self.val_history[-1][:, 1], axis=0)}')

            else:
                columns = ['Episode Reward', 'Portfolio Value',
                           'Portfolio Return', 'Sharpe Ratio',
                           'Portfolio Variance', 'Cumulative Costs']

                df = pd.DataFrame(self.val_history[-1], columns=columns,
                                  index=range(1, self.val_episodes + 1))

                mean = pd.DataFrame(df.mean(axis=0), columns=['Validation Average'])
                std = pd.DataFrame(df.std(axis=0), columns=['Validation Std Deviation'])
                maximum = pd.DataFrame(df.max(axis=0), columns=['Validation Maximum'])
                minimum = pd.DataFrame(df.min(axis=0), columns=['Validation Minimum'])

                evaluation = pd.DataFrame(pd.concat([mean, std, maximum, minimum], axis=1))

                print(evaluation)
                logger.debug(df)
                logger.info(f'{77 * "#"}\n{evaluation}')


    def run(self):
        self.start = time.time()
        return self._run()

    def close(self, save_full: str=None, save_short: str=None, save_full_val: str=None, save_short_val: str=None):
        """
        Args
            :param save_full: path for full epoch evaluation file if not None
            :param save_short:  path for small epoch evaluation file if not None
            :param save_full_val: path for full validation epoch evaluation file if not None
            :param save_short_val: path for small validation epoch evaluation file if not None

        :return: _close() method of child class
        """
        columns = ['Episode Reward', 'Portfolio Value',
                   'Portfolio Return', 'Sharpe Ratio',
                   'Portfolio Variance', 'Cumulative Costs']

        if len(self.history) == 1:
            evaluation = pd.DataFrame(self.history[0], index=range(1, self.episodes + 1),
                                      columns=columns).rename_axis('Episode')
        else:
            evaluation = pd.DataFrame(
                pd.concat([pd.DataFrame(epoch, index=range(1, self.episodes + 1),
                                        columns=columns).rename_axis('Episode') for epoch in self.history],
                          axis=1, keys=['Epoch ' + str(i + 1) for i in range(len(self.history))]))

        logger.info(f'{77 * "#"}\n{evaluation}')
        if self.verbose >= 2:
            print('\n', evaluation)

        mean = pd.DataFrame(evaluation.mean(axis=0), columns=['Average'])
        std = pd.DataFrame(evaluation.std(axis=0), columns=['Std Deviation'])
        var = pd.DataFrame(evaluation.var(axis=0), columns=['Variance'])
        maximum = pd.DataFrame(evaluation.max(axis=0), columns=['Maximum'])
        minimum = pd.DataFrame(evaluation.min(axis=0), columns=['Minimum'])

        short = pd.DataFrame(pd.concat([mean, std, var, maximum, minimum], axis=1))

        logger.info(f'{77 * "#"}\n{short}')
        print(short)

        # save evaluation files
        if save_full:
            # contains episode rewards etc
            evaluation.to_csv(save_full)

        if save_short:
            # contains epoch averages etc
            short.to_csv(save_short)

        if not self.inference:
            if len(self.val_history) == 1:
                evaluation = pd.DataFrame(self.val_history[0], index=range(1, self.val_episodes + 1),
                                          columns=columns).rename_axis('Episode')
            else:
                evaluation = pd.DataFrame(
                    pd.concat([pd.DataFrame(epoch, index=range(1, self.val_episodes + 1),
                                            columns=columns).rename_axis('Episode') for epoch in self.val_history],
                              axis=1, keys=['Epoch ' + str(i + 1) for i in range(len(self.val_history))]))

            logger.info(f'{77 * "#"}\nValidation:{evaluation}')
            if self.verbose >= 2:
                print('\n Validation:', evaluation)

            mean = pd.DataFrame(evaluation.mean(axis=0), columns=['Validation Average'])
            std = pd.DataFrame(evaluation.std(axis=0), columns=['Validation Std Deviation'])
            var = pd.DataFrame(evaluation.var(axis=0), columns=['Validation Variance'])
            maximum = pd.DataFrame(evaluation.max(axis=0), columns=['Validation Maximum'])
            minimum = pd.DataFrame(evaluation.min(axis=0), columns=['Validation Minimum'])

            short = pd.DataFrame(pd.concat([mean, std, var, maximum, minimum], axis=1))

            logger.info(f'{77 * "#"}\nValidation:{short}')
            print(f'Validation:{short}')

            # save evaluation files
            if save_full_val:
                # contains episode rewards etc
                evaluation.to_csv(save_full_val)

            if save_short_val:
                # contains epoch averages etc
                short.to_csv(save_short_val)

        return self._close()


class Runner(BasicRunner):
    def __init__(
            self,
            agent,              #TODO set to agent object
            environment,        #TODO set env to env object
            epochs: int=50,
            trn_episodes: int=100,
            val_episodes: int = 100,
            test_episodes: int = 100,
            horizon: int=20,
            inference: bool=False,
            verbose: int=0,
            model_path: str=None,
            seed: int=92
    ):
        """
        Args:
            :param agent: (object) rl or basic agent
            :param environment: (object) portfolio environment for agent
            :param mode: (str) specify run mode -> 'train', 'test'
        """
        super(Runner, self).__init__(
            epochs=epochs,
            trn_episodes=trn_episodes,
            val_episodes=val_episodes,
            test_episodes=test_episodes,
            horizon=horizon,
            verbose=verbose,
            model_path=model_path,
            seed=seed,
            inference=inference
        )

        self.agent = agent
        self.environment = environment
        self.model_path = os.path.join(self.model_path, str(self.agent))

    def _run(self):

        for epoch in range(self.epochs):

# -------------------------- start epoch -----------------------------------------#

            # reset the environment random seed -> same episode start points for next epoch
            self.environment.seed(seed=self.seed)

            # after an epoch finished return same episode entry point order
            state = self.environment.reset_epoch()

            performance = OrderedDict(reward=[], value=[], returns=[], sharpe=[], variance=[], costs=[])

            with tqdm(total=self.episodes) as pbar:
                for episode in range(self.episodes):
                    # reset the agent
                    self.agent.reset()
                    # reset the episode reward
                    episode_reward = 0

                    for step in range(self.horizon):

                        # get next step and results for the taken action
                        if not self.inference:
                            # do action based on observation
                            action = self.agent.act(state, deterministic=True)

                            # for testing no agent observations regarding the reward are required
                            result = self.environment.execute(action)
                        else:
                            # do action based on observation
                            action = self.agent.act(state, deterministic=False)

                            result = self.environment.execute(action)

                            # agent receives new observation and the result of his action(s)
                            self.agent.observe(terminal=result.done, reward=result.reward)

                        # update information regarding the current portfolio
                        info = result.info['info']

                        # increase episode reward by step reward
                        episode_reward += result.reward

                        # update state
                        state = result.state

                    # saves the episode results
                    performance["reward"].append(episode_reward)
                    performance["value"].append(info.portfolio_value)
                    performance["return"].append(performance["value"][-1] /
                                                 self.environment.init_portfolio_value - 1)
                    performance["sharpe"].append(info.sharpe_ratio)
                    performance["variance"].append(info.portfolio_variance)
                    performance["costs"].append(self.environment.episode_costs)

                    # log episode performance
                    self.episode_finished([performance["reward"][-1],
                                           performance["value"][-1],
                                           performance["returns"][-1]],
                                          episode=episode)

                    # update progressbar
                    pbar.update(1)

                    # reset the environment
                    if episode <= self.episodes:
                        state = self.environment.reset()

# --------------------------------- validation ---------------------------------- #

            # epoch finished -> validate agent
            if not self.inference:

                val_performance = OrderedDict(reward=[], value=[], returns=[], sharpe=[], variance=[], costs=[])

                for episode in range(self.val_episodes):
                    # reset the agent
                    self.agent.reset()
                    # reset the episode reward
                    episode_reward = 0

                    for step in range(self.horizon):
                        # do action based on observation current policy
                        action = self.agent.act(state, deterministic=True)

                        # for testing no agent observations regarding the reward are required
                        result = self.environment.execute(action)

                        # update information regarding the current portfolio
                        info = result.info['info']

                        # increase episode reward by step reward
                        episode_reward += result.reward

                        # update state
                        state = result.state

                    # saves the episode results
                    val_performance["reward"].append(episode_reward)
                    val_performance["value"].append(info.portfolio_value)
                    val_performance["return"].append(performance["value"][-1] /
                                                     self.environment.init_portfolio_value - 1)
                    val_performance["sharpe"].append(info.sharpe_ratio)
                    val_performance["variance"].append(info.portfolio_variance)
                    val_performance["costs"].append(self.environment.episode_costs)

                    # log episode performance
                    self.episode_finished([val_performance["reward"][-1],
                                           val_performance["value"][-1],
                                           val_performance["returns"][-1]],
                                          episode=episode)

                    # reset the environment
                    if episode <= self.episodes:
                        state = self.environment.reset()

# ---------------------------------- epoch finished -------------------------------------------------------------------#

                # epoch finished -> if that is the best epoch so far (average reward based on validation set):
                # -> save the model
                mean = np.mean(val_performance["reward"])
                if mean > self.global_epoch_reward:
                    try:
                        self.agent.save_model(directory=os.path.join(self.model_path, str(self.agent)))
                        logger.info(f'Agent has been saved to {os.path.join(self.model_path, str(self.agent))}')
                        self.global_epoch_reward = mean
                    except AttributeError:
                        pass
                    except FileNotFoundError:
                        if not os.path.isdir(self.model_path):
                            try:
                                os.mkdir(self.model_path, 0o755)
                            except OSError:
                                raise OSError("Cannot save agent to dir {} ()".format(self.model_path))
                        self.agent.save_model(directory=os.path.join(self.model_path, str(self.agent)))
                        logger.info(f'Agent has been saved to {os.path.join(self.model_path, str(self.agent))}')
                        print('\n> Agent has been saved')
                        self.global_epoch_reward = mean
                    except Exception as e:
                        logger.error(e)

                self.val_history.append(np.array([v for v in val_performance.values()]).T)
            self.history.append(np.array([v for v in performance.values()]).T)

            self.epoch_finished(epoch=epoch)

        print('Finished run. Time: {}'.format(time.time() - self.start))

    def _close(self):
        # close agent and environment
        self.agent.close()
        self.environment.close()


if __name__ == '__main__':
    pass
    """
    Run the Buy and Hold Agent on the environment to see if the runner and environment 
    are correctly working.
    """
    # import pgtaa.config as cfg
    # from pgtaa.environment.env import PortfolioEnv
    # from pgtaa.core.utils import PrepData

    """
    prep = PrepData(horizon=10,
                    window_size=cfg.WINDOW_SIZE,
                    nb_assets=cfg.NB_ASSETS,
                    split=cfg.TRAIN_SPLIT)
    data = prep.get_data(cfg.ENV_DATA)  # get data file and extract data
    train = data[0:int(cfg.TRAIN_SPLIT * data.shape[0])]  # train/test split
    scaler = prep.get_scaler(train)
    agent = BaHAgent(action_shape=(10,))
    env = PortfolioEnv(data, scaler=scaler, action_type='signal', random_starts=False)
    run = Runner(agent, env, mode='test', verbose=2, episodes=cfg.EPISODES, epochs=cfg.EPOCHS)
    run.run()
    run.close(
        save_full=os.path.join(cfg.RUN_DIR, 'train') + '/full_BuyAndHoldAgent_evaluation.csv',
        save_short=os.path.join(cfg.RUN_DIR, 'train') + '/short_BuyAndHoldAgent_evaluation.csv')
    """

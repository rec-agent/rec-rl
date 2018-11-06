from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import inspect
import json
import logging
import os
import sys
import time

import tensorflow as tf
from six.moves import xrange, shlex_quote

path = os.path.abspath('.')
sys.path.append(path)

from tensorforce import TensorforceError
from tensorforce.agents import Agent
from rec_env import RecTableEnv

"""
# example command
python examples/rec_es/rec_run_es_local.py -i examples/rec_es/rec_config_local.json
"""

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--config', help="Configuration file")

    args = parser.parse_args()
    print(args)
    sys.stdout.flush()

    if args.config is not None:
        with open(args.config, 'r') as fp:
            config = json.load(fp=fp)
    else:
        raise TensorforceError("No configuration provided.")

    if 'agent' not in config:
        raise TensorforceError("No agent configuration provided.")
    else:
        agent_config = config['agent']

    if 'network_spec' not in config:
        network_spec = None
        print("No network configuration provided.")
    else:
        network_spec = config['network_spec']

    if 'env' not in config:
        raise TensorforceError("No environment configuration provided.")
    else:
        env_config = config['env']

    environment = RecTableEnv(config)
    environment.set_up()

    agent_config['env'] = environment

    agent = Agent.from_spec(
        spec=agent_config,
        kwargs=dict(
            states_spec=environment.states,
            actions_spec=environment.actions,
            network_spec=network_spec,
            batch_data=environment.get_input_tensor()
        )
    )

    environment.set_session(agent.model.get_session())

    print("********** Configuration ************")
    for key, value in agent_config.items():
        print(str(key) + ": {}".format(value))

    agent.run_worker()
    agent.close()


if __name__ == '__main__':
    logging.info("start...")
    print('start')
    main()

from __future__ import absolute_importfrom __future__ import divisionfrom __future__ import print_functionimport argparseimport inspectimport jsonimport loggingimport osimport sysimport timeimport tensorflow as tffrom six.moves import xrange, shlex_quotepath = os.path.abspath('../../')sys.path.append(path)from tensorforce import TensorforceErrorfrom tensorforce.agents import Agentfrom rec_env import RecTableEnv"""# example commandpython examples/rec_es/rec_run_es_local.py -i examples/rec_es/rec_config_local.json"""def main():    parser = argparse.ArgumentParser()    #传入json文件地址    parser.add_argument('-i', '--config', help="Configuration file")    args = parser.parse_args()    print(args)    sys.stdout.flush()    #读取json文件，并赋值给config    if args.config is not None:        with open(args.config, 'r') as fp:            config = json.load(fp=fp)    else:        raise TensorforceError("No configuration provided.")    #从config字典中，为agent_config，network_spec，env_config赋值    if 'agent' not in config:        raise TensorforceError("No agent configuration provided.")    else:        agent_config = config['agent']    if 'network_spec' not in config:        network_spec = None        print("No network configuration provided.")    else:        network_spec = config['network_spec']    if 'env' not in config:        raise TensorforceError("No environment configuration provided.")    else:        env_config = config['env']    #实例化一个environment,parse或预设各种config，including env_config, states, actions    environment = RecTableEnv(config)    #set_up：读取数据，并通过build_graph建立graph    environment.set_up()    #在agent_config中添加关于env的信息    agent_config['env'] = environment    #states: {'state': {'type': 'float', 'shape': (8,)}}    #actions: {'action': {'type': 'float', 'shape': (2,), 'min_value': -1.0, 'max_value': 2.0}}    #实例化一个DeterminisitcESAgent，通过util.get_object()    agent = Agent.from_spec(        spec=agent_config,        kwargs=dict(            states_spec=environment.states,            actions_spec=environment.actions,            network_spec=network_spec,            batch_data=environment.get_input_tensor()        )    )    #set_session 需要传入一个session    #实例化DeterministicESAgent时，初始化了self.model.initialize_model()此方法被子类重写    #该方法返回一个DeterministicESModel的实例    #    environment.set_session(agent.model.get_session())    # print("********** Configuration ************")    # for key, value in agent_config.items():    #     print(str(key) + ": {}".format(value))    #run_worker调用self.model.update()    agent.run_worker()        agent.close()if __name__ == '__main__':    logging.info("start...")    print('start')    main()

config_args = {
    "ima":{
        "step": 8,
        'sample_num': 4,
    },
    "cifar10":{
        "step": 10,
        'sample_num': 5,
    },
    "cifar100":{
        "step": 10,
        'sample_num': 5,
    }
}


def auto_config(args):
    """
    Automatically configures some arguments.
    """
    if 'ima' in  args.dataset:
        dataset = 'ima'
    else:
        dataset = args.dataset
    args_dict = vars(args)
    if 'atk_pgd_step_size' in args_dict.keys():
        args.atk_pgd_step_size= args.atk_pgd_radius / (config_args[dataset]['step']/2)
        args.atk_pgd_steps = config_args[dataset]['step']
        args.atk_pgd_random_start = True
    if 'pgd_step_size' in args_dict.keys():
        args.pgd_step_size= args.pgd_radius / (config_args[dataset]['step']/2)
        args.pgd_steps = config_args[dataset]['step']
        args.pgd_random_start = True
    if 'samp_num' in args_dict.keys():
        args.samp_num = config_args[dataset]['sample_num']
    args.save_dir = f'./exp_data/{args.exp_name}/{args.state}_{args.exp_hyper}/'
    return args
    
    
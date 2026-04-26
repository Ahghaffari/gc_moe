import os
import time
import argparse
import numpy as np
import torch.optim as optim
from src.__init__ import *
import src.loralib as lora
import wandb

def main():
    start_time = time.time()
    args = get_config()
    args.log_dir = './experiments/{}/'.format(args.model)
    init_seed(args.seed)
    device = torch.device(args.device)
    args.data_path, args.adj_path, args.node_num = get_dataset_info(args.dataset)
    
    # wandb logging
    if args.use_wandb:
        run_name = f"{args.model}_{args.dataset}_{args.years}_seed{args.seed}"
        if hasattr(args, 'run_id') and args.run_id:
            run_name = f"{args.run_id}"
        if args.model in ['moe', 'moe_stlora'] and hasattr(args, 'router_type'):
            run_name += f"_rt-{args.router_type}"
        wandb.init(
            project=getattr(args, 'wandb_project', None) or os.environ.get('WANDB_PROJECT', 'gc-moe'),
            name=run_name,
            config={k: v for k, v in vars(args).items() if not callable(v) and k not in ('logger', 'dataloader', 'scaler', 'loss_fn', 'supports')}
        )
    args.logger.info('Adj path: ' + args.adj_path)

    args.adj_mx = load_adj_from_numpy(args.adj_path)
    if args.nor_adj:
        args.adj_mx = normalize_adj_mx(args.adj_mx, args.adj_type)
    args.supports = [torch.tensor(i).to(args.device) for i in args.adj_mx]

    args.loss_fn = masked_mae
    args.dataloader, args.scaler = load_dataset(args.data_path, args)
    engine = get_engine(args)

    if args.backbone_lora:
        include = set([s for s in args.lora_include.split(',') if s]) if hasattr(args, 'lora_include') else None
        exclude = set([s for s in args.lora_exclude.split(',') if s]) if hasattr(args, 'lora_exclude') else None
        from src.loralib.inject import apply_lora_to_module
        engine.model = apply_lora_to_module(
            engine.model,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            include_names=include,
            exclude_names=exclude,
        )
        engine.model.to(engine._device)

    if args.stlora:
        engine.model = STLoRA(device=args.device,
                                node_num=args.node_num,
                                input_dim=args.input_dim,
                                output_dim=args.output_dim,
                                horizon=args.horizon,
                                model=engine.model,
                                supports=args.supports,
                                frozen=args.frozen,
                                lagcn=args.lagcn,
                                embed_dim=args.embed_dim,
                                num_layers = args.num_nalls,
                                num_blocks = args.num_lablocks,
                                la_dropout=args.last_dropout,
                                last_lr=args.last_lr,
                                last_weight_decay=args.last_weight_decay,
                                last_pool_type=args.last_pool_type,
                                linear=args.linear
                                )
        engine.model.to(engine._device)
        engine._optimizer = engine.model.optimizer
        engine._lr_scheduler = engine.model.scheduler
    
    if args.pre_train:
        pretrained_dict = torch.load('./save/'+args.pre_train, map_location=device)
        
        if args.stlora:
            new_state_dict = {}
            for k, v in pretrained_dict.items():
                new_key = f'pre_model.{k}'
                new_state_dict[new_key] = v
            engine.model.load_state_dict(new_state_dict, strict=False)
            args.logger.info('Loaded pretrained backbone into STLoRA wrapper (pre_model)')
        else:
            engine.model.load_state_dict(pretrained_dict, strict=False)
            args.logger.info('Loaded pretrained weights')
    
    train_time = time.time()
    if args.mode == 'train':
        engine.train()
        
        if args.model in ['moe', 'moe_stlora']:
            args.logger.info('\nTraining completed! Evaluating on test set...')
            test_results = engine.test()
            
            args.logger.info('\n' + '='*70)
            args.logger.info('Final Test Results:')
            args.logger.info('='*70)
            for metric, value in test_results.items():
                if not isinstance(value, (list, np.ndarray)):
                    args.logger.info(f'  {metric}: {value:.4f}')
            args.logger.info('='*70)
    else:
        engine.evaluate(args.mode)
    
    if args.save:
        if not os.path.exists('./save'):
            os.makedirs('./save')
        torch.save(engine.model.state_dict(), './save/'+args.save)


    print_trainable_parameters(engine.model)
    print(args.model, args.mode, "finished!!")
    end_time = time.time()
    print("total run time: {} s".format(end_time - start_time))
    print("total train time: {} s".format(end_time - train_time))
    
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

import torch
import torch.distributed as dist
from swift import (HubStrategy, LoraConfig, Seq2SeqTrainer,
                   Seq2SeqTrainingArguments, Swift, get_logger)
from swift.llm.utils import data_collate_fn, print_example, stat_dataset
from swift.llm.utils.model import fix_gradient_checkpointing_warning
from swift.utils import (add_version_to_work_dir, get_model_info, is_master,
                         parse_args, seed_everything)
from transformers import BitsAndBytesConfig
from utils import (DEFAULT_PROMPT, MODEL_MAPPING, broadcast_string,
                   find_all_linear_for_lora, get_dist_setting,
                   get_model_tokenizer, get_ms_tool_dataset, is_dist,
                   plot_images, process_dataset, select_bnb, select_dtype,
                   show_layers, tokenize_function)

logger = get_logger()


@dataclass
class SftArguments:
    model_type: str = field(
        default='qwen-7b', metadata={'choices': list(MODEL_MAPPING.keys())})
    # qwen-7b: lora+4bitQ: 10G, lora+8bitQ: 14G, lora: 22G; full: 95G
    sft_type: str = field(
        default='lora', metadata={'choices': ['lora', 'full']})
    output_dir: str = 'runs'
    # currently, DDP+MP is not supported
    ddp_backend: Optional[str] = field(
        default=None, metadata={'choices': ['nccl', 'gloo', 'mpi', 'ccl']})

    seed: int = 42
    resume_from_ckpt: Optional[str] = None
    dtype: str = field(
        default='fp16', metadata={'choices': {'bf16', 'fp16', 'fp32'}})
    ignore_args_error: bool = False  # True: notebook compatibility

    dataset: str = field(
        default='alpaca-en,alpaca-zh', metadata={'help': 'dataset'})
    dataset_seed: int = 42
    dataset_sample: int = 20000  # -1: all dataset
    dataset_test_size: float = 0.01
    prompt: str = DEFAULT_PROMPT
    max_length: Optional[int] = 1024

    # If you want to use qlora, set the quantization_bit to 8 or 4.
    # And you need to install bitsandbytes: `pip install bitsandbytes -U`
    # note: bf16 and quantization have requirements for gpu architecture
    quantization_bit: Optional[int] = field(
        default=None, metadata={'choices': {4, 8}})
    bnb_4bit_comp_dtype: str = field(
        default='fp16', metadata={'choices': {'fp16', 'bf16', 'fp32'}})
    bnb_4bit_quant_type: str = field(
        default='nf4', metadata={'choices': {'fp4', 'nf4'}})
    bnb_4bit_use_double_quant: bool = True

    lora_target_modules: Optional[List[str]] = None
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout_p: float = 0.1

    gradient_checkpoint: bool = True
    batch_size: int = 1
    num_train_epochs: int = 1
    optim: str = 'adamw_torch'
    learning_rate: Optional[float] = None
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 16
    max_grad_norm: float = 1.
    lr_scheduler_type: str = 'cosine'
    warmup_ratio: float = 0.1

    eval_steps: int = 50
    save_steps: Optional[int] = None
    save_total_limit: int = 2
    logging_steps: int = 5

    skip_memory_metrics: bool = True

    push_to_hub: bool = False
    # 'user_name/repo_name' or 'repo_name'
    hub_model_id: Optional[str] = None
    hub_private_repo: bool = True
    hub_strategy: HubStrategy = HubStrategy.EVERY_SAVE
    # None: use env var `MODELSCOPE_API_TOKEN`
    hub_token: Optional[str] = None

    # fsdp
    fsdp: Optional[str] = None
    fsdp_config: Optional[str] = None

    # deepspeed
    deepspeed: Optional[str] = None

    # other
    use_flash_attn: Optional[bool] = field(
        default=None,
        metadata={
            'help': "This parameter is used only when model_type='qwen-7b'"
        })

    def __post_init__(self):
        if is_dist():
            rank, _, _ = get_dist_setting()
            self.seed += rank  # Avoid the same dropout
            if self.ddp_backend is None:
                self.ddp_backend = 'nccl'
            # Initialize in advance
            dist.init_process_group(backend=self.ddp_backend)

        if self.sft_type == 'lora':
            if self.learning_rate is None:
                self.learning_rate = 1e-4
            if self.save_steps is None:
                self.save_steps = self.eval_steps
        elif self.sft_type == 'full':
            assert self.quantization_bit is None, 'not supported'
            assert self.dtype != 'fp16', 'please use bf16 or fp32'
            if self.learning_rate is None:
                self.learning_rate = 1e-5
            if self.save_steps is None:
                # Saving the model takes a long time
                self.save_steps = self.eval_steps * 4
        else:
            raise ValueError(f'sft_type: {self.sft_type}')

        self.output_dir = os.path.join(self.output_dir, self.model_type)

        if self.lora_target_modules is None:
            self.lora_target_modules = MODEL_MAPPING[
                self.model_type]['lora_TM']
        self.torch_dtype, self.fp16, self.bf16 = select_dtype(self.dtype)
        self.bnb_4bit_compute_dtype, self.load_in_4bit, self.load_in_8bit = select_bnb(
            self.quantization_bit, self.bnb_4bit_comp_dtype)

        if self.hub_model_id is None:
            self.hub_model_id = f'{self.model_type}-sft'
        if self.use_flash_attn is None:
            self.use_flash_attn = 'auto'


def llm_sft(args: SftArguments) -> None:
    logger.info(f'device_count: {torch.cuda.device_count()}')
    rank, local_rank, world_size = get_dist_setting()
    logger.info(
        f'rank: {rank}, local_rank: {local_rank}, world_size: {world_size}')
    seed_everything(args.seed)

    # ### Loading Model and Tokenizer
    kwargs = {'low_cpu_mem_usage': True}
    if is_dist():
        kwargs['device_map'] = {'': local_rank}
    else:
        kwargs['device_map'] = 'auto'
    if args.load_in_8bit or args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            args.load_in_8bit,
            args.load_in_4bit,
            bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant)
        logger.info(f'quantization_config: {quantization_config.__dict__}')
        kwargs['quantization_config'] = quantization_config
    if args.model_type == 'qwen-7b':
        kwargs['use_flash_attn'] = args.use_flash_attn
    model, tokenizer = get_model_tokenizer(
        args.model_type, torch_dtype=args.torch_dtype, **kwargs)

    # ### Preparing lora
    if args.sft_type == 'lora':
        if 'ALL' in args.lora_target_modules:
            assert len(args.lora_target_modules) == 1
            args.lora_target_modules = find_all_linear_for_lora(
                model, args.quantization_bit, args.model_type)
            logger.info(
                f'Setting lora_target_modules: {args.lora_target_modules}')
        if args.resume_from_ckpt is None:
            lora_config = LoraConfig(
                r=args.lora_rank,
                target_modules=args.lora_target_modules,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout_p,
                task_type='CAUSAL_LM')
            logger.info(f'lora_config: {lora_config}')
            model = Swift.prepare_model(model, lora_config)
        else:
            model = Swift.from_pretrained(
                model, args.resume_from_ckpt, is_trainable=True)

        is_logging = False
        for p in model.parameters():
            if p.requires_grad and p.dtype == torch.float16:
                if not is_logging:
                    logger.info(
                        'Convert trainable parameters from fp16 to fp32.')
                    is_logging = True
                p.data = p.data.to(dtype=torch.float32)

    # # for fsdp
    # if args.fp16:
    #     model = model.half()
    # if args.bf16:
    #     model = model.bfloat16()

    show_layers(model)
    get_model_info(model)

    # ### Loading Dataset
    dataset = get_ms_tool_dataset(args.dataset)
    train_dataset, val_dataset = process_dataset(dataset,
                                                 args.dataset_test_size,
                                                 args.dataset_sample,
                                                 args.dataset_seed)
    tokenize_func = partial(
        tokenize_function, tokenizer=tokenizer, max_length=args.max_length)
    train_dataset = train_dataset.map(tokenize_func)
    val_dataset = val_dataset.map(tokenize_func)
    del dataset
    # Data analysis
    stat_dataset(train_dataset)
    stat_dataset(val_dataset)
    data_collator = partial(data_collate_fn, tokenizer=tokenizer)
    # print_example(train_dataset[0], tokenizer)

    # ### Setting trainer_args
    output_dir = None
    if is_master():
        output_dir = add_version_to_work_dir(args.output_dir)
    if is_dist():
        output_dir = broadcast_string(output_dir)
    trainer_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        evaluation_strategy='steps',
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy='steps',
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        eval_steps=args.eval_steps,
        dataloader_num_workers=1,
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        greater_is_better=False,
        sortish_sampler=True,
        optim=args.optim,
        hub_model_id=args.hub_model_id,
        hub_private_repo=args.hub_private_repo,
        hub_strategy=args.hub_strategy,
        hub_token=args.hub_token,
        push_to_hub=args.push_to_hub,
        resume_from_checkpoint=args.resume_from_ckpt,
        ddp_backend=args.ddp_backend,
        gradient_checkpointing=args.gradient_checkpoint,
        local_rank=local_rank,
        skip_memory_metrics=args.skip_memory_metrics,
        fsdp=args.fsdp or False,
        fsdp_config=args.fsdp_config,
        deepspeed=args.deepspeed)

    if args.gradient_checkpoint:
        # fix: gradients will be None
        model.enable_input_require_grads()
        if is_dist():
            trainer_args.ddp_find_unused_parameters = False
            trainer_args.ddp_broadcast_buffers = False
    logger.info(f'trainer_args: {trainer_args}')

    trainer = Seq2SeqTrainer(
        model=model,
        args=trainer_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    train_result = trainer.train(trainer_args.resume_from_checkpoint)

    # save metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    # ### Visualization
    # if is_master():
    # images_dir = os.path.join(output_dir, 'images')
    # tb_dir = os.path.join(output_dir, 'runs')
    # folder_name = os.listdir(tb_dir)[0]
    # tb_dir = os.path.join(tb_dir, folder_name)
    # plot_images(images_dir, tb_dir, ['train/loss'], 0.9)
    # if args.push_to_hub:
    #     trainer._add_patterns_to_gitignores(['images/'])
    #     trainer.push_to_hub()


if __name__ == '__main__':
    args, remaining_argv = parse_args(SftArguments)
    if len(remaining_argv) > 0:
        if args.ignore_args_error:
            logger.warning(f'remaining_argv: {remaining_argv}')
        else:
            raise ValueError(f'remaining_argv: {remaining_argv}')
    llm_sft(args)

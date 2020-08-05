#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

# Standard Library
import fileinput
import subprocess
import math
import os
import re
import sys
from collections import namedtuple

# Third Party
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# First Party
from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.data import encoders

Batch = namedtuple('Batch', 'ids src_tokens src_lengths')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')


def page_read(input, *args):
    buffer = []
    with fileinput.input(files=[input],
                         openhook=fileinput.hook_encoded("utf-8")) as h:
        for fragment in h:
            clean_frag = fragment.rstrip()

            if clean_frag and clean_frag != ' ':
                buffer.append(clean_frag)

            if buffer:
                yield buffer
                buffer = []


def buffered_read(input, buffer_size, pagesplit=False):
    buffer = []
    with fileinput.input(files=[input],
                         openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:



            if pagesplit:
                fragments = re.split('\.|\n', src_str.rstrip())
            
            else:
                fragments = [src_str.rstrip()]




            # buffer.append(src_str.strip())
            # for fragment in re.split('\.|\n',src_str.rstrip()):

            for fragment in fragments:

                if fragment:
                    buffer.append(fragment)

                if len(buffer) >= buffer_size:
                    yield buffer
                    buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, args, task, max_positions, encode_fn):


    tokenized = [encode_fn(src_str) for src_str in lines]




    tokens = [
        task.source_dictionary.encode_line(tokenized_line,
                                           add_if_not_exist=False).long()
        for tokenized_line in tokenized
    ]

    lengths = [t.numel() for t in tokens]
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(tokens, lengths),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            ids=batch['id'],
            src_tokens=batch['net_input']['src_tokens'],
            src_lengths=batch['net_input']['src_lengths'],
        ),tokenized


def main(args):
    utils.import_user_module(args)

    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.max_sentences is None:
        args.max_sentences = 1

    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    # Fix seed for stochastic decoding
    if args.seed is not None and not args.no_seed_provided:
        np.random.seed(args.seed)
        utils.set_torch_seed(args.seed)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(args)

    # Load ensemble
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(os.pathsep),
        arg_overrides=eval(args.model_overrides),
        task=task,
        suffix=getattr(args, "checkpoint_suffix", ""),
    )

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        model.prepare_for_inference_(args)
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Initialize generator
    generator = task.build_generator(models, args)

    # Handle tokenization and BPE
    tokenizer = encoders.build_tokenizer(args)
    bpe = encoders.build_bpe(args)

    def encode_fn(x):
        if tokenizer is not None:
            x = tokenizer.encode(x)
        if bpe is not None:
            x = bpe.encode(x)
        return x

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    max_positions = utils.resolve_max_positions(
        task.max_positions(), *[model.max_positions() for model in models])

    start_id = 0


    total=int(subprocess.check_output('wc -l ' + args.input,shell=True).decode().split()[0])

    total_batches = math.ceil(total/args.buffer_size)

    for inputs in tqdm(buffered_read(args.input, args.buffer_size, pagesplit=args.pagesplit),total=total_batches):

        results = []
        for batch,tokenized in make_batches(inputs, args, task, max_positions, encode_fn):




            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            if use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()

            sample = {
                'net_input': {
                    'src_tokens': src_tokens,
                    'src_lengths': src_lengths,
                },
            }

            translations = task.inference_step(generator, models, sample)
            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
                results.append((start_id + id, src_tokens_i, hypos, id))

        # sort output to match input order

        for id, src_tokens, hypos, position_in_batch in sorted(results, key=lambda x: x[0]):


            if src_dict is not None:
                src_str = src_dict.string(src_tokens, args.remove_bpe)

            # Process top predictions
            for hypo in hypos[:min(len(hypos), args.nbest)]:
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'],
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=args.remove_bpe,
                    tokenized_line=tokenized[position_in_batch]
                )
                detok_hypo_str = decode_fn(hypo_str)
                score = hypo['score'] / math.log(2)  # convert to base 2
                print(detok_hypo_str)


        # update running id counter
        start_id += len(inputs)


def cli_main():
    parser = options.get_interactive_generation_parser()
    parser.add_argument('--pagesplit',action='store_true')
    args = options.parse_args_and_arch(parser)
    distributed_utils.call_main(args, main)


if __name__ == '__main__':

    cli_main()

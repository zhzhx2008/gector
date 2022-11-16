import argparse

from utils.helpers import read_lines
from gector.gec_model import GecBERTModel

import time
import json

from flask import Flask, jsonify, request

app = Flask(__name__)
app.config.from_object(__name__)
app.config['JSON_AS_ASCII'] = False

@app.route('/')
def index():
    return "Hello"

@app.route('/gector', methods = ['GET'])
def hello():
    s = request.args.get('s')
    print('s={}'.format(s))
    predictions = []
    cnt_corrections = 0
    batch = []
    batch.append(s.split())
    preds, cnt = model.handle_batch(batch)
    predictions.extend(preds)
    cnt_corrections += cnt
    res = [" ".join(x) for x in predictions]
    data = {}
    data['s'] = s
    data['res'] = res[0]
    print(data)
    return jsonify(code=1000, message='success', data=data)

if __name__ == "__main__":
    # read parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        help='Path to the model file.', nargs='+',
                        default='/Users/chang/workspace_of_python/pretrained_language_model/huggingface_pretrained_models/xlnetmodel/xlnet_0_gector.th')
    parser.add_argument('--vocab_path',
                        help='Path to the model file.',
                        default='data/output_vocabulary'  # to use pretrained models
                        )
    parser.add_argument('--input_file',
                        help='Path to the evalset file')
    parser.add_argument('--output_file',
                        help='Path to the output file')
    parser.add_argument('--max_len',
                        type=int,
                        help='The max sentence length'
                             '(all longer will be truncated)',
                        default=50)
    parser.add_argument('--min_len',
                        type=int,
                        help='The minimum sentence length'
                             '(all longer will be returned w/o changes)',
                        default=3)
    parser.add_argument('--batch_size',
                        type=int,
                        help='The size of hidden unit cell.',
                        default=128)
    parser.add_argument('--lowercase_tokens',
                        type=int,
                        help='Whether to lowercase tokens.',
                        default=0)
    parser.add_argument('--transformer_model',
                        choices=['bert', 'gpt2', 'transformerxl', 'xlnet', 'distilbert', 'roberta', 'albert'],
                        help='Name of the transformer model.',
                        default='xlnet')
    parser.add_argument('--iteration_count',
                        type=int,
                        help='The number of iterations of the model.',
                        default=5)
    parser.add_argument('--additional_confidence',
                        type=float,
                        help='How many probability to add to $KEEP token.',
                        default=0)
    parser.add_argument('--min_probability',
                        type=float,
                        default=0.0)
    parser.add_argument('--min_error_probability',
                        type=float,
                        default=0.0)
    parser.add_argument('--special_tokens_fix',
                        type=int,
                        help='Whether to fix problem with [CLS], [SEP] tokens tokenization. '
                             'For reproducing reported results it should be 0 for BERT/XLNet and 1 for RoBERTa.',
                        default=0)
    parser.add_argument('--is_ensemble',
                        type=int,
                        help='Whether to do ensembling.',
                        default=0)
    parser.add_argument('--weights',
                        help='Used to calculate weighted average', nargs='+',
                        default=None)
    args = parser.parse_args()
    model = GecBERTModel(vocab_path=args.vocab_path,
                         model_paths=[args.model_path],
                         max_len=args.max_len, min_len=args.min_len,
                         iterations=args.iteration_count,
                         min_error_probability=args.min_error_probability,
                         min_probability=args.min_error_probability,
                         lowercase_tokens=args.lowercase_tokens,
                         model_name=args.transformer_model,
                         special_tokens_fix=args.special_tokens_fix,
                         log=False,
                         confidence=args.additional_confidence,
                         is_ensemble=args.is_ensemble,
                         weigths=args.weights)
    app.run(host='0.0.0.0', port=80, debug=True)
import os
DEVICE = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE
import argparse
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import numpy as np
import spacy
import matplotlib.pyplot as plt
from metrics import compute_exact_match
from utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)

def get_system_prompt(setting_type):
    if setting_type is not None and "cred" in setting_type:
        return "You are an assistant who can answer questions based on the given passages. Each passage has a credibility score that indicates the relevance and accuracy of the passage to the question. Your answer need to combine multiple passages and their credibility."
    else:
        return "You're a helpful AI assistant. The assistant answers questions based on given passages.\n"

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--save_suffix", type=str,default=None)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--setting_type", type=str, default=None,help='concat or rerank')
    # dataset
    parser.add_argument("--wikimulti", action='store_true')
    parser.add_argument("--hotpot", action='store_true')
    parser.add_argument("--musique", action='store_true')
    parser.add_argument("--wikiqa", action='store_true')
    parser.add_argument("--rgb", action='store_true')
    parser.add_argument("--evotemp", action='store_true')
    parser.add_argument("--misinfo", action='store_true')

    parser.add_argument("--qstart", action='store_true') # fastchat专用
    parser.add_argument("--parallel_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--zero_shot", action="store_true")

    # processed
    parser.add_argument("--processed", action="store_true")
    parser.add_argument("--result_suffix", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--inference_mode", type=str, default="vanilla", help="vanilla, dr, drdp, query, fg")

    args = parser.parse_args()
    if args.save_suffix is None:
        args.save_suffix = args.setting_type

    return args


def span_attention_one_token(output_att, n_layers,
        items,
        item_spans,
        context_span,
        marker_impstart,
        marker_impend,
        layer_span,
        threshold, use_norm=True, return_tokens=False):
    '''
    Obtain attention scores for grouping
    Parameters:
    ----------
    output_att : list of torch.Tensor
    n_layers: int
    items: list of str, different groups for context, like document, sentence
    item_spans : list of tuple, token span for each item
    context_span: tuple, token span for the entire context.
    maker_impstart: str, Marker indicating the start of important evidence.
    marker_impend : str, Marker indicating the end of important evidence.
    layer_span : tuple of int, Range of layers to consider for evidence selection.
    threshold : float, Threshold for selecting evidence sentences.

    '''
    # Compute attention scores for the specified range of layers, mean
    # attention: (batch_size, num_heads, generated_length, sequence_length)
    assert len(output_att) == n_layers, "Compute attention scores for the specified range of layers of one generated token."
    att_layer_scores = np.array(
            [
                output_att[l][0, :, -1, context_span[0] : context_span[1]]
                .detach()
                .cpu()
                .float()
                .numpy()
                .mean(axis=0)
                for l in range(layer_span[0], layer_span[1])
            ]
        )
    # Normalize the attention scores across layers.
    if use_norm:
        att_layer_scores /= att_layer_scores.sum(axis=1, keepdims=True)
    # Aggregate token-level scores into group-level scores.
    att_token_scores = att_layer_scores.mean(axis=0)
    if return_tokens:
        return att_token_scores
    group_scores = np.array(
        [
            att_token_scores[item_span[0]: item_span[1]].mean()
            for item_span in item_spans
        ]
    )
    #for i, s in zip(items, group_scores):
    #    print(f"context:\n<a>{i}<b>\nscore:\n{s}")
    # Select group with scores exceeding the threshold. relative
    target_group_index = (group_scores >= group_scores.max() * threshold).nonzero()[0]
    sorted_index = np.argsort(-group_scores)
    #print(sorted_index)
    #print(target_group_index)
    return group_scores


def inference_original(temperature, max_new_tokens, eval_data, shots, tokenizer, model, model_type, f, system, processed=False, new_prompt=False):
    i = 0
    for idx, item in enumerate(tqdm(eval_data)):
        if idx < 5:
            print(idx)
            verbose = True
        else:
            verbose = False
        demo = shots
        if new_prompt:
            prompt = system + demo + "\n\n" + item['new_prompt']
        else:
            prompt = system + demo + "\n\n" + item["conversations"][0]["value"]
        golden = item["conversations"][1]["value"]
        input_ids = tokenizer([prompt], return_tensors="pt").input_ids
        input_ids = input_ids.to(device)

        output_ids = model.generate(input_ids, do_sample=True, temperature=temperature,
                                        max_new_tokens=max_new_tokens)

        output_ids = output_ids[0][len(input_ids[0]):]
        output = tokenizer.decode(output_ids)
        for special_token in tokenizer.special_tokens_map.values():
            if isinstance(special_token, list):
                for special_tok in special_token:
                    output = output.replace(special_tok, "")
            else:
                output = output.replace(special_token, "")
        output = output.strip()
        output = output.split('\n\n')[0]
        if not processed:
            f.write(json.dumps({"output": output, "golden": golden}, ensure_ascii=False) + "\n")
        else:
            item['output'] = output
            item['golden'] = golden
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

        if verbose:
            print(prompt)
            print(f"output: {output}")
            print(f"golden: {golden}")
            print('\n')
            #exit(1)

def reconstruct_input(item, new_doc_orders, qstart):
    question = item['question']
    docs = item['docs']
    docs = [docs[i] for i in new_doc_orders]
    item['new_docs'] = docs
    item['new_order'] = new_doc_orders.tolist()
    doc_prompt = ""
    for doc in docs:
        if doc['title'] != "":
            title = f"{doc['title']}:"
        else:
            title = ""
        doc_prompt += f"{title}{doc['text']}\n"
    if qstart:
        new_prompt = f"Question:{question}\nDocs:{doc_prompt}\nAnswer:"
    else:
        new_prompt =  f"Docs:{doc_prompt}\nQuestion:{question}\nAnswer:"
    item["new_prompt"] = new_prompt
    return new_prompt

def external_factor(docs):
    factors = []
    maps = {
        "high":1.0, "middle":0.95, "low":0.9
    }
    for i, doc in enumerate(docs):
        #print(doc)
        factors.append(maps[doc['cred']])
    return np.array(factors)

def attention_scores_query(output_att, context_span, doc_spans, target, layer_span, n_layers, use_norm=True, return_tokens=False):
    #attention: n_layers (batch_size, num_heads, generated_length, sequence_length)
    assert len(output_att) == n_layers, "Compute attention scores for the specified range of layers of one generated token."
    if type(target) == int:
        att_layer_scores = np.array(
            [
                output_att[l][0, :, target, context_span[0]: context_span[1]]
                    .detach()
                    .cpu()
                    .float()
                    .numpy()
                    .mean(axis=0)
                for l in range(layer_span[0], layer_span[1])
            ]
        )
    else:
        att_target_layer_scores = np.array(
            [
                output_att[l][0, :, target[0]: target[1], context_span[0]: context_span[1]]
                    .detach()
                    .cpu()
                    .float()
                    .numpy()
                    .mean(axis=0)
                for l in range(layer_span[0], layer_span[1])
            ]
        )
        att_layer_scores = att_target_layer_scores.mean(axis=1) #排除query token数的影响
    # Normalize the attention scores across layers.
    if use_norm:
        att_layer_scores /= att_layer_scores.sum(axis=1, keepdims=True)
    # Aggregate token-level scores into group-level scores.
    att_token_scores = att_layer_scores.mean(axis=0)
    if return_tokens:
        return att_token_scores
    group_scores = np.array(
            [
                att_token_scores[item_span[0]: item_span[1]].mean()
                for item_span in doc_spans
            ]
    )
    return group_scores

def attention_scores_output(len_docs, output_ids, attention, tokenizer, n_layers, docs, docs_spans, sents, sent_spans, context_spans, layer_span, method, use_norm=True, return_tokens=False):
    if not return_tokens:
        all_scores = np.zeros(len_docs, dtype=np.float32)
        for i in range(len(output_ids)):
            output_idx = i
            attention_one_token = attention[output_idx]
            output_token = tokenizer.convert_ids_to_tokens(output_ids)[output_idx]
            # print(output_token)
            if method == 0:
            # method0: direct average
                doc_group_scores = span_attention_one_token(attention_one_token, n_layers, docs, docs_spans, context_spans, "<imstart>","<imend>",layer_span,0.5,use_norm)
            # method1: sentence first, then docs
            elif method == 1:
                sentence_group_scores = span_attention_one_token(attention_one_token, n_layers, sents, sent_spans,
                                                                 context_spans, "<imstart>", "<imend>", layer_span, 0.5,use_norm)
                # print(len(sentence_group_scores))
                # print(sentence_group_scores)
                doc_sent = get_doc_sentence_span(sent_spans, docs_spans)
                # print(doc_sent)
                doc_group_scores = np.array(
                    [
                        sentence_group_scores[id[0]: id[-1] + 1].mean()
                        for id in doc_sent
                    ]
                )

            # method2: sentence first, then docs, sentence mean, doc sum
            elif method == 2:
                sentence_group_scores = span_attention_one_token(attention_one_token, n_layers, sents, sent_spans, context_spans, "<imstart>","<imend>",layer_span,0.5, use_norm)
                #print(len(sentence_group_scores))
                #print(sentence_group_scores)
                doc_sent = get_doc_sentence_span(sent_spans, docs_spans)
                #print(doc_sent)
                doc_group_scores = np.array(
                    [
                        sentence_group_scores[id[0]: id[-1] + 1].sum()
                        for id in doc_sent
                    ]
                )
            else:
                raise NotImplementedError
            # print(len(doc_group_scores))
            # print(doc_group_scores)
            all_scores += doc_group_scores
        if len(output_ids) > 0:
            all_scores /= len(output_ids)
        return all_scores
    else:
        all_token_scores = np.zeros(context_spans[1] - context_spans[0], dtype=np.float32)
        for i in range(len(output_ids)):
            output_idx = i
            attention_one_token = attention[output_idx]
            output_token = tokenizer.convert_ids_to_tokens(output_ids)[output_idx]
            # print(output_token)
            if method == 0:
                # method0: direct average
                token_scores = span_attention_one_token(attention_one_token, n_layers, docs, docs_spans,
                                                            context_spans, "<imstart>", "<imend>", layer_span, 0.5,
                                                            use_norm, return_tokens = True)
            # method1: sentence first, then docs
            else:
                raise NotImplementedError
            # print(len(doc_group_scores))
            # print(doc_group_scores)
            all_token_scores += token_scores
        if len(output_ids) > 0:
            all_token_scores /= len(output_ids)
        return all_token_scores

def inference_attention_merge(temperature, max_new_tokens, eval_data, shots, tokenizer, model, model_type, f, system, processed=False, qstart=False, mode="dr"):
    layer0, layer1 = 0.5, 1
    print(f"in inference_attention_merge: {mode}")
    blank_id = tokenizer.pad_token_id
    print(f" blank token: id is {blank_id}")
    passes = 0
    for idx, item in enumerate(tqdm(eval_data)):
        if idx < 5:
            print(idx)
            verbose = True
        else:
            verbose = False
        demo = shots
        #prompt = system + demo + "\n\n" + item["conversations"][0]["value"]
        if qstart:
            prompt = system + demo + "\n\n" + f"Question:{item['question']}\nDocs:{item['doc_prompt']}\nAnswer:"
        else:
            prompt = system + demo + "\n\n" + f"Docs:{item['doc_prompt']}\nQuestion:{item['question']}\nAnswer:"
        golden = item["conversations"][1]["value"]
        input_ids = tokenizer([prompt], return_tensors="pt").input_ids
        input_ids = input_ids.to(device)
        context = item['doc_prompt']
        question = item['question']
        context_spans, context_ids = get_context_ids(input_ids,context, tokenizer)
        question_spans, question_ids = get_context_ids(input_ids, question, tokenizer)
        if verbose:
            print(question_ids)
            print(question_spans)
            print(tokenizer.decode(question_ids[0]))
        sent_spans, sents = get_sentence_token_spans(context_ids, tokenizer)
        #test_spans(sent_spans, sents, context_ids, tokenizer)
        docs_spans, docs = get_document_token_spans(context_ids, tokenizer)
        #test_spans(docs_spans, docs, context_ids, tokenizer)
        if len(docs) != len(item["docs"]):
            print(idx)
            print(f"len(docs, {len(docs)}) != len(item[docs], {len(item['docs'])})")
            print(f"prompt:\n{prompt}")
            print(f"docs:\n{docs}")
            print(f"doc_span:\n{docs_spans}")
            print(f"item[docs]:\n{item['docs']}")
            #test_spans(docs_spans, docs, context_ids, tokenizer)
            exit
        try:
            all_output = model.generate(input_ids, do_sample=True, temperature=temperature,
                                        max_new_tokens=max_new_tokens,
                                        return_dict_in_generate=True, output_attentions=True,)
        except Exception as e:
            print(f"Exception1: {e} in {idx}")
            #exit(1)
            passes += 1
            continue
        #print(all_output)
        attention = all_output.attentions
        #hidden_states = all_output.hidden_states
        output_ids = all_output.sequences
        if verbose:
            print(f"input_id shape: {input_ids.shape}")
            print(f"attention: Tuple (one element for each generated token, {len(attention)}) of tuples (one element for each layer of the decoder, {len(attention[0])}) of torch.FloatTensor of shape (batch_size, num_heads, generated_length, sequence_length).{attention[0][0].shape}")

            #print(f"hidden_states shape: Tuple (one element for each generated token, {len(hidden_states)}) of tuples (one element for each layer of the decoder, {len(hidden_states[0])}) of torch.FloatTensor of shape (batch_size, generated_length, hidden_size){hidden_states[0][0].shape}")
            print(f"output_id shape: {output_ids.shape}")
        output_ids = output_ids[0][len(input_ids[0]):]
        if verbose:
            print(output_ids)
        output, end = get_output(output_ids, tokenizer)
        if verbose:
            print(prompt)
            print(output)
            print(f"end is {end}, {output_ids[end]}, <a>{tokenizer.decode(output_ids[end])}</a>")

        n_layers = len(attention[0])
        layer_span = (
            int(layer0 * n_layers),int(layer1 * n_layers)
        )
        first_layer_span = (0, int(layer0 * n_layers))
        all_layer_span = (0, n_layers)
        last_layer_span = layer_span
        # attention可视化，
        #print(sents[0])
        #ss, se = context_spans[0] + sent_spans[0][0], context_spans[0] + sent_spans[0][1]
        #draw_specific_attention(output_ids,attention, ss, se, all_output, tokenizer)
        # attention score group -- output

        method = 0
        if "dr" in mode:
            all_scores_a = attention_scores_output(len(docs), output_ids[:end], attention, tokenizer, n_layers, docs,
                                                      docs_spans, sents, sent_spans, context_spans, last_layer_span, method,
                                                      use_norm=True)
            all_scores = all_scores_a

        # attention score group -- query

        elif "query" in mode:
            all_scores_q = attention_scores_query(attention[0], context_spans, docs_spans, question_spans,
                                                  all_layer_span, n_layers,use_norm=False)
            all_scores = all_scores_q

        elif "fg" in mode:
            all_scores_q_1 = attention_scores_query(attention[0], context_spans, docs_spans, -1, layer_span, n_layers,
                                                use_norm=True)
            all_scores = all_scores_q_1

        else:
            raise NotImplementedError

        # combined_index

        if qstart:
            combined_index = np.argsort(-all_scores)
        else:
            combined_index = np.argsort(all_scores)

        #combined_index = np.arange(len(docs))
        #combined_index = combined_index[::-1]
        if verbose:
            print(f"attention: {combined_index}")

        if "dp" in mode:
            item['att_order'] = combined_index.tolist()
            token_scores_end = attention_scores_query(attention[end], context_spans, docs_spans, -1,
                                                      first_layer_span, n_layers, use_norm=False, return_tokens=True)
            token_scores_b = attention_scores_query(attention[0], context_spans, docs_spans, -2,
                                                    first_layer_span, n_layers, use_norm=False, return_tokens=True)
            token_scores_p = (token_scores_end + token_scores_b) / 2
            #position_index = np.zeros(len(docs_spans), dtype=np.float32)
            # att_order, 最相关的在最前
            if qstart:
                att_order = combined_index
            else:
                att_order = combined_index[::-1]
            #att_order = att_order[::-1]
            assert context_spans[1] - context_spans[0] == len(token_scores_p), f"{context_spans[1] - context_spans[0]} vs {len(token_scores_p)}"
            left, right = 0, context_spans[1] - context_spans[0]
            lefts, rights = [], []
            for att_idx in att_order:
                doc_len = docs_spans[att_idx][1] - docs_spans[att_idx][0]
                scores_left = token_scores_p[left: left + doc_len].mean()
                scores_right = token_scores_p[right - doc_len: right].mean()
                if scores_left > scores_right:
                    lefts.append(att_idx)
                    left = left + doc_len
                else:
                    rights.append(att_idx)
                    right = right - doc_len
            assert len(lefts) + len(rights) == len(docs_spans), f"{len(lefts)},{len(rights)},{len(docs_spans)}"
            position_index = np.array(lefts + rights[::-1])
            item['position_order'] = position_index.tolist()
            #new_order = np.zeros_like(position_index)
            #new_order[position_index] = combined_index
            combined_index = position_index
            if verbose:
                print(f"position: {position_index}")
                #print(f"attention+position: {combined_index}")

        del all_output, attention
        new_prompt = reconstruct_input(item, combined_index, qstart)
        if verbose:
            print(new_prompt)
        prompt = system + demo + "\n\n" + new_prompt
        input_ids = tokenizer([prompt], return_tensors="pt").input_ids
        input_ids = input_ids.to(device)
        try:
            output_ids = model.generate(input_ids, do_sample=True, temperature=temperature,
                                        max_new_tokens=max_new_tokens)
        except Exception as e:
            print(f"Exception2: {e} in {idx}")
            #exit(2)
            continue
        output_ids = output_ids[0][len(input_ids[0]):]
        output, end = get_output(output_ids, tokenizer)

        if not processed:
            f.write(json.dumps({"output": output, "golden": golden}, ensure_ascii=False) + "\n")
        else:
            item['output'] = output
            item['golden'] = golden
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

        if verbose:
            #print(prompt)
            print(f"output: {output}")
            print(f"golden: {golden}")
            print('\n')
            #exit(1)

def main():
    set_seed(42)
    args = parser()
    assert args.setting_type is not None, "Setting type is required in single scenario!"
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        padding_side="left"
    )
    if "wen" in args.model_path or "3" in args.model_path:
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto",
        output_attentions=True,
    )
    data_list = []
    if args.hotpot:
        data_list.append("HotpotQA")
    if args.musique:
        data_list.append("Musique")
    if args.wikimulti:
        data_list.append("2wikiMultiHopQA")
    system_prompt = get_system_prompt(args.setting_type)
    print(data_list)
    for data_name in data_list:
        if args.qstart:
            eval_data = load_data(args.data_path, data_name, f"{args.setting_type}_qstart")
            if args.zero_shot:
                output_path = os.path.join("./test_zs", data_name,
                                           f"{args.model_type}_{args.save_suffix}_tmp{args.temperature}_qstart_{args.inference_mode}.json")
                shots = ""
            else:
                output_path = os.path.join("./test", data_name,
                                           f"{args.model_type}_{args.save_suffix}_tmp{args.temperature}_qstart_{args.inference_mode}.json")
                with open(f'./prompt/{data_name}.txt', 'r') as f_shot:
                    shots = f_shot.read()
        else:
            eval_data = load_data(args.data_path, data_name, f"{args.setting_type}")
            if args.zero_shot:
                output_path = os.path.join("./test_zs", data_name,
                                           f"{args.model_type}_{args.save_suffix}_tmp{args.temperature}_{args.inference_mode}.json")
                shots = ""
            else:
                output_path = os.path.join("./test", data_name,
                                           f"{args.model_type}_{args.save_suffix}_tmp{args.temperature}_{args.inference_mode}.json")
                with open(f'./prompt/{data_name}.txt', 'r') as f_shot:
                    shots = f_shot.read()
        if args.debug:
            output_path = "delete/test.json"
        with open(output_path, "w") as f:
            if args.inference_mode == "vanilla":
                inference_original(args.temperature, args.max_new_tokens, eval_data, shots, tokenizer, model,
                                   args.model_type, f, system_prompt, args.processed)
            else:
                inference_attention_merge(args.temperature, args.max_new_tokens, eval_data, shots, tokenizer, model,
                                          args.model_type, f, system_prompt, args.processed, args.qstart,
                                          args.inference_mode)

        compute_exact_match(output_path, data_name)


if __name__ == '__main__':
    main()

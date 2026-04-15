import json
import os
import numpy as np
import spacy

def load_data(eval_data_path, dataset, suffix):
    print(dataset)
    print(suffix)
    dataset_path = os.path.join(eval_data_path, dataset)
    for file in os.listdir(dataset_path):
        if suffix + ".json" in file:
            print(f"Load data file: {file}")
            with open(os.path.join(dataset_path, file), "r") as f:
                eval_data = json.load(f)
            break
    return eval_data

def get_n_match(string, substring):
    #  Count the number of occurrences of a substring within a string.
    all_starts = []
    start = 0
    while True:
        start = string.find(substring, start)
        if start == -1:
            break
        all_starts.append(start)
        start += 1  # Increment start to avoid overlapping matches
    return len(all_starts)

def find_text_token_span(tensor_input_ids, target_text, tokenizer):
    """
           Locate spans of a target text within tokenized input.

           Parameters
           ----------
           input_ids : list of int
               Tokenized input as a 1D list of token IDs.

           target_text : str
               Target text to find within the tokenized input.

           Returns
           -------
           spans : list of tuple
               List of (start, end) indices for each occurrence of the target text.
           """
    # Ensure input_ids is a list of integers
    #print(f"input_ids shape: {tensor_input_ids.shape}")
    input_ids = tensor_input_ids.tolist()[0]
    #print(type(input_ids))
    #print(type(input_ids[0]))
    assert (type(input_ids) == list) and (
            type(input_ids[0]) == int
    ), "input_ids should be a 1-d list, make sure it's not a tensor."

    # Decode input tokens to text and encode the target text into tokens
    source = tokenizer.decode(input_ids)
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)
    target = tokenizer.decode(target_ids)
    # Initialize variables for finding spans
    n_match_left = get_n_match(source, target)
    spans = []
    start = 0

    while True:
        start += 1
        source_seg = tokenizer.decode(input_ids[start:])
        n_match_cur = get_n_match(source_seg, target)

        # If the number of matches decreases, start of a match is found
        if n_match_cur < n_match_left:
            assert (
                    n_match_left - n_match_cur == 1
            ), f"{n_match_left - n_match_cur} matches in a same token"
            n_match_left = n_match_cur
            start -= 1
            # Find the end of the match
            end = max(start + len(target_ids) - 5, start)
            while True:
                end += 1
                seg_text = tokenizer.decode(input_ids[start:end])
                if target in seg_text:
                    break
            # Save the span and update the start position
            spans.append((start, end))
            start = end

        # Exit condition
        if n_match_left == 0 or start >= len(input_ids):
            break

    return spans

def get_context_ids(input_ids, context, tokenizer):
    # input_ids: corresponding to prompt
    #print(f"before: {type(input_ids)}")
    spans = find_text_token_span(input_ids, context,tokenizer)
    assert (
            len(spans) == 1
    ), f"Multiple/no context spans found: {spans}"
    context_span = spans[0]
    #print(f"After: {type(input_ids)}")
    context_ids = input_ids[:, context_span[0]: context_span[1]]
    #test_spans(spans, [context], input_ids, tokenizer)
    #print(context_ids.shape)
    return context_span, context_ids

def get_sentence_token_spans(context_ids, tokenizer):
    context_text = tokenizer.decode(context_ids[0])
    context_tokens_text = [
        tokenizer.decode([token_id]).replace(" ", "") for token_id in context_ids[0]
    ]
    sents = [sent.text for sent in spacy.load("en_core_web_sm")(context_text).sents]
    no_enter_sents = []
    for sent in sents:
        s = sent.split('\n')
        for i, ss in enumerate(s):
            if i == len(s) - 1:
                no_enter_sents.append(ss)
            else:
                no_enter_sents.append(ss + '\n')
    sents = no_enter_sents
    #print(f"<a>{context_text}</a>")
    #print(len(context_tokens_text))
    #print(sents)
    for i in range(len(sents)):
        # if sents[i].strip() == "":
        if len(sents[i].strip()) <= 5:
            # if a sent is all " ", then merge it with the previous sent
            if i > 0:
                sents[i - 1] = sents[i - 1] + sents[i]
                sents[i] = ""
            else:
                sents[i + 1] = sents[i] + sents[i + 1]
                sents[i] = ""
            '''
            # if a sent is all " ", then merge it with the next sent
            if i < len(sents) - 1:
                sents[i + 1] = sents[i] + sents[i + 1]
                sents[i] = ""
            else:
                sents[i - 1] = sents[i - 1] + sents[i]
                sents[i] = ""
            '''
    sents = [sent for sent in sents if sent != ""]

    # find sentence token spans
    sent_token_spans = []
    tk_start_idx = 0
    for i, sent in enumerate(sents):
        sent = sent.lstrip(" ")
        sent_num_tokens = len(tokenizer.encode(sent, add_special_tokens=False))
        # find the end token index
        sent_text = sent.replace(" ", "")
        span_text = tokenizer.decode(
            context_ids[0, tk_start_idx: tk_start_idx + sent_num_tokens]
        ).replace(" ", "")
        span_include_sent = span_text.find(sent_text) >= 0
        sent_include_span = sent_text.find(span_text) >= 0
        #print(i)
        #print(f"span_text:{span_text}")
        #print(f"sent_text:{sent_text}")
        len_span = sent_num_tokens
        if span_include_sent and sent_include_span:  # pass
            pass
        elif span_include_sent and not sent_include_span:  # span is longer
            while True:
                len_span -= 1
                if tk_start_idx + len_span < len(context_tokens_text):
                    del_token = context_tokens_text[tk_start_idx + len_span]
                else:
                    del_token = context_tokens_text[-1]
                span_text = span_text.rstrip(del_token)
                if span_text.find(sent_text) < 0:  # span is shorter than sent
                    # len_span += 1
                    span_text = span_text + del_token
                    break
        elif not span_include_sent:  # span is shorter
            while True:
                add_token = context_tokens_text[tk_start_idx + len_span]
                len_span += 1
                span_text = span_text + add_token
                if span_text.find(sent_text) >= 0:
                    break

        tk_end_idx = tk_start_idx + len_span
        sent_token_spans.append((tk_start_idx, tk_end_idx))
        tk_start_idx = tk_end_idx

        if not span_text.endswith(sent_text):  # last token contains the next sentence
            tk_start_idx -= 1

    assert len(sent_token_spans) == len(sents)

    return sent_token_spans, sents

def get_document_token_spans(context_ids, tokenizer):
    context_text = tokenizer.decode(context_ids[0])
    #print(context_text)
    #print(context_ids)
    context_tokens_text = [
        tokenizer.decode([token_id]).replace(" ","") for token_id in context_ids[0]
    ]
    #print(len(context_tokens_text))
    docs = context_text.strip().split('\n')
    #print(len(docs))
    # find document token spans
    docs_token_spans = []
    tk_start_idx = 0
    for i, doc in enumerate(docs):
        doc = doc.strip()
        doc_num_tokens = len(tokenizer.encode(doc, add_special_tokens=False))
        # find the end token index
        doc_text = doc.replace(" ","")
        #print(doc_text)
        span_text = tokenizer.decode(
            context_ids[0, tk_start_idx : tk_start_idx + doc_num_tokens]
        ).replace(" ","")
        #print(span_text)
        span_include_doc = span_text.find(doc_text) >= 0
        doc_include_span = doc_text.find(span_text) >= 0
        #print(f"span_include_doc: {span_include_doc}")
        #print(f"doc_include_span: {doc_include_span}")
        len_span = doc_num_tokens
        if span_include_doc and doc_include_span:
            pass
        elif span_include_doc and not doc_include_span: # span is longer
            while True:
                len_span -= 1
                del_token = context_tokens_text[tk_start_idx + len_span]
                span_text = span_text.rstrip(del_token)
                if span_text.find(doc_text) < 0: # span is shorter than doc
                    span_text = span_text + del_token
                    break
            #print("Deleting: ")
            #print(doc_text)
            #print(span_text)
        elif not span_include_doc: # span is shorter
            while True:
                add_token = context_tokens_text[tk_start_idx + len_span]
                len_span += 1
                span_text = span_text + add_token
                #print(span_text)
                if span_text.find(doc_text) >= 0:
                    break
            #print("Adding: ")
            #print(doc_text)
            #print(span_text)
        tk_end_idx = tk_start_idx + len_span
        docs_token_spans.append((tk_start_idx, tk_end_idx))
        tk_start_idx = tk_end_idx + 1
        if not span_text.endswith(doc_text):  # last token contains the next sentence
            tk_start_idx -= 1
    assert len(docs_token_spans) == len(docs), f"{len(docs_token_spans)} vs {len(docs)}"
    return docs_token_spans, docs

def get_doc_sentence_span(sent_spans, doc_spans):
    result_span = []
    i, j = 0, 0
    span = []
    #print(len(sent_spans))
    #print(len(doc_spans))
    while i < len(sent_spans) and j < len(doc_spans):
        #print(f"{i},{sent_spans[i]}")
        #print(f"{j},{doc_spans[j]}")
        ending = doc_spans[j][1]
        if sent_spans[i][0] + 1 < ending:
            span.append(i)
            i += 1
        else:
            result_span.append(span)
            span = []
            j += 1
    if len(span) > 0:
        result_span.append(span)
    assert len(result_span) == len(doc_spans), f"{len(result_span)} vs {len(doc_spans)}"
    for idx, span in enumerate(result_span):
        assert len(span) > 0, f"{idx} is empty, {result_span}\n{sent_spans}\n{doc_spans}"
    '''
    print(result_span)
    for i in range(len(doc_spans)):
        print(i)
        print(doc_spans[i])
        for j in result_span[i]:
            print(sent_spans[j])
    exit(2)
    '''
    return result_span

def test_spans(spans, items, input_ids,tokenizer):
    print(f"input_ids shape: {input_ids.shape}")
    source = tokenizer.decode(input_ids[0])
    print(source)
    assert len(spans) == len(items)
    print(len(spans))
    idx = 0
    for span, item in zip(spans,items):
        print(f"idx {idx}")
        print(item)
        split_ids = input_ids[:, span[0]: span[1]]
        print(f"split_ids shape: {split_ids.shape}")
        print(tokenizer.decode(split_ids[0]))
        idx += 1
    exit(0)


def get_output(output_ids, tokenizer):
    output = tokenizer.decode(output_ids)
    for special_token in tokenizer.special_tokens_map.values():
        if isinstance(special_token, list):
            for special_tok in special_token:
                output = output.replace(special_tok, "")
        else:
            output = output.replace(special_token, "")
    output = output.strip()
    output = output.split('\n\n')[0]
    for i, id in enumerate(output_ids):
        s = tokenizer.decode(id)
        if "\n" in s:
            break
        if s == tokenizer.eos_token:
            break
    return output, i

def find_target_in_list(l, target):
    for i, item in enumerate(l):
        if item == target:
            return i
    return len(l)




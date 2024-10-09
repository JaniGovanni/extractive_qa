import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


def get_model_and_tokenizer(model_checkpoint):
    """
    Loads model and tokenizer from the provided checkpoint
    """
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    return model, tokenizer


def get_possible_answers(preds,
                         model_inputs,
                         context):
    """
    searches for all valid predictions of the model.
    """
    best_score = float('-inf')
    possible_answers = []
    # loop over the splitted sample
    for i, offsets in enumerate(model_inputs['new_offsets']):
        n_largest = 20
        max_answer_length = 30

        start_logit = preds.start_logits[i]
        end_logit = preds.end_logits[i]

        start_indices = (-start_logit).argsort()[:n_largest]
        end_indices = (-end_logit).argsort()[:n_largest]
        for start_idx in start_indices:
            for end_idx in end_indices:

                # skip answers not contained in context window
                if offsets[start_idx] is None or offsets[end_idx] is None:
                    continue

                # skip answers where end < start
                if end_idx < start_idx:
                    continue

                # skip answers that are too long
                if end_idx - start_idx + 1 > max_answer_length:
                    continue

                # skip when there is no answer
                if start_idx == end_idx:
                    continue

                score = start_logit[start_idx] + end_logit[end_idx]
                if score > best_score:
                    best_score = score
                    # find positions of start and end characters
                    first_ch = offsets[start_idx][0]
                    last_ch = offsets[end_idx][1]

                    possible_answers.append(context[first_ch:last_ch])
    return possible_answers


def get_question_answers(question:str, context:str, model, tokenizer):
    """
    Searches for possible answers to the users question in the provided context.
    """

    # constants that were used for training, maybe write these in the config
    max_length = 384
    stride = 128
    
    inputs = tokenizer(
        question,
        context,
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors='pt'
    )
    new_offsets = []
    for i in range(len(inputs["input_ids"])):
        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        #inputs["new_offset_mapping"][i] = [
        #    x if sequence_ids[j] == 1 else None for j, x in enumerate(offset)]
        new_offsets_i = [
                x if sequence_ids[j] == 1 else None for j, x in enumerate(offset)]
        new_offsets.append(new_offsets_i)
    inputs.data["new_offsets"] = new_offsets

    preds = model(inputs['input_ids'], inputs['attention_mask'])
    return get_possible_answers(preds, inputs, context)
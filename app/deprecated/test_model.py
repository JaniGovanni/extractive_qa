from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModel
import torch

def get_best_answer(preds,
                    model_inputs,
                    context):

    # loop over the splitted sample
    for i, offsets in enumerate(model_inputs['offset_mapping']):

        best_score = float('-inf')
        best_answer = None
        n_largest = 20
        max_answer_length = 30

        start_logit = preds.start_logits[i]  # (seq_len,) vector
        end_logit = preds.end_logits[i]  # (seq_len,) vector

        # do NOT do the reverse indexing: ['offset_mapping'][idx], much slower

        # get the highest probability tokens
        start_indices = (-start_logit).argsort()
        end_indices = (-end_logit).argsort()
        for start_idx in start_indices[:n_largest]:
            for end_idx in end_indices[:n_largest]:

                # skip answers not contained in context window
                if offsets[start_idx] is None or offsets[end_idx] is None:
                    continue

                # skip answers where end < start
                if end_idx < start_idx:
                    continue

                # skip answers that are too long
                if end_idx - start_idx + 1 > max_answer_length:
                    continue

                # see cell down under for score calculation
                score = start_logit[start_idx] + end_logit[end_idx]
                if score > best_score:
                    best_score = score

                    # find positions of start and end characters
                    first_ch = offsets[start_idx][0]
                    last_ch = offsets[end_idx][1]

                    best_answer = context[first_ch:last_ch]
        return best_answer

context = """Architecturally, the school has a Catholic character. 
Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. 
Immediately in front of the Main Building and facing it, is a copper statue of Christ 
with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building 
is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian 
place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the 
Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive 
(and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone 
statue of Mary. Super Bowl 50 was an American football game to determine the champion of the 
National Football League (NFL) for the 2015 season. The American Football Conference (AFC) 
champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 
24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at 
Levi\'s Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, 
the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily 
suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have 
been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50."""

question = 'What is in front of the Notre Dame Main Building?'

max_length = 30
stride = 10
model_checkpoint = "/Users/jan/Desktop/pdf_extractive_qa/app/models/trained_model_extr_qa_1"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
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


for i in range(len(inputs["input_ids"])):
    sequence_ids = inputs.sequence_ids(i)
    offset = inputs["offset_mapping"][i]
    inputs["offset_mapping"][i] = [
        x if sequence_ids[j] == 1 else None for j, x in enumerate(offset)]


model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

preds = model(inputs['input_ids'], inputs['attention_mask'])

answer_start_index = preds.start_logits.argmax(axis=1)
answer_end_index = preds.end_logits.argmax(axis=1)


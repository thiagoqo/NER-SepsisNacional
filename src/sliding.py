import json

MAX_INPUT_SIZE = 20_000
MAX_NEW_TOKENS = 6000

def clean_json_text(text):
    text = text.strip()
    text = text.replace("\#", "#").replace("\&", "&")
    return text

def predict_chunk(text, template, current, model, tokenizer):
    current = clean_json_text(current)

    input_llm =  f"<|input|>\n### Template:\n{template}\n### Current:\n{current}\n### Text:\n{text}\n\n<|output|>" + "{"
    input_ids = tokenizer(input_llm, return_tensors="pt", truncation=True, max_length=MAX_INPUT_SIZE).to("cuda")
    output = tokenizer.decode(model.generate(**input_ids, max_new_tokens=MAX_NEW_TOKENS)[0], skip_special_tokens=True)

    return clean_json_text(output.split("<|output|>")[1])

def split_document(document, window_size, overlap):
    tokens = tokenizer.tokenize(document)
    print(f"\tLength of document: {len(tokens)} tokens")

    chunks = []
    if len(tokens) > window_size:
        for i in range(0, len(tokens), window_size-overlap):
            print(f"\t{i} to {i + len(tokens[i:i + window_size])}")
            chunk = tokenizer.convert_tokens_to_string(tokens[i:i + window_size])
            chunks.append(chunk)

            if i + len(tokens[i:i + window_size]) >= len(tokens):
                break
    else:
        chunks.append(document)
    print(f"\tSplit into {len(chunks)} chunks")

    return chunks

def handle_broken_output(pred, prev):
    try:
        if all([(v in ["", []]) for v in json.loads(pred).values()]):
            # if empty json, return previous
            pred = prev
    except:
        # if broken json, return previous
        pred = prev

    return pred

def sliding_window_prediction(text, template, model, tokenizer, window_size=4000, overlap=128):
    # split text into chunks of n tokens
    tokens = tokenizer.tokenize(text)
    chunks = split_document(text, window_size, overlap)

    # iterate over text chunks
    prev = template
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i}...")
        pred = predict_chunk(chunk, template, prev, model, tokenizer)

        # handle broken output
        pred = handle_broken_output(pred, prev)
            
        # iterate
        prev = pred

    return pred

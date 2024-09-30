from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch, AuxFun, spacy

# Load model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Input text
Amostras = AuxFun.Amostras(AuxFun.File("TNasa.txt"))

for Amostra in Amostras:
    inputs = tokenizer(Amostra, return_tensors="pt")

    # Get logits (unnormalized probabilities)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Convert logits to probabilities
    probabilities = torch.softmax(logits, dim=-1)

    # Get token IDs for the input text
    input_ids = inputs["input_ids"][0]

    # Extract the probabilities for the input tokens
    for i, token_id in enumerate(input_ids):
        token = tokenizer.decode([token_id])  # Decode token ID to the corresponding word
        prob = probabilities[0, i, token_id].item()  # Get the probability for the token at position i
        if token.isalpha():
            print(f"Token: {token}, Probability: {prob}")

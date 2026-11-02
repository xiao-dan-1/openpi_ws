"""Length admission using a public tokenizer configured with limit + 1 tokens."""


def checked_token_length(tokenizer, prompt, state, limit):
    _, mask = tokenizer.tokenize(prompt, state)
    count = int(mask.sum())
    if count > limit:
        raise ValueError(f"State prompt exceeds the token limit ({limit})")
    return count

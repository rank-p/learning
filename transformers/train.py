import torch
import torch.nn.functional as F
from model import GPT

SEP = 10  # separator token (digits are 0-9)
SEQ_LEN = 10

def generate_batch(seq_len, batch_size):
    """
    Generates sequences like: [3, 7, 1, 4, SEP, 4, 1, 7, 3]
    For next-token prediction:
        input:  [3, 7, 1, 4, SEP, 4, 1, 7]
        target: [7, 1, 4, SEP, 4, 1, 7, 3]
    The model learns to predict each next token.
    """
    digits = torch.randint(0, 10, (batch_size, seq_len))
    reversed_digits = digits.flip(1)
    sep = torch.full((batch_size, 1), SEP)
    # full sequence: digits + SEP + reversed_digits
    full = torch.cat([digits, sep, reversed_digits], dim=1)  # (batch_size, seq_len*2 + 1)
    # input is everything except last token, target is everything except first
    x = full[:, :-1]
    y = full[:, 1:]
    return x, y

def train():
    NUM_LOOPS = 1000
    BATCH_SIZE = 32
    LR = 3e-4
    d_model = 64
    n_heads = 8
    vocab_size = 11
    n_layers = 2
    max_seq_len = SEQ_LEN * 2 # for this example we use [1,2,3,SEP,3,2] as input
    model = GPT(d_model, n_heads, vocab_size, max_seq_len, n_layers)
    optim = torch.optim.Adam(model.parameters(), LR)

    for n in range(NUM_LOOPS):
        x, y = generate_batch(SEQ_LEN, BATCH_SIZE)
        logits = model(x)
        # For the loss we don't care about the batch dimension
        # The logits are in shape (BATCH_SIZE, seq_len * 2, vocab_size).
        # This is transformed to shape (BATCH_SIZE * seq_len * 2, vocab_size)
        # reshape with -1 automatically calculates the dimension
        logits = logits.reshape(-1, vocab_size)
        # Same for y
        y = y.reshape(-1)
        loss = F.cross_entropy(logits, y)
        if n % 100 == 0:
            print(f"step {n} | loss: {loss.item():.4f}")
        optim.zero_grad()
        loss.backward()
        optim.step()
    return model

def test(model):
    with torch.no_grad():
        digits = torch.randint(0, 10, (1, SEQ_LEN))
        prompt = torch.cat([digits, torch.full((1,1), SEP)], dim=1)
        for _ in range(SEQ_LEN):
            out = model(prompt)
            prediction = out[:, -1, :].argmax(dim=-1, keepdim=True)
            prompt = torch.cat([prompt, prediction], dim=1)
        print(f"input:    {digits[0].tolist()}")
        print(f"expected: {digits[0].flip(0).tolist()}")
        print(f"got:      {prompt[0, SEQ_LEN + 1:].tolist()}")

if __name__ == "__main__":
    model = train()
    test(model)




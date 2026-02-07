import torch
import torch.nn as nn
import torch.nn.functional as F


class GPT(nn.Module):
    def __init__(self, d_model: int, n_heads: int, vocab_size: int, max_seq_len: int, n_layers: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq_len, d_model)
        self.transformers = nn.ModuleList(
                [Transformer(d_model, n_heads) for i in range(n_layers)]
        )
        self.ln = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T) integer tensor of token IDs.
        The model never sees raw text — a tokenizer converts text to integer IDs first.
        The integer IDs are effectively an index into the token vocabulary.
            "the cat sat" -> tokenizer -> [4, 17, 92] -> model
        The embedding layer maps each ID to a learned vector.
        """
        _, T = x.shape
        positions = torch.arange(T, device=x.device)
        x = self.embedding(x) + self.pos(positions)
        for t in self.transformers:
            x = t(x)
        x = self.ln(x)
        x = self.linear(x)
        return x
        


class Transformer(nn.Module):
    
    """
    LayerNorm to keep vectors stable at mean ~0, variance ~1
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model)
    """
    Why the x = x + ...? -> residual connections. It stabilizes training.
    It lets each layer learn "what to add" to the original information rathern than constructing the full feedback.
    This is interesting, I should look more into the research and mathematical properties of these.
    (He et al., 2015 - Deep Residual Learning for Image Recognition)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mha(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class MultiHeadAttention(nn.Module):
    """
    d_model: The dimension of each token (how many elements per vector)
    n_heads: Number of attention heads
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    r"""
    x: (B, T, C) where
        B = batch size (number of sequences processed in parallel)
        T = time/sequence length (number of tokens)
        C = channels/d_model (dimension of each token's vector)

    x (B,T,C)
        |
        +---> Linear_q ---> Q (B,T,C)
        +---> Linear_k ---> K (B,T,C)
        +---> Linear_v ---> V (B,T,C)
        |         |              |
        |     _reshape        _reshape
        |         |              |
        |     (B,nh,T,dh)    (B,nh,T,dh)    nh = n_heads, dh = d_head
        |         \       |       /
        |          \      |      /
        |           attention()
        |               |
        |          (B,nh,T,dh)
        |               |
        |       transpose + view
        |               |
        |          (B,T,C)
        |               |
        |          Linear_out
        |               |
        +----------> (B,T,C)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        
        # Returns tensor of shape (B, n_heads, T, d_head)
        y = self.attention(self._reshape(Q), self._reshape(K), self._reshape(V))
        # We want tensor of shape (B, T, d_model)
        # 1. Transpose to (B, T, n_heads, d_head)
        y = y.transpose(1, 2)
        # 2. Use .view to transform back to (B, T, d_model)
        y = y.contiguous().view(B, T, C)
        return self.out(y)
    
    """
    Say d_model is 16, meaning each token is represented by a vector with 16 elements
    If we use 4 attention heads, each attention head will receive 4 (16/4) of these vector elements from every token.
    More formal: Output after linear projection is (B, T, d_model) (e.g. (1, 2, 16) = 1 sentence, 2 tokens, each 16 elements)
    This is reshaped into (B, n_heads, T, d_head) (e.g. (1, 4, 2, 4) = 4 heads receiving a 2x4 matrix)

    Q: Why do we not pass all 16 (d_model) elements to the heads?
    A: Not necessary as the model can direct relevant features to the different heads via the linear projection layer
    """
    def _reshape(self, x: torch.Tensor):
        B, T, _ = x.shape
        d_head = self.d_model // self.n_heads
        x = x.view(B, T, self.n_heads, d_head)
        x = x.transpose(1, 2)
        return x

    """
    Q, K, V: Tensors of dimension (X, Y, Z).

    dim X: Batches - can be thought of as sentences
    dim Y: Rows/Tokens - amount of tokens in batch
    dim Z: Columns/Vector - amount of vector elements in token 
    """
    def attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        d_k = K.shape[-1]
        d_q = Q.shape[-1]
        assert d_k == d_q
        
        # score_matrix is of dim (X, Y, Y) and represents how much each token attends to each other
        score_matrix = torch.matmul(Q, torch.transpose(K, -2, -1)) / (d_k ** 0.5)
        # tokens should only attend to previous tokens, we need to null the rest
        nullify = torch.triu(torch.full(score_matrix.shape, float('-inf'), device=Q.device), diagonal=1)
        score_matrix += nullify

        return torch.matmul(torch.softmax(score_matrix, -1), V)

class FeedForward(nn.Module):
    
    """
    Intuition: After the attention block, each token's vector is a weighted mix of other tokens' information.
    The model can figure out which tokens to look at, but at this point it's just a linear combination.
    The FeedForward layer applies non-linear processing to each token independently.
    Intuitively, this can be described as giving the model space to think about the results of the attention layer.
    The expansion to d_model * 4 gives it more room to work.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.l = nn.Linear(d_model, d_model * 4)
        self.gelu = nn.GELU()
        self.out = nn.Linear(d_model * 4, d_model)
    def forward(self, x: torch.Tensor):
        x = self.l(x)
        x = self.gelu(x)
        x = self.out(x)
        return x



if __name__ == "__main__":
    model = GPT(d_model=64, n_heads=4, vocab_size=11, max_seq_len=32, n_layers=2)
    x = torch.randint(0, 11, (1, 10))  # 1 batch, 10 random token IDs (0-10)
    out = model(x)
    print("output shape:", out.shape)  # expect (1, 10, 11)


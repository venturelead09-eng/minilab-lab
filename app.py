
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

st.set_page_config(page_title="MiniLM Lab", page_icon="🧠")
st.title("🧠 MiniLM Lab")
st.caption("A tiny language model trained from scratch")

st.info(
    "This experimental model generates text continuations. "
    "It is not yet a general-purpose question-answering assistant."
)

DEVICE = "cpu"

class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer(
            "mask", torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        _, T, _ = x.shape
        k, q = self.key(x), self.query(x)
        w = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        w = w.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        return F.softmax(w, dim=-1) @ self.value(x)

class MultiHead(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(n_embd, n_embd // n_head, block_size)
            for _ in range(n_head)
        ])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.attn = MultiHead(n_embd, n_head, block_size)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))

class MiniLM(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        n_embd = config["n_embd"]
        n_head = config["n_head"]
        n_layer = config["n_layer"]
        block_size = config["block_size"]
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, block_size)
            for _ in range(n_layer)
        ])
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        _, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        return self.head(self.ln(self.blocks(x)))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=0.8):
        self.eval()
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.block_size:])
            logits = logits[:, -1, :] / max(temperature, 0.1)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

@st.cache_resource
def load_model():
    checkpoint = torch.load(
        "minilm.pt", map_location=DEVICE, weights_only=True
    )
    chars = checkpoint["chars"]
    config = checkpoint["config"]
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    model = MiniLM(len(chars), config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, stoi, itos

try:
    model, stoi, itos = load_model()
    st.success("Your trained model is loaded!")
except Exception as e:
    st.error("Could not load minilm.pt. Check that it is uploaded beside app.py.")
    st.code(str(e))
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

for item in st.session_state.history:
    with st.chat_message(item["role"]):
        st.write(item["text"])

prompt = st.chat_input("Type a short text prompt...")
if prompt:
    st.session_state.history.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    clean = "".join(ch for ch in prompt if ch in stoi)
    if not clean:
        answer = "I don't recognize those characters. Try English text."
    else:
        ids = torch.tensor([[stoi[ch] for ch in clean]], dtype=torch.long)
        with st.spinner("Generating text..."):
            result = model.generate(ids)[0].tolist()
        answer = "".join(itos[i] for i in result[len(ids[0]):])
        if not answer.strip():
            answer = "(No visible text generated.)"

    st.session_state.history.append({"role": "assistant", "text": answer})
    with st.chat_message("assistant"):
        st.write(answer)

if st.button("Clear conversation"):
    st.session_state.history = []
    st.rerun()

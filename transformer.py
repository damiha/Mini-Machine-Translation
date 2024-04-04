import torch.nn as nn
import torch
import math
import torch.nn.functional as F

class SelfAttentionBlock(nn.Module):
    
    def __init__(self, d_model, n_heads, device, mask_future=True):
        
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.device = device
        
        assert d_model % n_heads == 0, "n_heads must divide d_model"
        
        # just matrix multiplications
        self.key_net = nn.Sequential(
            nn.Linear(d_model, d_model)
        )
        
        self.query_net = nn.Sequential(
            nn.Linear(d_model, d_model)
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(d_model, d_model)
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(p = 0.1)
        
        self.mask_future = mask_future
    
    def forward(self, x):
        
        B, T, _ = x.shape
        
        # != self.key_net(x).view((B, self.n_heads, T, -1))
        # with the above, future leaks into past
        keys = self.key_net(x).view((B, T, self.n_heads, -1)).transpose(1, 2)
        queries = self.query_net(x).view((B, T, self.n_heads, -1)).transpose(1, 2)
        values = self.value_net(x).view((B, T, self.n_heads, -1)).transpose(1, 2)
        
        scaling_factor = 1.0 / math.sqrt(self.d_model / self.n_heads)
        attention_matrices = scaling_factor * torch.matmul(queries, keys.transpose(2, 3))
        
        # necessary for the decoder but not for encoder
        if self.mask_future:
            neg_inf = -1e10

            # mask the future (upper triangle)
            mask = torch.tril(torch.ones(T, T)).to(self.device)
            mask = mask.masked_fill(mask == 0, -float("inf"))

            # softmax per row
            attention_matrices = F.softmax(attention_matrices + mask, dim=-1)
                
        # (B, head, T, dim_per_head)
        # d_model = head * dim_per_head
        att_output = torch.matmul(attention_matrices, values)
        
        att_output = torch.transpose(att_output, 1, 2)
        
        after_attention_dropout = self.dropout(att_output.reshape((B, T, -1)))
        
        return self.layer_norm(after_attention_dropout + x)
    
class CrossAttentionBlock(nn.Module):
    
    def __init__(self, d_model, n_heads):
        
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        
        assert d_model % n_heads == 0, "n_heads must divide d_model"
        
        # just matrix multiplications
        self.key_net = nn.Sequential(
            nn.Linear(d_model, d_model)
        )
        
        self.query_net = nn.Sequential(
            nn.Linear(d_model, d_model)
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(d_model, d_model)
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(p = 0.1)
    
    def forward(self, encoder_seq, decoder_seq):
        
        # Batch should be the same, but length might be different!
        B, T_enc, _ = encoder_seq.shape
        B, T_dec, _ = decoder_seq.shape
        
        # decoder (target sequence) wants to find the right encoder tokens
        queries = self.query_net(decoder_seq).view((B, T_dec, self.n_heads, -1)).transpose(1, 2)
        
        keys = self.key_net(encoder_seq).view((B, T_enc, self.n_heads, -1)).transpose(1, 2)
        values = self.value_net(encoder_seq).view((B, T_enc, self.n_heads, -1)).transpose(1, 2)
        
        # heads and d_model are shared between decoder and encoder
        
        scaling_factor = 1.0 / math.sqrt(self.d_model / self.n_heads)
        
        # q * K.T = (T_dec, d_model) * (T_enc, d_model).T = (T_dec, d_model) * (d_model, T_enc)
        # = (T_dec, T_enc)
        attention_matrices = scaling_factor * torch.matmul(queries, keys.transpose(2, 3))
        
        # (T_dec, T_enc) * (T_enc, d_model) = T_dec, d_model (gets passed further down the decoder)
        att_output = torch.matmul(attention_matrices, values)
        
        att_output = torch.transpose(att_output, 1, 2)
        
        after_attention_dropout = self.dropout(att_output.reshape((B, T_dec, -1)))
        
        return self.layer_norm(after_attention_dropout + decoder_seq)
    
class FeedForwardBlock(nn.Module):
    
    def __init__(self, d_model):
        
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p = 0.1)
        
        # 4x comes from original paper
        self.ffn = nn.Sequential(
            nn.Linear(d_model,  4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        
    def forward(self, x):
        
        ffn_output = self.ffn(x)

        return self.layer_norm(self.dropout(ffn_output) + x)
    
class EncoderBlock(nn.Module):
    
    def __init__(self, d_model, n_heads):
        
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        
        self.net = nn.Sequential(
            SelfAttentionBlock(d_model, n_heads, device=None, mask_future=False),
            FeedForwardBlock(d_model)
        )
    
    def forward(self, x):
        return self.net(x)
    
class DecoderBlock(nn.Module):
    
    def __init__(self, d_model, n_heads, device):
        
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.device = device
        
        self.self_attention = SelfAttentionBlock(d_model, n_heads, device, mask_future=True)
        self.cross_attention = CrossAttentionBlock(d_model, n_heads)
        self.ffn = FeedForwardBlock(d_model)
    
    def forward(self, encoder_seq, decoder_seq):
        
        decoder_seq = self.self_attention(decoder_seq)
        
        decoder_seq = self.cross_attention(encoder_seq, decoder_seq)
        
        return self.ffn(decoder_seq)

    
class Transformer(nn.Module):
    
    def __init__(self,
                 n_symbols_enc, n_symbols_dec,
                 context_length_enc, context_length_dec,
                 d_model, # share
                 n_heads, # share
                 n_layers_enc, n_layers_dec,
                 device):
        
        super().__init__()
        
        self.n_symbols_enc = n_symbols_enc
        self.n_symbols_dec = n_symbols_dec
        
        self.d_model = d_model
        self.n_heads = n_heads
        
        self.context_length_enc = context_length_enc
        self.context_length_dec = context_length_dec
        
        self.n_layers_enc = n_layers_enc
        self.n_layers_dec = n_layers_dec
        
        self.device=device
        
        self.token_embedding_enc = nn.Embedding(num_embeddings=n_symbols_enc, embedding_dim=d_model)
        self.pos_embedding_enc = nn.Embedding(num_embeddings=context_length_enc, embedding_dim=d_model)
        
        self.token_embedding_dec = nn.Embedding(num_embeddings=n_symbols_dec, embedding_dim=d_model)
        self.pos_embedding_dec = nn.Embedding(num_embeddings=context_length_dec, embedding_dim=d_model)
        
        ebs = [EncoderBlock(d_model = d_model, n_heads = n_heads) for _ in range(n_layers_enc)]
        self.encoder = nn.Sequential(*ebs)
        
        dbs = [DecoderBlock(d_model = d_model, n_heads = n_heads, device=device) for _ in range(n_layers_dec)]
        self.decoder = nn.ModuleList(dbs)
        
        # predict next token in target sequence
        self.to_logits = nn.Sequential(
            nn.Linear(d_model, n_symbols_dec, device=device)
        )
        
        self.embedding_dropout = nn.Dropout(p = 0.1)
        
    def forward(self, encoder_seq, decoder_seq):
        
        # batch, time
        B, T_enc = encoder_seq.shape
        B, T_dec = decoder_seq.shape
        
        embedded_enc = self.token_embedding_enc(encoder_seq)
        embedded_dec = self.token_embedding_dec(decoder_seq)
        #print(f"{embedded.shape}")
        
        positions_enc = torch.arange(T_enc).to(self.device)
        positions_dec = torch.arange(T_dec).to(self.device)
        #print(f"{positions.shape}")

        embedded_enc = self.embedding_dropout(embedded_enc + self.pos_embedding_enc(positions_enc))
        embedded_dec = self.embedding_dropout(embedded_dec + self.pos_embedding_dec(positions_dec))
        
        encoder_output = self.encoder(embedded_enc)
        
        decoder_output = embedded_dec
        
        for i in range(self.n_layers_dec):
            decoder_output = self.decoder[i](encoder_output, decoder_output)
                
        return self.to_logits(decoder_output)
    
    def sample(self, prompt_tokens, n_tokens, n_samples, beta = 1.0):
        self.eval()
        self.to(self.device)

        # Process the prompt to fit within the context length
        prompt_tokens = prompt_tokens[-self.context_length_enc:]
        print(f"Prompt tokens: {prompt_tokens}")

        encoder_input = torch.tensor(prompt_tokens, dtype=torch.long).repeat(n_samples, 1).to(self.device)
        
        dec_start_token = self.n_symbols_dec - 1
        decoder_output = torch.full((n_samples, 1), fill_value = dec_start_token).to(self.device)
        
        history = torch.zeros_like(decoder_output)
        
        for _ in range(n_tokens):
            with torch.no_grad():
                logits = self(encoder_input, decoder_output)[:, -1, :] / beta  # Get logits for the last token position only
                probs = F.softmax(logits, dim=-1)
                last_sampled_token = torch.multinomial(probs, num_samples=1)
                
                history = torch.cat((history, last_sampled_token), dim=1)
                decoder_output =\
                    torch.cat((decoder_output, last_sampled_token), dim=1)[:, -self.context_length_dec:]  # Update context
                
                
        response = history[:, -n_tokens:]
        return response
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
        print(f'Model saved to {path}')
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        print(f'Model loaded from {path}')
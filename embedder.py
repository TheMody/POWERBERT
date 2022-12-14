import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import time
import numpy as np
from transformers.utils import logging
from transformers import glue_convert_examples_to_features, DataCollatorForLanguageModeling
from transformers import BertTokenizer, BertModel
from transformers.models.bert.modeling_bert import BertEmbeddings
from torch.autograd import variable
from torch.utils.data import DataLoader
from data import load_wiki
import wandb
logging.set_verbosity_error()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from hourglass.hourglass_transformer_pytorch.hourglass_transformer_pytorch import HourglassTransformerLM


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
    

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15) 
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o

class EncoderBlock(nn.Module):

    def __init__(self, input_dim, num_heads, dim_feedforward = None, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()
        if dim_feedforward == None:
            dim_feedforward = input_dim*4
        # Attention layer
        self.self_attn =  MultiheadAttention(input_dim, input_dim, num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(dim_feedforward, input_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out,att = self.self_attn(x, mask=mask, return_attention = True)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x, att

class TransformerEncoder(nn.Module):

    def __init__(self, num_layers, d_model, n_head, reduce = False):
        super().__init__()
        self.reduce = reduce
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head) for _ in range(num_layers)])

    def forward(self, x, mask=None):
      #  trackarray = torch.zeroes(x.shape[0,1])
        trackarray = torch.cat( [torch.LongTensor(list(range(x.shape[1])) ).unsqueeze(0) for i in range(x.shape[0])])
    #    mask = torch.matmul(mask.unsqueeze(-1).transpose(1,2).float(),mask.unsqueeze(-1).float()).long().unsqueeze(1)
        mask = mask.transpose(0,1)#.bool()
      #  print(mask)
      #  print(mask.type())
      #  print(trackarray)
        for a,l in enumerate(self.layers):
          #  print("shape of x at layer ",x.shape)
            x,att = l(x, src_key_padding_mask=mask)
          #  print("shape of x at layer after compute",x.shape)

            if self.reduce and a % 3 == 0:
                x, mask, trackarray = self.extract(x,att, mask = mask, trackarray = trackarray)
           # print("shape of x at layer after extract",x.shape)
        return x, trackarray

    def extract(self, x, attention, mask = None, trackarray = None, attention_based = True, similarity_based = True, reduction_fac = 2): 
       # x is assumed to be (batch, seqlen, d_model)
        #attentions is assumed to be (batch,num_heads, seqlen, seqlen )

        if attention_based:
           attentions =  torch.sum(attention, dim = (1,2))
           sorted_att, indices = torch.topk(attentions,int(attentions.shape[1]/reduction_fac), dim = -1, sorted = False) #dont know which to reduce(row or col)
           x = torch.cat( [x[i,indices[i]].unsqueeze(0) for i in range(x.shape[0])] ) 
           mask = torch.cat( [mask[i,indices[i]].unsqueeze(0) for i in range(x.shape[0])] )
           trackarray = torch.cat( [trackarray[i,indices[i]].unsqueeze(0) for i in range(x.shape[0])] )
        
        if similarity_based:
            x = x
        return x, mask, trackarray



class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

    

class TransformerPredictor(nn.Module):

    def __init__(self, input_dim, model_dim, num_classes, num_heads, num_layers,reduce = False, lr=1e-4,batch_size=8, dropout=0.1, weight_decay = 1e-2, mode = "mlm"):
        """
        Inputs:
            input_dim - Hidden dimensionality of the input
            model_dim - Hidden dimensionality to use inside the Transformer
            num_classes - Number of classes to predict per sequence element
            num_heads - Number of heads to use in the Multi-Head Attention blocks
            num_layers - Number of encoder blocks to use.
            lr - Learning rate in the optimizer
            dropout - Dropout to apply inside the model
        """
        super().__init__()
        self.dropout = dropout
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.lr = lr
        self.mode = mode
        self.loss = nn.CrossEntropyLoss()
        self.reduce = reduce
        self.batch_size = batch_size
        self.padding = True
        self.weight_decay = weight_decay
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        from transformers import BertConfig
        self.config = BertConfig()
        
        self.create_new_head(num_classes)
      #  self._create_model()
        
       # self.model = BertModel(self.config)#.from_pretrained('bert-base-uncased')
        self.model =  HourglassTransformerLM(num_tokens = self.config.vocab_size,dim = 768,causal = False,attn_resampling = True,
        max_seq_len = 1024,shorten_factor = 2,depth = (3, (3, (3, 3, 3), 3), 3), heads = 12)


        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay = self.weight_decay)

    def _create_model(self):
        # Input dim -> Model 
        self.embeddings = BertEmbeddings(self.config)
        # self.word_embeddings = nn.Embedding(self.input_dim, self.model_dim, padding_idx=self.config.pad_token_id)
        # self.position_embeddings = nn.Embedding(self.config.max_position_embeddings, self.model_dim)
        # self.token_type_embeddings = nn.Embedding(1, self.model_dim)
        # self.register_buffer("position_ids", torch.arange(self.config.max_position_embeddings).expand((1, -1)))
        # self.emblayernorm = nn.LayerNorm(self.model_dim)
        # self.embdropout = nn.Dropout(self.dropout)

        # self.input_net = nn.Sequential(
        #   #  nn.Dropout(self.input_dropout),
        #     nn.Linear(self.input_dim, self.model_dim),
        #     # nn.ReLU(inplace=True),
        #     # nn.Linear(self.model_dim, self.model_dim),
        # )
        # self.seg_embedding = nn.Sequential(nn.Linear(1, self.model_dim)
        # )
        # # Positional encoding for sequences
        # self.positional_encoding = PositionalEncoding(d_model=self.model_dim)
        # # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.model_dim, nhead=self.num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        # self.transformer = TransformerEncoder(num_layers=self.num_layers,
        #                                       d_model=self.model_dim,
        #                                       n_head=self.num_heads,
        #                                       reduce = self.reduce)

        # Output classifier per sequence lement
        self.output_net = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.LayerNorm(self.model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.model_dim, self.num_classes),
      #      nn.Softmax(dim = -1)
        )

    def create_new_head(self, num_classes):
        self.output_net = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.LayerNorm(self.model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.model_dim, num_classes)
        )
        self.output_net.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
    

    def oldforward(self, x, token_type_ids = None, mask=None, add_positional_encoding=True):
        """
        Inputs:
            x - Input features of shape [Batch, SeqLen, input_dim]
            mask - Mask to apply on the attention outputs (optional)
            add_positional_encoding - If True, we add the positional encoding to the input.
                                      Might not be desired for some tasks.
        """

        # inputs_embeds = self.word_embeddings(x)
        # token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # embeddings = inputs_embeds + token_type_embeddings
        # if add_positional_encoding:
        #     position_ids = self.position_ids[:, : x.size()[1] ]
        #     position_embeddings = self.position_embeddings(position_ids)
        #     embeddings += position_embeddings
        # embeddings = self.emblayernorm(embeddings)
        # embedding_output = self.embdropout(embeddings)

    #    print(x)
      #  print(x.shape)
        embedding_output = self.embeddings(
            input_ids=x,
            token_type_ids=token_type_ids,
        )

        # x = F.one_hot(x, self.input_dim)
        # x = self.input_net(x.float())
        # if add_positional_encoding:
        #     x = self.positional_encoding(x)
        # if not token_type_ids == None:
        #     x = x + self.seg_embedding(token_type_ids.unsqueeze(-1).type(torch.float))
        # x = self.emblayernorm(x)
        # embedding_output = self.embdropout(x)
        # print(embedding_output)
        # print(embedding_output.shape)
        x = self.transformer(embedding_output, src_key_padding_mask=mask.transpose(0,1))
       # print(x.shape)
        if self.mode == "cls":
            x = x[:,0]
      #  print(x.shape)
            #x,_ = torch.max(x, dim = 1)
        x = self.output_net(x)
        #x = F.softmax(x, dim = 1)
        return x, trackarray

    def forward(self, x, mask=None, token_type_ids = None):
        x = self.model(x, mask = mask, token_type_ids = token_type_ids)   
        
       # x = x.last_hidden_state[:,0]
        if self.mode == "cls":
            x = x[:,0]
        x = self.output_net(x)

        return x


    def mask(self, input_ids, mask_token_id = 103):
        # create random array of floats in equal dimension to input_ids
        rand = torch.rand(input_ids.shape).to(device)
        # where the random array is less than 0.15, we set true
        mask_arr = (rand < 0.15) * (input_ids != 101) * (input_ids != 102)* (input_ids != 0)
        mask_arr1 = (rand < 0.15*0.8)* (input_ids != 101) * (input_ids != 102)* (input_ids != 0) #* (mask_arr)
        mask_arr2 = (0.15*0.8 < rand)* (rand < 0.15*0.9)* (input_ids != 101) * (input_ids != 102)* (input_ids != 0)
      #  mask_arr3 = (0.15*0.9 < rand < 0.15)* (input_ids != 101) * (input_ids != 102)* (input_ids != 0) not needed since just nothing is done

        input_ids = torch.where(mask_arr1, mask_token_id, input_ids)#80% normal masking
        input_ids = torch.where(mask_arr2, (torch.rand(1, device = device)* self.tokenizer.vocab_size).long(), input_ids)#10% random token
        #last 10% is just original token and does not need to be replaced
        return input_ids, mask_arr
    
    def compute_mlm_loss(self,output,labels):
        output = output.flatten(start_dim = 0, end_dim = 1)
        labels = labels.flatten(start_dim = 0, end_dim = 1)
        loss = self.loss(output, labels)
        return loss

    #todo test mlm results

    def fitmlm(self,dataset, steps, checkpoint_pth = None):
        self.scheduler =CosineWarmupScheduler(optimizer= self.optimizer, 
                                               warmup = 3000 ,
                                                max_iters = steps)
        wandb.init(project="mlmwith-hourglass")
        wandb.watch(self)
        self.mode = "mlm"
        runinngloss = 0.0
        stepsperloss = 0
        for i,data in enumerate(dataset):
            start = time.time()
            
            input = self.tokenizer(data, return_tensors="pt", padding=True, max_length = 256, truncation = True)
            input = input.to(device)
            labels = torch.clone(input["input_ids"])
            input["input_ids"], mask = self.mask(input["input_ids"])
            labels = torch.where(mask, labels, -100)

            self.zero_grad()

            output = self(input["input_ids"], mask = input["attention_mask"].bool())
            loss = self.compute_mlm_loss(output,labels)
            loss.backward()
            stepsperloss += 1
            runinngloss += loss.item()
            self.optimizer.step()
            self.scheduler.step()


            # sentence = torch.argmax(output[0], dim = -1)
            # strsent = self.tokenizer.decode(sentence)
            # print("og sentence",data[0])
            # print("pred sentence",strsent)
            # pred = torch.masked_select(sentence, mask)
            # sellabels = torch.masked_select(labels, mask)
            # print("og words",self.tokenizer.decode(sellabels))
            # print("pred words",self.tokenizer.decode(pred))
            
            if i % np.max((1,int(steps*0.001))) == 0:
                wandb.log({"loss": runinngloss / stepsperloss})
                wandb.log({"lr": self.scheduler.get_last_lr()[0]})
                with torch.no_grad():
                    y_pred = torch.argmax(output,dim = -1)
                    acc =  torch.sum(y_pred == labels)
                    wandb.log({"acc": acc.item()/self.batch_size})
                print( runinngloss/ stepsperloss, "at", i , "of", steps, "time per step",time.time()-start, "estimated time until end of epoch", (steps -i) * (time.time()-start))
                runinngloss = 0.0
                stepsperloss = 0 
            if not checkpoint_pth == None and i % np.max((1,int(steps*0.1))) == 0:
                torch.save(self, checkpoint_pth)
            if i >= steps:
                break


    def oldfit(self,X,y, epochs, num_classes):
        self.create_new_head(num_classes)
        self.scheduler =CosineWarmupScheduler(optimizer= self.optimizer, 
                                               warmup = math.ceil(len(X)*epochs *0.01 / self.batch_size) ,
                                                max_iters = math.ceil(len(X)*epochs  / self.batch_size))
        self.mode = "cls"
        wandb.init(project="my-test-project")
        wandb.config = {
            "learning_rate": self.lr,
            "num_layers": self.num_layers,
            "num_classes": num_classes,
            "batch_size": self.batch_size
            }
        wandb.watch(self)
        for e in range(epochs):
            print("at epoch", e)
            runinngloss = 0.0
            stepsperloss = 0
            for i in range(math.ceil(len(X) / self.batch_size)):
                start = time.time()
                ul = min((i+1) * self.batch_size, len(X))
                batch_x = X[i*self.batch_size: ul]
                batch_y = y[i*self.batch_size: ul]
                batch_x = self.tokenizer(batch_x, return_tensors="pt", padding=self.padding, max_length = 256, truncation = True)

                batch_y = batch_y.to(device)
                batch_x = batch_x.to(device)
                self.optimizer.zero_grad()
                # print(batch_x)
                # print(batch_y)
                y_pred,_ = self(batch_x["input_ids"], mask = batch_x["attention_mask"],token_type_ids = batch_x["token_type_ids"])
                # print(y_pred.shape)
                # print(y_pred)
                # print(batch_y)
                loss = self.loss(y_pred, batch_y)    
                loss.backward()

                runinngloss += loss.item()
                stepsperloss += 1
                self.optimizer.step()
                self.scheduler.step()
                if i % np.max((1,int((len(X)/self.batch_size)*0.001))) == 0:
                    wandb.log({"loss": runinngloss / stepsperloss})
                 #   print("lr", self.scheduler.get_last_lr())
                    wandb.log({"lr": self.scheduler.get_last_lr()[0]})
                    with torch.no_grad():
                        y_pred = torch.argmax(y_pred,dim = -1)
                        acc =  torch.sum(y_pred == batch_y)
                        wandb.log({"acc": acc.item()/self.batch_size})
                    print( runinngloss/ stepsperloss, "at", ul , "of", len(X), "time per step",time.time()-start, "estimated time until end of epoch", (math.ceil(len(X) / self.batch_size) -i) * (time.time()-start))
                    runinngloss = 0.0
                    stepsperloss = 0

    def fit(self,X,y, epochs, num_classes):
        self.create_new_head(num_classes)
        self.scheduler =CosineWarmupScheduler(optimizer= self.optimizer, 
                                               warmup = math.ceil(len(X)*epochs *0.01 / self.batch_size) ,
                                                max_iters = math.ceil(len(X)*epochs  / self.batch_size))
        self.mode = "cls"
        wandb.init(project="my-test-project")
        wandb.config = {
            "learning_rate": self.lr,
            "num_layers": self.num_layers,
            "num_classes": num_classes,
            "batch_size": self.batch_size
            }
        wandb.watch(self)
        for e in range(epochs):
            print("at epoch", e)
            runinngloss = 0.0
            stepsperloss = 0
            for i in range(math.ceil(len(X) / self.batch_size)):
                start = time.time()
                ul = min((i+1) * self.batch_size, len(X))
                batch_x = X[i*self.batch_size: ul]
                batch_y = y[i*self.batch_size: ul]
                batch_x = self.tokenizer(batch_x, return_tensors="pt", padding=self.padding, max_length = 256, truncation = True)

                batch_y = batch_y.to(device)
                batch_x = batch_x.to(device)
                self.optimizer.zero_grad()
                # print(batch_x)
                # print(batch_y)
                y_pred = self(batch_x["input_ids"], mask = batch_x["attention_mask"].bool(),token_type_ids = batch_x["token_type_ids"])
                # print(y_pred.shape)
                # print(y_pred)
                # print(batch_y)
                loss = self.loss(y_pred, batch_y)    
                loss.backward()

                runinngloss += loss.item()
                stepsperloss += 1
                self.optimizer.step()
                self.scheduler.step()
                if i % np.max((1,int((len(X)/self.batch_size)*0.001))) == 0:
                    wandb.log({"loss": runinngloss / stepsperloss})
                 #   print("lr", self.scheduler.get_last_lr())
                    wandb.log({"lr": self.scheduler.get_last_lr()[0]})
                    with torch.no_grad():
                        y_pred = torch.argmax(y_pred,dim = -1)
                        acc =  torch.sum(y_pred == batch_y)
                        wandb.log({"acc": acc.item()/self.batch_size})
                    print( runinngloss/ stepsperloss, "at", ul , "of", len(X), "time per step",time.time()-start, "estimated time until end of epoch", (math.ceil(len(X) / self.batch_size) -i) * (time.time()-start))
                    runinngloss = 0.0
                    stepsperloss = 0


    @torch.no_grad()
    def evaluate(self,X,y):
        acc  = 0.0
        for i in range(math.ceil(len(X) / self.batch_size)):
            start = time.time()
            ul = min((i+1) * self.batch_size, len(X))
            batch_x = X[i*self.batch_size: ul]
            batch_y = y[i*self.batch_size: ul]
            batch_x = self.tokenizer(batch_x, return_tensors="pt", padding=self.padding, max_length = 256, truncation = True)

            batch_y = batch_y.to(device)
            batch_x = batch_x.to(device)
            y_pred = self(batch_x["input_ids"], mask = batch_x["attention_mask"].bool(),token_type_ids = batch_x["token_type_ids"])
            y_pred = torch.argmax(y_pred,dim = -1)
            acc = acc + torch.sum(y_pred == batch_y)

        acc = acc / len(X)
        return acc.item()

                
        



    

  



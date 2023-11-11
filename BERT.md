# BERT

```mermaid
classDiagram
  class Module["torch.nn.Module"]
  class ModuleUtilsMixin[".modeling_utils.ModuleUtilsMixin"]
  class GenerationMixin[".generation.GenerationMixin"]
  class PushToHubMixin[".utils.PushToHubMixin"]
  class PeftAdapterMixin[".integrations.PeftAdapterMixin"]
  class PreTrainedModel[".modeling_utils.PreTrainedModel"]
  class BertPreTrainedModel[".models.bert.BertPreTrainedModel"]
  class BertModel[".models.bert.BertModel"] {
    +BertConfig config
    +BertEmbeddings embeddings
    +BertEncoder encoder
    +BertPooler pooler
  }

  Module <|-- PreTrainedModel
  ModuleUtilsMixin <|-- PreTrainedModel
  GenerationMixin <|-- PreTrainedModel
  PushToHubMixin <|-- PreTrainedModel
  PeftAdapterMixin <|-- PreTrainedModel
  PreTrainedModel <|-- BertPreTrainedModel
  BertPreTrainedModel <|-- BertModel
```

## BertEmbeddings

```mermaid
classDiagram
  class Module["torch.nn.Module"]
  class BertEmbeddings[".models.bert.BertEmbeddings"] {
    +torch.nn.Embedding word_embeddings
    +torch.nn.Embedding position_embeddings
    +torch.nn.Embedding token_type_embeddings
    +torch.nn.LayerNorm LayerNorm
    +torch.nn.Dropout dropout
  }

  Module <|-- BertEmbeddings
```

LayerNorm は Tensorflow の Checkpoint File から Model の変数を Load できるように Camel Case になっているらしい.

## BertEncoder

```mermaid
classDiagram
  class Module["torch.nn.Module"]
  class BertEncoder[".models.bert.BertEncoder"] {
    +BertConfig config
    +torch.nn.ModuleList~BertLayer~ layer
    +bool gradient_checkpointing
  }

  Module <|-- BertEncoder
```

## BertLayer

```mermaid
classDiagram
  class Module["torch.nn.Module"]
  class BertLayer[".models.bert.BertLayer"] {
    +int chunk_size_feed_forward
    +int seq_len_dim
    +BertAttention attention
    +bool is_decoder
    +bool add_cross_attention
    +BertAttention crossattention
    +BertIntermediate intermediate
    +BertOutput output
  }

  Module <|-- BertLayer
```

## BertAttention

```mermaid
classDiagram
  class Module["torch.nn.Module"]
  class BertAttention[".models.bert.BertAttention"] {
    +BertSelfAttention self
    +BertSelfOutput output
    +set pruned_heads 
  }
  class BertSelfAttention[".models.bert.BertSelfAttention"] {
    +int num_attention_heads
    +int attention_head_size
    +int all_head_size
    +torch.nn.Linear query
    +torch.nn.Linear key
    +torch.nn.Linear value
    +torch.nn.Dropout dropout
    +str position_embedding_type
    +bool is_decoder
  }
  class BertSelfOutput[".models.bert.BertSelfOutput"] {
    +torch.nn.Linear dense
    +torch.nn.LayerNorm LayerNorm
    +torch.nn.Dropout dropout
  }

  BertSelfAttention -- BertAttention
  BertSelfOutput -- BertAttention
  Module <|-- BertAttention
  Module <|-- BertSelfAttention
  Module <|-- BertSelfOutput
```

### BertSelfAttention

TODO: forward() について

## BertIntermediate

```mermaid
classDiagram
  class Module["torch.nn.Module"]
  class BertIntermediate[".models.bert.BertIntermediate"] {
    +int chunk_size_feed_forward
  }

  Module <|-- BertIntermediate
```

## BertOutput

```mermaid
classDiagram
  class Module["torch.nn.Module"]
  class BertOutput[".models.bert.BertOutput"] {
    +int chunk_size_feed_forward
  }

  Module <|-- BertOutput
```

## BertPooler

```mermaid
classDiagram
  class Module["torch.nn.Module"]
  class BertPooler[".models.bert.BertPooler"] {
    +torch.nn.Linear dense
    +torch.nn.Tanh activation
  }

  Module <|-- BertPooler
```


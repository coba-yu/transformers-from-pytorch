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

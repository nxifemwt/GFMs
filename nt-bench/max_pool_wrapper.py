import torch.nn as nn


class MaxPoolWrapper(nn.Module):
    def __init__(self, base_model, model_type, num_labels):
        super().__init__()
        self.base_model = base_model
        self.model_type = model_type.lower()
        if "hyena" in self.model_type:
            base_model.config.hidden_size = self.base_model.config.d_model
            print("Added hidden size to Hyena config")
        elif "caduceus" in self.model_type:
            base_model.config.hidden_size = self.base_model.config.d_model * 2
            print("Added hidden size to Caduceus config")
        self.score = nn.Linear(base_model.config.hidden_size, num_labels, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        if "hyena" in self.model_type or "caduceus" in self.model_type:
            output = self.base_model(input_ids, output_hidden_states=True)
            hidden_states = output.hidden_states[-1]
        elif "nt" in self.model_type or "mistral" in self.model_type:
            output = self.base_model(
                input_ids, attention_mask=attention_mask, output_hidden_states=True
            )
            hidden_states = output.hidden_states[-1]
        elif "dnabert" in self.model_type:
            hidden_states = self.base_model(input_ids, output_hidden_states=True)[1]
        elif "genalm" in self.model_type:
            hidden_states = self.base_model(
                input_ids, attention_mask=attention_mask
            ).hidden_states[-1]
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        max_pooled = hidden_states.max(dim=1)[0]
        logits = self.score(max_pooled)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.score.out_features), labels.view(-1))

        return (loss, logits) if loss is not None else logits


def create_max_pool_model(base_model, model_type, num_labels):
    wrapped_model = MaxPoolWrapper(base_model, model_type, num_labels)
    return wrapped_model

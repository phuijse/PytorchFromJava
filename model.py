import argparse
import torch
import torch.nn as nn


class DummyModel(nn.Module):

    def __init__(self, input_dim=2, hidden_dim=5, output_dim=2):
        super(type(self), self).__init__()
        activation = nn.ReLU()
        self.layers = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                    activation,
                                    nn.Dropout(0.2),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    activation,
                                    nn.Dropout(0.2),
                                    nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        _, x, _, _ = x['light_curve']
        x = x[0]
        for layer in self.layers:
            x = layer(x)
        return {'logits': x, 'class': x.argmax(dim=1)}


def create_dummy_data():
    return {'light_curve': [torch.ones(1, 1, 100),
                            torch.ones(1, 1, 100),
                            torch.ones(1, 1, 100),
                            torch.ones(1, 1, 100).to(torch.bool)
                            ]
            }


def create_model_trace(model_path):
    model = DummyModel(100, 1)
    model.eval()  # Eval has to be called before tracing!
    traced_module = torch.jit.trace(model, create_dummy_data(), strict=False)
    traced_module.save(model_path)


def run_model_trace(model_path):
    traced_model = torch.jit.load(model_path)
    with torch.no_grad():
        preds = traced_model.forward(create_dummy_data())
    print(preds['logits'], preds['class'].item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['create', 'run'])
    parser.add_argument('--model_path')
    args = parser.parse_args()
    match args.mode:
        case 'create':
            create_model_trace(args.model_path)
        case 'run':
            run_model_trace(args.model_path)

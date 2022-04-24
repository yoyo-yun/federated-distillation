import torch

model = torch.load('./center_model_bert-base-uncased.pkl')
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameter: %.2fM" % (total / 1e6))

model = torch.load('./center_model_scibert.pkl')
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameter: %.2fM" % (total / 1e6))

model = torch.load('./center_model_roberta-base.pkl')
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameter: %.2fM" % (total / 1e6))

model = torch.load('./center_model_biobert.pkl')
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameter: %.2fM" % (total / 1e6))
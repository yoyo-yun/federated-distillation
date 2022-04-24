import torch

model = torch.load('./center_model_ClinicalBERT.pkl')
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameter: %.2fM" % (total / 1e6))

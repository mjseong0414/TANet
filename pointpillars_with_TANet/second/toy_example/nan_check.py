import torch
from torch import nn
import pdb

fc1 = nn.Linear(100, 8)
relu = nn.ReLU(inplace=True)
fc2 = nn.Linear(3, 2)

x = torch.randn(1077, 100)
y_isnan = torch.isnan(torch.tensor([[[float("nan"), 3.00], [float("nan"), 3.00]], [[1.01, 2], [float("nan"), 3.00]], [[float("nan"), 1], [float("nan"), 3.00]]]))
# import pdb; pdb.set_trace()
y_numpy = y_isnan.numpy()

for i in range(len(y_numpy)):
    for m in range(len(y_numpy[i])):
        for e in range(len(y_numpy[i][m])):
            if int(y_numpy[i][m][e]) == 1:
                print(y_numpy[i][m], i, m)
        


# for i in y_numpy:
#     for k in i:
#         if int(k) == 1:
#             print("Ddddddddddddddddddddddddddddddd")



# x[0][0] = 
    
# y = fc1(x)



import torch
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score

def eval_normal(model, x, y, adj):
    with torch.no_grad():
        pred = torch.argmax(torch.softmax(model.cuda()(x.cuda().float(),
                                        adj.cuda()),
                            dim=1), dim=1)
    print(classification_report(y, pred.cpu()))
    return pred
    
def eval_hybrid(model, x, y, adjs):
    with torch.no_grad():
        prob = torch.softmax(model.cuda()(x.cuda().float(),
                                        *list(map(lambda s: s.float().cuda(), adjs))),
                            dim=1)
        pred = torch.argmax(torch.softmax(model.cuda()(x.cuda().float(),
                                        *list(map(lambda s: s.float().cuda(), adjs))),
                            dim=1), dim=1)
        f1 = f1_score(y,pred.cpu())
        acc = accuracy_score(y, pred.cpu())
    return pred, f1, acc, prob
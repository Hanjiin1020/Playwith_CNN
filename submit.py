model.eval()
preds = []

with torch.no_grad():
    for inputs, _ in test_loader:   # label 없어도 상관 없음
        inputs = inputs.cuda()
        outputs = model(inputs)     # shape = (batch, 1000)

        # top-1 예측 클래스 index 가져오기
        _, predicted = torch.max(outputs, 1)

        preds.extend(predicted.cpu().tolist())


logits_list = []

with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.cuda()
        outputs = model(inputs)   # logits shape = [batch, 1000]
        logits_list.append(outputs.cpu())
        
# 하나의 큰 tensor로 합치기
logits = torch.cat(logits_list, dim=0)

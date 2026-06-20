import torch


class Predictor:

    def __init__(
            self,
            model,
            device,
            threshold=0.5,
    ):
        self.model = model
        self.device = device
        self.threshold = threshold

    
    def predict_batch(self, x):
        self.model.to(self.device) # put model to device
        self.model.eval() # disable dropout
        x = x.to(self.device) # faster inference if GPU available

        with torch.no_grad():
            logits = self.model(x)
            logits = logits.squeeze() # remove extra dimension (e.g., [[0, 1, 2, 3]] => [0, 1, 2, 3])

            preds = (logits >= self.threshold).float()
        
        # use .cpu() to move the logits/preds back to CPU and return
        return logits.cpu(), preds.cpu()


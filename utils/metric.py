import torch
import torch.nn.functional as F

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # for label smoothing, the target is a soft one-hot batched matrix with shape (B, num_classes)
        # so change it to (B, )
        if len(target.shape) > 1:
            target = torch.argmax(target, dim=-1)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count


def cosine_similarity(text_embeds, image_embeds):
    """
    model_fn for CLIP as required in Cleverhans' attacks.
    :param image_embeds: Image tensor of shape (B, dim)
    :param text_embeddings: Precomputed text embeddings for all labels (N_labels, dim).
    :return: Similarity scores between the image and all text labels.
    """

    image_embeds = F.normalize(image_embeds, p=2., dim=-1)

    if type(text_embeds) == list:
        
        assert type(text_embeds) == list, "To use descriptors, text_embeds must be a list of tensors"

        for label in text_embeds: # label: (B, N_descriptions, 768)
            label = F.normalize(label, p=2., dim=-1)
            logit = torch.mm(image_embeds, label.t()).mean(-1).squeeze(0) # average over all descriptions
            logits_per_image.append(logit)

        logits_per_image = torch.cat(logits_per_image)
    else:
        text_embeds = F.normalize(text_embeds, p=2., dim=-1)
        logits_per_image = torch.matmul(image_embeds, text_embeds.t())

    return logits_per_image
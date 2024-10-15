import torch
import torch.nn as nn
import torch.nn.functional as F

threshold = -0.5

#start#
class MIouLoss(nn.Module):
	def __init__(self):
		super(MIouLoss, self).__init__()

	@staticmethod
	def score(outputs, labels, smooth=1):
		# outputs = F.softmax(outputs, dim=1).round().long()
		outputs = outputs.reshape(outputs.shape[0],-1)
		labels = labels.reshape(outputs.shape[0],-1)
		inter = torch.sum(outputs * labels, dim=-1)
		union = torch.sum(outputs, dim=-1) + torch.sum(labels, dim=-1) - inter + smooth
		# print('iou:', inter.shape, union.shape)

		score = torch.mean((inter + smooth) / union)
		return score
		
	@staticmethod
	def scorem(outputs, labels, start_idx=0):
		if isinstance(labels, dict):
			labels = labels['label']

		# onehot
		if isinstance(outputs, dict):
			if 'sdf_output' in outputs:
				seg_outputs = outputs['segment_output']
				sdf_output = outputs['sdf_output']

				num_classes = seg_outputs.shape[1]
				argmax_seg_outputs = torch.argmax(seg_outputs, dim=1)
				argmax_sdf_output = torch.argmax(sdf_output, dim=1)

				one_hot_seg_outputs = F.one_hot(argmax_seg_outputs, num_classes=num_classes).contiguous().permute(0, 3, 1, 2)
				sdf_in_seg_outputs = torch.sum(torch.where(one_hot_seg_outputs.bool(), sdf_output, torch.tensor(0)), dim=1)
				outlier_idx = (sdf_in_seg_outputs < threshold)

				new_argmax_outputs = torch.where(outlier_idx, argmax_sdf_output, argmax_seg_outputs)
				new_one_hot_outputs = F.one_hot(new_argmax_outputs, num_classes=num_classes).contiguous().permute(0, 3, 1, 2)
				outputs = new_one_hot_outputs
		else:
			seg_outputs = outputs
			num_classes = seg_outputs.shape[1]
			argmax_seg_outputs = torch.argmax(seg_outputs, dim=1)
			one_hot_seg_outputs = F.one_hot(argmax_seg_outputs, num_classes=num_classes).contiguous().permute(0, 3, 1, 2)
			outputs = one_hot_seg_outputs

		# label转为ont_hot
		if labels.shape != outputs.shape:
			labels = F.one_hot(labels, outputs.shape[1]).contiguous().permute(0, 3, 1, 2)

		score = sum([MIouLoss.score(outputs[:,i:i+1],labels[:,i:i+1]) for i in range(start_idx,outputs.shape[1])])
		return score/(outputs.shape[1]-start_idx)

	def forward(self, outputs, labels, smooth=1e-6):
		batch_size = outputs.size(0)
		nb_class = outputs.shape[1]

		if labels.shape != outputs.shape:
			labels = F.one_hot(labels, outputs.shape[1]).contiguous().permute(0,3,1,2)
		labels = labels.view(batch_size, nb_class, -1)
		outputs = F.softmax(outputs, dim=1).view(batch_size, nb_class, -1)
		# print('outputs labels:', outputs.shape, labels.shape)

		inter = torch.sum(outputs * labels, dim=-1)
		union = torch.sum(outputs, dim=-1) + torch.sum(labels, dim=-1) - inter + smooth
		# print('iou:', inter.shape, union.shape)

		score = torch.sum(inter / union)
		score = 1.0 - score / (float(batch_size) * float(nb_class))
		return score

class MDiceLoss(nn.Module):
	def __init__(self, bi=False):
		super(MDiceLoss, self).__init__()
		self.func = self.dice2 if bi else self.dice

	@staticmethod
	def score(outputs, labels, smooth=1):
		# print('score:', outputs.shape, labels.shape, outputs.max().item(), labels.max().item())
		# outputs = F.softmax(outputs, dim=1).round().long()
		outputs = outputs.reshape(outputs.shape[0],-1)
		labels = labels.reshape(outputs.shape[0],-1)
		inter = torch.sum(outputs * labels, dim=-1)
		union = torch.sum(outputs, -1) + torch.sum(labels, -1) + smooth
		# print('iou:', inter.shape, union.shape)

		score = (2*inter + smooth) / union
		return score.mean()
		
	@staticmethod
	def scores(outputs, labels):
		if isinstance(labels, dict):
			labels = labels['label']

		# onehot
		if isinstance(outputs, dict):
			if 'sdf_output' in outputs:
				seg_outputs = outputs['segment_output']
				sdf_output = outputs['sdf_output']

				num_classes = seg_outputs.shape[1]
				argmax_seg_outputs = torch.argmax(seg_outputs, dim=1)
				argmax_sdf_output = torch.argmax(sdf_output, dim=1)

				one_hot_seg_outputs = F.one_hot(argmax_seg_outputs, num_classes=num_classes).contiguous().permute(0, 3,
																												  1, 2)
				sdf_in_seg_outputs = torch.sum(torch.where(one_hot_seg_outputs.bool(), sdf_output, torch.tensor(0)),
											   dim=1)
				outlier_idx = (sdf_in_seg_outputs < -threshold)

				new_argmax_outputs = torch.where(outlier_idx, argmax_sdf_output, argmax_seg_outputs)
				new_one_hot_outputs = F.one_hot(new_argmax_outputs, num_classes=num_classes).contiguous().permute(0, 3,
																												  1, 2)
				outputs = new_one_hot_outputs
		else:
			seg_outputs = outputs
			num_classes = seg_outputs.shape[1]
			argmax_seg_outputs = torch.argmax(seg_outputs, dim=1)
			one_hot_seg_outputs = F.one_hot(argmax_seg_outputs, num_classes=num_classes).contiguous().permute(0, 3, 1,
																											  2)
			outputs = one_hot_seg_outputs

		# label转为ont_hot
		if labels.shape != outputs.shape:
			labels = F.one_hot(labels, outputs.shape[1]).contiguous().permute(0, 3, 1, 2)

		return [MDiceLoss.score(outputs[:,i:i+1], labels[:,i:i+1]).cpu().item() for i in range(outputs.shape[1])]
		
	@staticmethod
	def scorem(outputs, labels, start_idx=0):
		if isinstance(labels, dict):
			labels = labels['label']

		# onehot
		if isinstance(outputs, dict):
			if 'sdf_output' in outputs:
				seg_outputs = outputs['segment_output']
				sdf_output = outputs['sdf_output']

				num_classes = seg_outputs.shape[1]
				argmax_seg_outputs = torch.argmax(seg_outputs, dim=1)
				argmax_sdf_output = torch.argmax(sdf_output, dim=1)

				one_hot_seg_outputs = F.one_hot(argmax_seg_outputs, num_classes=num_classes).contiguous().permute(0, 3, 1, 2)
				sdf_in_seg_outputs = torch.sum(torch.where(one_hot_seg_outputs.bool(), sdf_output, torch.tensor(0)), dim=1)
				outlier_idx = (sdf_in_seg_outputs < threshold)

				new_argmax_outputs = torch.where(outlier_idx, argmax_sdf_output, argmax_seg_outputs)
				new_one_hot_outputs = F.one_hot(new_argmax_outputs, num_classes=num_classes).contiguous().permute(0, 3, 1, 2)
				outputs = new_one_hot_outputs
		else:
			seg_outputs = outputs
			num_classes = seg_outputs.shape[1]
			argmax_seg_outputs = torch.argmax(seg_outputs, dim=1)
			one_hot_seg_outputs = F.one_hot(argmax_seg_outputs, num_classes=num_classes).contiguous().permute(0, 3, 1, 2)
			outputs = one_hot_seg_outputs

		# label转为ont_hot
		if labels.shape != outputs.shape:
			labels = F.one_hot(labels, outputs.shape[1]).contiguous().permute(0, 3, 1, 2)

		score = sum([MDiceLoss.score(outputs[:,i:i+1],labels[:,i:i+1]) for i in range(start_idx,outputs.shape[1])])
		return score/(outputs.shape[1]-start_idx)

	def forward(self, outputs, labels):
		batch_size = outputs.size(0)
		nb_class = outputs.shape[1]
		# print('MDiceLoss:', outputs.shape, labels.shape)
		if labels.shape != outputs.shape:
			labels = F.one_hot(labels, outputs.shape[1]).contiguous().permute(0,3,1,2)
		labels = labels.view(batch_size, nb_class, -1)
		outputs = F.softmax(outputs, dim=1).view(batch_size, nb_class, -1)
		# labels = labels[:, self.class_ids, :]
		return self.func(outputs, labels)

	def dice2(self, outputs, labels):
		return self.dice(outputs, labels) + self.dice(1-outputs, 1-labels)

	def dice(self, outputs, labels, smooth=1e-6):
		batch_size = outputs.size(0)
		nb_class = outputs.shape[1]

		inter = torch.sum(outputs * labels, 2) + smooth
		union = torch.sum(outputs, 2) + torch.sum(labels, 2) + smooth

		score = torch.sum(2.0 * inter / union)
		score = 1.0 - score / (float(batch_size) * float(nb_class))

		return score







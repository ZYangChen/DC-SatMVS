import torch
import torch.nn.functional as F


def cas_mvsnet_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", None)

    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    depth_loss = None

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        depth_est = stage_inputs["depth"]
        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key]
        mask = mask > 0.5

        depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')

        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            total_loss += depth_loss_weights[stage_idx] * depth_loss
        else:
            total_loss += 1.0 * depth_loss

    return total_loss, depth_loss

def depth_distribution_similarity_loss(depth, depth_gt, mask, depth_min, depth_max):
    # depth_norm = depth * 128 / (depth_max - depth_min)[:,None,None]
    depth_norm = depth * 128 / (depth_max - depth_min)[:,None,None]
    # depth_norm = (depth - depth_min) * 128 / (depth_max - depth_min)[:,None,None]
    # depth_gt_norm = depth_gt * 128 / (depth_max - depth_min)[:,None,None]
    depth_gt_norm = depth_gt * 128 / (depth_max - depth_min)[:,None,None]
    # depth_gt_norm = (depth_gt - depth_min) * 128 / (depth_max - depth_min)[:,None,None]

    M_bins = 48
    kl_min = torch.min(torch.min(depth_gt), depth.mean()-3.*depth.std()).item()
    kl_max = torch.max(torch.max(depth_gt), depth.mean()+3.*depth.std()).item()
    bins = torch.linspace(kl_min, kl_max, steps=M_bins)

    kl_divs = []
    for i in range(len(bins) - 1):
        bin_mask = (depth_gt >= bins[i]) & (depth_gt < bins[i+1])
        merged_mask = mask & bin_mask

        if merged_mask.sum() > 0:
            p = depth_norm[merged_mask]
            # p_clamped = torch.clamp(p, min=1e-8)
            # 检查张量中是否有 NaN 值
            # contains_nan = torch.isnan(p).any().item()
            # if contains_nan:
            #     print("p中包含 NaN 值")
            q = depth_gt_norm[merged_mask]
            # 对 q 进行修正，确保不含零概率值
            # q_clamped = torch.clamp(q, min=1e-8)
            # contains_nan = torch.isnan(q).any().item()
            # if contains_nan:
            #     print("q中包含 NaN 值")
            # print('q',q)
            # kl_div = F.kl_div(torch.log(p_clamped)-torch.log(q_clamped), p, reduction='batchmean')
            # print('!!!!!!!!!1111',p.shape)
            # print('!!!!!!!!!2222',q.shape)
            kl_div = F.kl_div(F.log_softmax(p, dim=0)-F.log_softmax(q, dim=0), F.softmax(p, dim=0), reduction='batchmean')
            # print('1',kl_div)
            # contains_nan = torch.isnan(kl_div).any().item()
            # if contains_nan:
            #     print("kl_div中包含 NaN 值")
            # kl_div = torch.log(torch.clamp(kl_div, min=1e-8))
            kl_div = torch.log(torch.clamp(kl_div, min=1))
            # contains_nan = torch.isnan(kl_div).any().item()
            # if contains_nan:
            #     print("kl_div2中包含 NaN 值")
            # print('2',kl_div)
            kl_divs.append(kl_div)

    dds_loss = sum(kl_divs)
    return dds_loss

def STsatmvsloss(inputs, depth_gt_ms, mask_ms, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", None)
    # depth_loss_weights = kwargs.get("dlossw", [1, 1, 1])
    depth_values = kwargs.get("depth_values")
    # 获取 depth_values
    # depth_values = inputs[2]  # 假设 depth_values 是 inputs 的第三个元素
    # depth_values = inputs["depth_values"]
    # print('depth_values', depth_values)
    # print('inputs', inputs)
    # print('depth_values', type(depth_values))
    depth_min, depth_max = depth_values[:, 0], depth_values[:, -1]
    # print('min',depth_min.shape)

    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    dds_loss_stages = []
    depth_loss = None

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        depth_est = stage_inputs["depth"]
        # depth_est = stage_inputs["depth_filtered"]
        depth_fre = stage_inputs["depth_filtered"]

        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key]
        mask = mask > 0.5

        depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')
        # print(depth_fre)
        dds_loss = depth_distribution_similarity_loss(depth_fre, depth_gt, mask, depth_min, depth_max)
        # dds_loss = depth_distribution_similarity_loss(depth_est, depth_gt, mask, depth_min, depth_max)
        dds_loss_stages.append(dds_loss)

        # total loss
        lam1, lam2 = 0.8, 0.2
        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            total_loss += depth_loss_weights[stage_idx] * (lam1 * depth_loss + lam2 * dds_loss)
        else:
            total_loss += 1.0 * (lam1 * depth_loss + lam2 * dds_loss)

    return total_loss, depth_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.module import *
from modules.warping import *
from modules.depth_range import *
import matplotlib.pyplot as plt
from modules.fusion import FeatureFusion

def compute_depth_when_train(features, proj_matrices, depth_values, num_depth, cost_regularization, geo_model, use_qc):

    if not use_qc:
        proj_matrices = torch.unbind(proj_matrices, 1)

    assert len(features) == len(proj_matrices), "Different number of images and projection matrices"
    assert depth_values.shape[1] == num_depth, "depth_values.shape[1]:{}  num_depth:{}".format(depth_values.shapep[1],
                                                                                               num_depth)
    num_views = len(features)

    # step 1. feature extraction
    # in: images; out: 32-channel feature maps
    ref_feature, src_features = features[0], features[1:]
    ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

    # step 2. differentiable homograph, build cost volume
    ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
    volume_sum = ref_volume
    volume_sq_sum = ref_volume ** 2
    del ref_volume

    if geo_model == "rpc" and not use_qc:
        # Create tensor in advance to save time
        b_num, f_num, img_h, img_w = ref_feature.shape
        coef = torch.ones((b_num, img_h * img_w * num_depth, 20), dtype=torch.double).cuda()
    else:
        coef = None

    for src_fea, src_proj in zip(src_features, src_projs):
        # warpped features
        if geo_model == "rpc" and not use_qc:
            warped_volume = rpc_warping(src_fea, src_proj, ref_proj, depth_values, coef)
        elif geo_model == "rpc" and use_qc:
            warped_volume = rpc_warping_enisum(src_fea, src_proj, ref_proj, depth_values)
        else:
            warped_volume = homo_warping(src_fea, src_proj, ref_proj, depth_values)

        volume_sum = volume_sum + warped_volume
        volume_sq_sum = volume_sq_sum + warped_volume ** 2

        del warped_volume

    # aggregate multiple feature volumes by variance
    volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))

    # step 3. cost volume regularization
    prob_volume = cost_regularization(volume_variance)
    # cost_reg = F.upsample(cost_reg, [num_depth * 4, img_height, img_width], mode='trilinear')
    prob_volume = F.softmax(prob_volume, dim=1)

    # regression
    depth = depth_regression(prob_volume, depth_values=depth_values)
    photometric_confidence, indices = prob_volume.max(1)

    return {"depth": depth, "photometric_confidence": photometric_confidence}


# train+test SAMSatMVS
class ST_SatMVS(nn.Module):
    def __init__(self, geo_model, min_interval=2.5, ndepths=[48, 32, 8],
                 depth_interals_ratio=[4, 2, 1], cr_base_chs=[8, 8, 8], use_qc=False):
        super(ST_SatMVS, self).__init__()
        self.geo_model = geo_model
        assert self.geo_model in ["rpc", "pinhole"]
        self.ndepths = ndepths
        self.depth_interals_ratio = depth_interals_ratio
        self.cr_base_chs = cr_base_chs
        self.num_stage = len(ndepths)
        # print('len',self.num_stage)
        self.min_interval = min_interval
        self.use_qc = use_qc

        if self.use_qc and geo_model == "rpc":
            print("Quaternary Cubic Form is used for RPC warping")
        print("**********netphs:{}, depth_intervals_ratio:{}, chs:{}************".format(ndepths,
              depth_interals_ratio, self.cr_base_chs))
        assert len(ndepths) == len(depth_interals_ratio)
        if self.num_stage == 3:
            self.stage_infos = {
                "stage1": {
                    "scale": 4.0,
                },
                "stage2": {
                    "scale": 2.0,
                },
                "stage3": {
                    "scale": 1.0,
                }
            }
        if self.num_stage == 2:
            self.stage_infos = {
                "stage1":{
                    "scale": 4.0,
                },
                "stage2": {
                    "scale": 1.0,
                }
            }

        self.feature = FeatureNet(base_channels=8, stride=4, num_stage=self.num_stage, arch_mode='fpn') # unet
        self.CoarestFeatureNet = TransformerNet(base_channels=8, stride=4, num_stage=self.num_stage, arch_mode='fpn') #FPN
        # print('len', len(self.feature.out_channels), len(self.cr_base_chs))
        self.cost_regularization = nn.ModuleList([RED_Regularization(in_channels=self.feature.out_channels[i],
                                                                     base_channels=self.cr_base_chs[i])
                                                  for i in range(self.num_stage)])

        self.GeoFeatureFusionNet = FeatureFusion(convolutional_layer_encoding="z", mask_type="basic", add_origin_feat_flag=True)
        # frequency domain filter settings
        self.curriculum_learning_rho_ratios = [4, 2, 1]

    def forward(self, imgs, proj_matrices, depth_values):
        B, N, C, H, W = imgs.shape
        # step 1. feature extraction
        features = []
        coarsest_features = []
        for nview_idx in range(imgs.size(1)):  #imgs shape (B, N, C, H, W)
            img = imgs[:, nview_idx]
            features.append(self.feature(img))
            coarsest_features.append(self.CoarestFeatureNet(img))

        imgs_transposed = torch.transpose(imgs, 1, 2)
        imgs_flattened = imgs_transposed.reshape(B * C, N, H, W)
        # 选择 N 维度的某个索引，例如，选择第一个 N 的张量
        selected_img = imgs_flattened[:, 0, :, :]
        # 将结果转回形状为 (B, C, H, W)
        img_ = selected_img.reshape(B, C, H, W)

        img_h = int(img.shape[2])
        img_w = int(img.shape[3])
        outputs = {}
        depth, cur_depth = None, None
        for stage_idx in range(self.num_stage):
            # print("*********************stage{}*********************".format(stage_idx + 1))
            #stage feature, proj_mats, scales
            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
            stage_scale = self.stage_infos["stage{}".format(stage_idx + 1)]["scale"]
            if stage_idx == 0:
                features_stage = [feat["stage{}".format(stage_idx + 1)] for feat in coarsest_features]
            else:
                features_stage = [feat["stage{}".format(stage_idx + 1)] for feat in features]

                # print('img',img_.shape)
                ref_img_stage = F.interpolate(img_, size=None, scale_factor=1. / 2 ** (self.num_stage - 1 - stage_idx),
                                              mode="bilinear", align_corners=False)
                # print('img',img[0].shape)
                # depth_last = F.interpolate(depth_last.unsqueeze(1), size=None, scale_factor=2, mode="bilinear", align_corners=False)
                # print('last0',depth_last.shape)
                depth_last = F.interpolate(depth_last.unsqueeze(1), size=None, scale_factor=2, mode="bilinear", align_corners=False)
                # print('last1',depth_last.shape)
                confidence_last = F.interpolate(confidence_last.unsqueeze(1), size=None, scale_factor=2,
                                                mode="bilinear", align_corners=False)
                # print('00',features_stage[0].shape)
                # reference feature
                features_stage[0] = self.GeoFeatureFusionNet(
                    ref_img_stage, depth_last, confidence_last, depth_values,
                    stage_idx, features_stage[0], proj_matrices_stage
                )
                # print('01',features_stage[0].shape)

            if depth is not None:
                cur_depth = depth
                cur_depth = F.interpolate(cur_depth.unsqueeze(1),
                                          [img_h, img_w], mode='bilinear', align_corners=False).squeeze(1)
            else:
                cur_depth = depth_values
            depth_range_samples = get_depth_range_samples(
                cur_depth=cur_depth, ndepth=self.ndepths[stage_idx],
                depth_inteval_pixel=self.depth_interals_ratio[stage_idx] * self.min_interval,
                dtype=img[0].dtype, device=img[0].device, shape=[img.shape[0], img_h, img_w])

            dv = F.interpolate(depth_range_samples.unsqueeze(1),
                               [self.ndepths[stage_idx], img.shape[2] // int(stage_scale),
                                img.shape[3] // int(stage_scale)], mode='trilinear', align_corners=False)

            outputs_stage = compute_depth_when_train(
                features_stage, proj_matrices_stage, depth_values=dv.squeeze(1), num_depth=self.ndepths[stage_idx],
                cost_regularization=self.cost_regularization[stage_idx], geo_model=self.geo_model, use_qc=self.use_qc)

            depth = outputs_stage['depth']
            # print('depth',depth.shape)
            depth_est_filtered = frequency_domain_filter(depth, rho_ratio=self.curriculum_learning_rho_ratios[stage_idx])
            outputs_stage['depth_filtered'] = depth_est_filtered
            # print('depth_est_filtered', depth_est_filtered.shape)
            depth_last = depth_est_filtered
            confidence_last = outputs_stage['photometric_confidence']

            outputs["stage{}".format(stage_idx + 1)] = outputs_stage
            outputs.update(outputs_stage)

        return outputs

def fftshift(x, dim=None):
    if dim is None:
        dim = tuple(range(x.dim()))
    elif not isinstance(dim, tuple):
        dim = (dim,)
    for d in dim:
        n = x.size(d)
        p2 = (n + 1) // 2
        x = torch.cat((x.index_select(d, torch.arange(p2, n, device=x.device)),
                       x.index_select(d, torch.arange(p2, device=x.device))), dim=d)
    return x

def ifftshift(x, dim=None):
    if dim is None:
        dim = tuple(range(x.dim()))
    elif not isinstance(dim, tuple):
        dim = (dim,)
    for d in dim:
        n = x.size(d)
        p2 = n - (n + 1) // 2
        x = torch.cat((x.index_select(d, torch.arange(p2, n, device=x.device)),
                       x.index_select(d, torch.arange(p2, device=x.device))), dim=d)
    return x

def frequency_domain_filter(depth, rho_ratio):
    """
    large rho_ratio -> more information filtered
    """
    # f = torch.fft.fft2(depth)
    f = torch.rfft(depth, signal_ndim=2, normalized=False, onesided=False)
    # print('f',f.shape)
    # fshift = torch.fft.fftshift(f)
    fshift = fftshift(f, dim=(-2, -1))
    # print('f2',fshift.shape)

    b, h, w = depth.shape
    k_h, k_w = h/rho_ratio, w/rho_ratio
    # print('k',k_h,k_w)

    # 确定对称地设置为零的范围
    h_start, h_end = int(h / 2 - k_h / 2), int(h / 2 + k_h / 2)
    w_start, w_end = int(w / 2 - k_w / 2), int(w / 2 + k_w / 2)

    # 通过将正负分量都设置为零来确保对称性
    fshift[:, :h_start, :] = 0
    fshift[:, h_end:, :] = 0
    fshift[:, :, :w_start] = 0
    fshift[:, :, w_end:] = 0

    # 确保正确处理奇数维度
    if h % 2 != 0:
        fshift[:, int(h / 2), :] = 0
    if w % 2 != 0:
        fshift[:, :, int(w / 2)] = 0
    # print('f3',fshift.shape)
    # ishift = torch.fft.ifftshift(fshift)
    ishift = ifftshift(fshift, dim=(-2, -1))
    # print('f4',ishift.shape)
    # idepth = torch.fft.ifft2(ishift)
    idepth = torch.irfft(ishift, signal_ndim=2, normalized=False, onesided=False)
    # print('idepth',idepth.shape)
    depth_filtered = torch.abs(idepth)

    return depth_filtered

def frequency_domain_filter1(depth, rho_ratio):
    """
    large rho_ratio -> more information filtered
    """
    # f = torch.fft.fft2(depth)
    f = torch.rfft(depth, signal_ndim=2)
    # fshift = torch.fft.fftshift(f)
    fshift = fftshift(f, dim=(-2, -1))

    b, h, w = depth.shape
    k_h, k_w = h/rho_ratio, w/rho_ratio

    fshift[:,:int(h/2-k_h/2),:] = 0
    fshift[:,int(h/2+k_h/2):,:] = 0
    fshift[:,:,:int(w/2-k_w/2)] = 0
    fshift[:,:,int(w/2+k_w/2):] = 0

    # ishift = torch.fft.ifftshift(fshift)
    ishift = ifftshift(fshift, dim=(-2, -1))
    # idepth = torch.fft.ifft2(ishift)
    idepth = torch.irfft(ishift, signal_ndim=2)
    depth_filtered = torch.abs(idepth)

    return depth_filtered
##################Inference########################################

def compute_depth_when_pred(features, proj_matrices, depth_values, num_depth, cost_regularization, geo_model, use_qc):
    proj_matrices = torch.unbind(proj_matrices, 1)
    assert len(features) == len(proj_matrices), "Different number of images and projection matrices"
    assert depth_values.shape[1] == num_depth, "depth_values.shape[1]:{}  num_depth:{}".format(
        depth_values.shapep[1], num_depth)
    num_views = len(features)

    # step 1. feature extraction
    # in: images; out: 32-channel feature maps
    ref_feature, src_features = features[0], features[1:]
    ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

    # step 2. differentiable homograph, build cost volume

    b_num, f_num, img_h, img_w = ref_feature.shape
    state1 = torch.zeros((b_num, 8, img_h, img_w)).cuda()
    state2 = torch.zeros((b_num, 16, int(img_h / 2), int(img_w / 2))).cuda()
    state3 = torch.zeros((b_num, 32, int(img_h / 4), int(img_w / 4))).cuda()
    state4 = torch.zeros((b_num, 64, int(img_h / 8), int(img_w / 8))).cuda()

    # initialize variables
    exp_sum = torch.zeros((b_num, 1, img_h, img_w), dtype=torch.double).cuda()
    depth_image = torch.zeros((b_num, 1, img_h, img_w), dtype=torch.double).cuda()
    max_prob_image = torch.zeros((b_num, 1, img_h, img_w), dtype=torch.double).cuda()

    if geo_model == "rpc":
        coef = torch.ones((b_num, img_h * img_w * 1, 20), dtype=torch.double).cuda()
    else:
        coef = None

    for d in range(num_depth):
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, 1, 1, 1)
        depth_value = depth_values[:, d:d + 1]

        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2
        del ref_volume

        for src_fea, src_proj in zip(src_features, src_projs):
            if geo_model == "rpc":
                warped_volume = rpc_warping(src_fea, src_proj, ref_proj, depth_value, coef)
            else:
                warped_volume = homo_warping(src_fea, src_proj, ref_proj, depth_value)
            # TODO: this is only a temporal solution to save memory, better way?

            volume_sum += warped_volume
            volume_sq_sum += warped_volume.pow_(2)  # the memory of warped_volume has been modified
            del warped_volume

        # aggregate multiple feature volumes by variance
        volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))
        volume_variance = volume_variance.squeeze(2)

        # step 3. Recurrent Regularization
        # start = time.time()
        reg_cost, state1, state2, state3, state4 = cost_regularization(volume_variance, state1, state2, state3, state4)

        reg_cost = reg_cost.double()
        prob = reg_cost.exp()

        update_flag_image = (max_prob_image < prob).double()
        new_max_prob_image = update_flag_image * prob + (1 - update_flag_image) * max_prob_image

        # update the best
        # new_depth_image = update_flag_image * depth_value + (1 - update_flag_image) * depth_image
        # update the sum_avg
        new_depth_image = depth_value.double() * prob + depth_image

        max_prob_image = new_max_prob_image
        depth_image = new_depth_image
        exp_sum = exp_sum + prob

    # update the sum_avg
    forward_exp_sum = exp_sum + 1e-10
    forward_depth_map = (depth_image / forward_exp_sum).squeeze(1).float()
    forward_prob_map = (max_prob_image / forward_exp_sum).squeeze(1).float()

    return {"depth": forward_depth_map, "photometric_confidence": forward_prob_map}


# predict  CascadeREDNet
class Infer_CascadeREDNet(nn.Module):
    def __init__(self, geo_model, min_interval=2.5, ndepths=[48, 32, 8], depth_interals_ratio=[4, 2, 1],
                 cr_base_chs=[8, 8, 8], use_qc=False):
        super(Infer_CascadeREDNet, self).__init__()
        self.geo_model = geo_model
        assert self.geo_model in ["rpc", "pinhole"]
        self.ndepths = ndepths
        self.depth_interals_ratio = depth_interals_ratio
        self.cr_base_chs = cr_base_chs
        self.num_stage = len(ndepths)
        self.min_interval = min_interval
        self.use_qc = use_qc

        print("**********netphs:{}, depth_intervals_ratio:{}, chs:{}************".format(ndepths,
              depth_interals_ratio, self.cr_base_chs))
        assert len(ndepths) == len(depth_interals_ratio)
        if self.num_stage == 3:
            self.stage_infos = {
                "stage1":{
                    "scale": 4.0,
                },
                "stage2": {
                    "scale": 2.0,
                },
                "stage3": {
                    "scale": 1.0,
                }
            }
        if self.num_stage == 2:
            self.stage_infos = {
                "stage1": {
                    "scale": 4.0,
                },
                "stage2": {
                    "scale": 1.0,
                }
            }

        self.feature = FeatureNet(base_channels=8, stride=4, num_stage=self.num_stage, arch_mode='unet') # unet
        self.cost_regularization = nn.ModuleList([slice_RED_Regularization(
            in_channels=self.feature.out_channels[i], base_channels=self.cr_base_chs[i])
            for i in range(self.num_stage)])

    def forward(self, imgs, proj_matrices, depth_values):
        # step 1. feature extraction
        features = []
        for nview_idx in range(imgs.size(1)):  #imgs shape (B, N, C, H, W)
            img = imgs[:, nview_idx]
            features.append(self.feature(img))

        img_h = int(img.shape[2])
        img_w = int(img.shape[3])
        outputs = {}
        depth, cur_depth = None, None
        for stage_idx in range(self.num_stage):
            #stage feature, proj_mats, scales
            features_stage = [feat["stage{}".format(stage_idx + 1)] for feat in features]
            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
            stage_scale = self.stage_infos["stage{}".format(stage_idx + 1)]["scale"]

            if depth is not None:
                cur_depth = depth
                cur_depth = F.interpolate(cur_depth.unsqueeze(1),
                                          [img_h, img_w], mode='bilinear',
                                          align_corners=False).squeeze(1)
            else:
                cur_depth = depth_values
            depth_range_samples = get_depth_range_samples(
                cur_depth=cur_depth, ndepth=self.ndepths[stage_idx],
                depth_inteval_pixel=self.depth_interals_ratio[stage_idx] * self.min_interval,
                dtype=img[0].dtype, device=img[0].device, shape=[img.shape[0], img_h, img_w])

            dv = F.interpolate(depth_range_samples.unsqueeze(1),
                               [self.ndepths[stage_idx], img.shape[2] // int(stage_scale),
                                img.shape[3] // int(stage_scale)], mode='trilinear', align_corners=False)

            outputs_stage = compute_depth_when_pred(features_stage, proj_matrices_stage,
                                                    depth_values=dv.squeeze(1),
                                                    num_depth=self.ndepths[stage_idx],
                                                    cost_regularization=self.cost_regularization[stage_idx],
                                                    geo_model=self.geo_model, use_qc=self.use_qc)

            depth = outputs_stage['depth']

            """import matplotlib.pyplot as plt
            plt.imshow(depth.cpu().numpy()[0])
            plt.show()"""

            outputs["stage{}".format(stage_idx + 1)] = outputs_stage
            outputs.update(outputs_stage)

        return outputs
"""
Adapted from  https://github.com/SaoYan/DnCNN-PyTorch SaoYan
Re-implemented by Yuqian Zhou
"""
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from torch.autograd import Variable


class RealNoisyEstimator(nn.Module):
    """
    Noise estimator, with original 3 layers
    """

    def __init__(self, input_channels=1, output_channels=3, num_of_layers=3):
        super(RealNoisyEstimator, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=features,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            )
        )
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(
                nn.Conv2d(
                    in_channels=features,
                    out_channels=features,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(
                in_channels=features,
                out_channels=output_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            )
        )
        self.dncnn = nn.Sequential(*layers)

    def forward(self, input):
        x = self.dncnn(input)
        return x


class NoisyEstimator(nn.Module):
    def __init__(self, channels, num_of_layers=17, num_of_est=3):
        super(NoisyEstimator, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels=channels + num_of_est,
                out_channels=features,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            )
        )
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(
                nn.Conv2d(
                    in_channels=features,
                    out_channels=features,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(
                in_channels=features,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            )
        )
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x, c):
        input_x = torch.cat([x, c], dim=1)
        out = self.dncnn(input_x)
        return out


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode="fan_in")
    elif classname.find("Linear") != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode="fan_in")
    elif classname.find("BatchNorm") != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2.0 / 9.0 / 64.0)).clamp_(
            -0.025, 0.025
        )
        nn.init.constant(m.bias.data, 0.0)


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return PSNR / Img.shape[0]


def data_augmentation(image, mode):
    out = np.transpose(image, (1, 2, 0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2, 0, 1))


def visual_va2np(
    Out,
    mode=1,
    ps=0,
    pss=1,
    scal=1,
    rescale=0,
    w=10,
    h=10,
    c=3,
    refill=0,
    refill_img=0,
    refill_ind=[0, 0],
):
    if mode == 0 or mode == 1 or mode == 3:
        out_numpy = Out.data.squeeze(0).cpu().numpy()
    elif mode == 2:
        out_numpy = Out.data.squeeze(1).cpu().numpy()
    if out_numpy.shape[0] == 1:
        out_numpy = np.tile(out_numpy, (3, 1, 1))
    if mode == 0 or mode == 1:
        out_numpy = (np.transpose(out_numpy, (1, 2, 0))) * 255.0 * scal
    else:
        out_numpy = np.transpose(out_numpy, (1, 2, 0))

    if ps == 1:
        out_numpy = reverse_pixelshuffle(out_numpy, pss, refill, refill_img, refill_ind)
    if rescale == 1:
        out_numpy = cv2.resize(out_numpy, (h, w))
        # print(out_numpy.shape)
    return out_numpy


def temp_ps_4comb(Out, In):
    pass


def np2ts(x, mode=0):  # now assume the input only has one channel which is ignored
    w, h, c = x.shape
    x_ts = x.transpose(2, 0, 1)
    x_ts = torch.from_numpy(x_ts).type(torch.FloatTensor)
    if mode == 0 or mode == 1:
        x_ts = x_ts.unsqueeze(0)
    elif mode == 2:
        x_ts = x_ts.unsqueeze(1)
    return x_ts


def np2ts_4d(x):
    x_ts = x.transpose(0, 3, 1, 2)
    x_ts = torch.from_numpy(x_ts).type(torch.FloatTensor)
    return x_ts


def get_salient_noise_in_maps(lm, thre=0.0, chn=3):
    """
    Description: To find out the most frequent estimated noise level in the images
    ----------
    [Input]
    a multi-channel tensor of noise map

    [Output]
    A list of  noise level value
    """
    lm_numpy = lm.data.cpu().numpy()
    lm_numpy = np.transpose(lm_numpy, (0, 2, 3, 1))
    nl_list = np.zeros((lm_numpy.shape[0], chn, 1))
    for n in range(lm_numpy.shape[0]):
        for c in range(chn):
            selected_lm = np.reshape(
                lm_numpy[n, :, :, c], (lm_numpy.shape[1] * lm_numpy.shape[2], 1)
            )
            selected_lm = selected_lm[selected_lm > thre]
            if selected_lm.shape[0] == 0:
                nl_list[n, c] = 0
            else:
                hist = np.histogram(selected_lm, density=True)
                nl_ind = np.argmax(hist[0])
                # print(nl_ind)
                # print(hist[0])
                # print(hist[1])
                nl = (hist[1][nl_ind] + hist[1][nl_ind + 1]) / 2.0
                nl_list[n, c] = nl
    return nl_list


def get_cdf_noise_in_maps(lm, thre=0.8, chn=3):
    """
    Description: To find out the most frequent estimated noise level in the images
    ----------
    [Input]
    a multi-channel tensor of noise map

    [Output]
    A list of  noise level value
    """
    lm_numpy = lm.data.cpu().numpy()
    lm_numpy = np.transpose(lm_numpy, (0, 2, 3, 1))
    nl_list = np.zeros((lm_numpy.shape[0], chn, 1))
    for n in range(lm_numpy.shape[0]):
        for c in range(chn):
            selected_lm = np.reshape(
                lm_numpy[n, :, :, c], (lm_numpy.shape[1] * lm_numpy.shape[2], 1)
            )
            H, x = np.histogram(selected_lm, normed=True)
            dx = x[1] - x[0]
            F = np.cumsum(H) * dx
            F_ind = np.where(F > 0.9)[0][0]
            nl_list[n, c] = x[F_ind]
            print(nl_list[n, c])
    return nl_list


def get_pdf_in_maps(lm, mark, chn=1):
    """
    Description: get the noise estimation cdf of each channel
    ----------
    [Input]
    a multi-channel tensor of noise map and channel dimension
    chn: the channel number for gaussian
    [Output]
    CDF function of each sample and each channel
    """
    lm_numpy = lm.data.cpu().numpy()
    lm_numpy = np.transpose(lm_numpy, (0, 2, 3, 1))
    pdf_list = np.zeros((lm_numpy.shape[0], chn, 10))
    for n in range(lm_numpy.shape[0]):
        for c in range(chn):
            selected_lm = np.reshape(
                lm_numpy[n, :, :, c], (lm_numpy.shape[1] * lm_numpy.shape[2], 1)
            )
            H, x = np.histogram(selected_lm, range=(0.0, 1.0), bins=10, normed=True)
            dx = x[1] - x[0]
            F = H * dx
            pdf_list[n, c, :] = F
            # sio.savemat(mark + str(c) + '.mat',{'F':F})
            plt.bar(range(10), F)
            # plt.savefig(mark + str(c) + '.png')
            plt.close()
    return pdf_list


def get_pdf_matching_score(F1, F2):
    """
    Description: Given two sets of CDF, get the overall matching score for each channel
    -----------
    [Input] F1, F2
    [Output] score for each channel
    """
    return np.mean((F1 - F2) ** 2)


def decide_scale_factor(
    noisy_image, estimation_model, color=1, thre=0, stopping=4, mark=""
):
    """
    Description: Given a noisy image and the noise estimation model, keep multiscaling the image\\
                 using pixel-shuffle methods, and estimate the pdf and cdf of AWGN channel
                 Compare the changes of the density function and decide the optimal scaling factor
    ------------
    [Input]  noisy_image, estimation_model, plot_flag, stopping
    [Output]  plot the middle vector
              score_seq: the matching score sequence between the two subsequent pdf
              opt_scale: the optimal scaling factor 
    """
    if color == 1:
        c = 3
    elif color == 0:
        c = 1
    score_seq = []
    Pre_CDF = None
    flag = 0
    for pss in range(1, stopping + 1):  # scaling factor from 1 to the limit
        noisy_image = pixelshuffle(noisy_image, pss)
        INoisy = np2ts(noisy_image, color)
        INoisy = Variable(INoisy, volatile=True)
        EMap = torch.clamp(estimation_model(INoisy), 0.0, 1.0)
        EPDF = get_pdf_in_maps(EMap, mark + str(pss), c)[0]
        if flag != 0:
            score = get_pdf_matching_score(
                EPDF, Pre_PDF
            )  # TODO: How to match these two
            print(score)
            score_seq.append(score)
            if score <= thre:
                print("optimal scale is %d:" % (pss - 1))
                return (pss - 1, score_seq)
        Pre_PDF = EPDF
        flag = 1
    return (stopping, score_seq)


def get_max_noise_in_maps(lm, chn=3):
    """
    Description: To find out the maximum level of noise level in the images
    ----------
    [Input]
    a multi-channel tensor of noise map

    [Output]
    A list of  noise level value
    """
    lm_numpy = lm.data.cpu().numpy()
    lm_numpy = np.transpose(lm_numpy, (0, 2, 3, 1))
    nl_list = np.zeros((lm_numpy.shape[0], chn, 1))
    for n in range(lm_numpy.shape[0]):
        for c in range(chn):
            nl = np.amax(lm_numpy[n, :, :, c])
            nl_list[n, c] = nl
    return nl_list


def get_smooth_maps(lm, dilk=50, gsd=10):
    """
    Description: To return the refined maps after dilation and gaussian blur
    [Input] a multi-channel tensor of noise map
    [Output] a multi-channel tensor of refined noise map
    """
    kernel = np.ones((dilk, dilk))
    lm_numpy = lm.data.squeeze(0).cpu().numpy()
    lm_numpy = np.transpose(lm_numpy, (1, 2, 0))
    ref_lm_numpy = lm_numpy.copy()  # a refined map
    for c in range(lm_numpy.shape[2]):
        nmap = lm_numpy[:, :, c]
        nmap_dilation = cv2.dilate(nmap, kernel, iterations=1)
        ref_lm_numpy[:, :, c] = nmap_dilation
        # ref_lm_numpy[:, :, c] = scipy.ndimage.filters.gaussian_filter(nmap_dilation, gsd)
    RF_tensor = np2ts(ref_lm_numpy)
    RF_tensor = Variable(RF_tensor, volatile=True)


def zeroing_out_maps(lm, keep=0):
    """
    Only Keep one channel and zero out other channels
    [Input] a multi-channel tensor of noise map
    [Output] a multi-channel tensor of noise map after zeroing out items
    """
    lm_numpy = lm.data.squeeze(0).cpu().numpy()
    lm_numpy = np.transpose(lm_numpy, (1, 2, 0))
    ref_lm_numpy = lm_numpy.copy()  # a refined map
    for c in range(lm_numpy.shape[2]):
        if np.isin(c, keep) == 0:
            ref_lm_numpy[:, :, c] = 0.0
    print(ref_lm_numpy)
    RF_tensor = np2ts(ref_lm_numpy)
    RF_tensor = Variable(RF_tensor, volatile=True)
    return RF_tensor


def level_refine(NM_tensor, ref_mode, chn=3):
    """
    Description: To refine the estimated noise level maps
    [Input] the noise map tensor, and a refinement mode
    Mode:
    [0] Get the most salient (the most frequent estimated noise level)
    [1] Get the maximum value of noise level
    [2] Gaussian smooth the noise level map to make the regional estimation more smooth
    [3] Get the average maximum value of the noise level
    [5] Get the CDF thresholded value 
    
    [Output] a refined map tensor with four channels
    """
    # RF_tensor = NM_tensor.clone()  #get a clone version of NM tensor without changing the original one
    if (
        ref_mode == 0 or ref_mode == 1 or ref_mode == 4 or ref_mode == 5
    ):  # if we use a single value for the map
        if ref_mode == 0 or ref_mode == 4:
            nl_list = get_salient_noise_in_maps(NM_tensor, 0.0, chn)

            if ref_mode == 4:  # half the estimation
                nl_list = nl_list - nl_list
            print(nl_list)
        elif ref_mode == 1:
            nl_list = get_max_noise_in_maps(NM_tensor, chn)
        elif ref_mode == 5:
            nl_list = get_cdf_noise_in_maps(NM_tensor, 0.999, chn)

        noise_map = np.zeros(
            (NM_tensor.shape[0], chn, NM_tensor.size()[2], NM_tensor.size()[3])
        )  # initialize the noise map before concatenating
        for n in range(NM_tensor.shape[0]):
            noise_map[n, :, :, :] = np.reshape(
                np.tile(nl_list[n], NM_tensor.size()[2] * NM_tensor.size()[3]),
                (chn, NM_tensor.size()[2], NM_tensor.size()[3]),
            )
        RF_tensor = torch.from_numpy(noise_map).type(torch.FloatTensor)
        RF_tensor = Variable(RF_tensor, volatile=True)

    elif ref_mode == 2:
        RF_tensor = get_smooth_maps(NM_tensor, 10, 5)
    elif ref_mode == 3:
        lb = get_salient_noise_in_maps(NM_tensor)
        up = get_max_noise_in_maps(NM_tensor)
        nl_list = (lb + up) * 0.5
        noise_map = np.zeros(
            (1, chn, NM_tensor.size()[2], NM_tensor.size()[3])
        )  # initialize the noise map before concatenating
        noise_map[0, :, :, :] = np.reshape(
            np.tile(nl_list, NM_tensor.size()[2] * NM_tensor.size()[3]),
            (chn, NM_tensor.size()[2], NM_tensor.size()[3]),
        )
        RF_tensor = torch.from_numpy(noise_map).type(torch.FloatTensor)
        RF_tensor = Variable(RF_tensor, volatile=True)

    return (RF_tensor, nl_list)


def normalize(a, len_v, min_v, max_v):
    """
    normalize the sequence of factors
    """
    norm_a = np.reshape(a, (len_v, 1))
    norm_a = (norm_a - float(min_v)) / float(max_v - min_v)
    return norm_a


def generate_training_noisy_image(current_image, s_or_m, limit_set, c, val=0):
    noise_level_list = np.zeros((c, 1))
    if s_or_m == 0:  # single noise type
        if val == 0:
            for chn in range(c):
                noise_level_list[chn] = np.random.uniform(
                    limit_set[0][0], limit_set[0][1]
                )
        elif val == 1:
            for chn in range(c):
                noise_level_list[chn] = 35
        noisy_img = generate_noisy(current_image, 0, noise_level_list / 255.0)

    return (noisy_img, noise_level_list)


def generate_ground_truth_noise_map(
    noise_map, n, noise_level_list, limit_set, c, pn, pw, ph
):
    for chn in range(c):
        noise_level_list[chn] = normalize(
            noise_level_list[chn], 1, limit_set[0][0], limit_set[0][1]
        )  # normalize the level value
    noise_map[n, :, :, :] = np.reshape(
        np.tile(noise_level_list, pw * ph), (c, pw, ph)
    )  # total number of channels
    return noise_map


# Add noise to the original images
def generate_noisy(image, noise_type, noise_level_list=0, sigma_s=20, sigma_c=40):
    """
    Description: To generate noisy images of different types
    ----------
    [Input]
    image : ndarray of float type: [0,1] just one image, current support gray or color image input (w,h,c)
    noise_type: 0,1,2,3
    noise_level_list: pre-defined noise level for each channel, without normalization: only information of 3 channels
    [0]'AWGN'     Multi-channel Gaussian-distributed additive noise
    [1]'RVIN'    Replaces random pixels with 0 or 1.  noise_level: ratio of the occupation of the changed pixels
    [2]'Gaussian-Poisson'   GP noise approximator, the combinatin of signal-dependent and signal independent noise
    [Output]
    A noisy image
    """
    w, h, c = image.shape
    # Some unused noise type: Poisson and Uniform
    # if noise_type == *:
    # vals = len(np.unique(image))
    # vals = 2 ** np.ceil(np.log2(vals))
    # noisy = np.random.poisson(image * vals) / float(vals)

    # if noise_type == *:
    # uni = np.random.uniform(-factor,factor,(w, h, c))
    # uni = uni.reshape(w, h, c)
    # noisy = image + uni

    noisy = image.copy()

    if noise_type == 0:  # MC-AWGN model
        gauss = np.zeros((w, h, c))
        for chn in range(c):
            gauss[:, :, chn] = np.random.normal(0, noise_level_list[chn], (w, h))
        noisy = image + gauss
    elif noise_type == 1:  # MC-RVIN model
        for chn in range(c):  # process each channel separately
            prob_map = np.random.uniform(0.0, 1.0, (w, h))
            noise_map = np.random.uniform(0.0, 1.0, (w, h))
            noisy_chn = noisy[:, :, chn]
            noisy_chn[prob_map < noise_level_list[chn]] = noise_map[
                prob_map < noise_level_list[chn]
            ]

    elif noise_type == 2:
        pass

    return noisy


# generate AWGN-RVIN noise together
def generate_comp_noisy(image, noise_level_list):

    """
    Description: To generate mixed AWGN and RVIN noise together
    ----------
    [Input]
    image: a float image between [0,1]
    noise_level_list: AWGN and RVIN noise level
    [Output]
    A noisy image
    """
    w, h, c = image.shape
    noisy = image.copy()
    for chn in range(c):
        mix_thre = noise_level_list[c + chn]  # get the mix ratio of AWGN and RVIN
        gau_std = noise_level_list[chn]  # get the gaussian std
        prob_map = np.random.uniform(0, 1, (w, h))  # the prob map
        noise_map = np.random.uniform(0, 1, (w, h))  # the noisy map
        noisy_chn = noisy[:, :, chn]
        noisy_chn[prob_map < mix_thre] = noise_map[prob_map < mix_thre]
        gauss = np.random.normal(0, gau_std, (w, h))
        noisy_chn[prob_map >= mix_thre] = (
            noisy_chn[prob_map >= mix_thre] + gauss[prob_map >= mix_thre]
        )

    return noisy


def generate_denoise(image, model, noise_level_list):
    """
    Description: Generate Denoised Blur Images
    ----------
    [Input]
    image: 
    model: 
    noise_level_list: 
    
    [Output]
    A blur image patch
    """
    # input images
    ISource = np2ts(image)
    ISource = torch.clamp(ISource, 0.0, 1.0)
    ISource = Variable(ISource, volatile=True)
    # input denoise conditions
    noise_map = np.zeros(
        (1, 6, image.shape[0], image.shape[1])
    )  # initialize the noise map before concatenating
    noise_map[0, :, :, :] = np.reshape(
        np.tile(noise_level_list, image.shape[0] * image.shape[1]),
        (6, image.shape[0], image.shape[1]),
    )
    NM_tensor = torch.from_numpy(noise_map).type(torch.FloatTensor)
    NM_tensor = Variable(NM_tensor, volatile=True)
    # generate blur images
    Res = model(ISource, NM_tensor)
    Out = torch.clamp(ISource - Res, 0.0, 1.0)
    out_numpy = Out.data.squeeze(0).cpu().numpy()
    out_numpy = np.transpose(out_numpy, (1, 2, 0))
    return out_numpy


# TODO: two pixel shuffle functions to process the images
def pixelshuffle(image, scale):
    """
    Discription: Given an image, return a reversible sub-sampling
    [Input]: Image ndarray float
    [Return]: A mosic image of shuffled pixels
    """
    if scale == 1:
        return image
    w, h, c = image.shape
    mosaic = np.array([])
    for ws in range(scale):
        band = np.array([])
        for hs in range(scale):
            temp = image[ws::scale, hs::scale, :]  # get the sub-sampled image
            band = np.concatenate((band, temp), axis=1) if band.size else temp
        mosaic = np.concatenate((mosaic, band), axis=0) if mosaic.size else band
    return mosaic


def reverse_pixelshuffle(image, scale, fill=0, fill_image=0, ind=[0, 0]):
    """
    Discription: Given a mosaic image of subsampling, recombine it to a full image
    [Input]: Image
    [Return]: Recombine it using different portions of pixels
    """
    w, h, c = image.shape
    real = np.zeros((w, h, c))  # real image
    wf = 0
    hf = 0
    for ws in range(scale):
        hf = 0
        for hs in range(scale):
            temp = real[ws::scale, hs::scale, :]
            wc, hc, cc = temp.shape  # get the shpae of the current images
            if fill == 1 and ws == ind[0] and hs == ind[1]:
                real[ws::scale, hs::scale, :] = fill_image[
                    wf : wf + wc, hf : hf + hc, :
                ]
            else:
                real[ws::scale, hs::scale, :] = image[wf : wf + wc, hf : hf + hc, :]
            hf = hf + hc
        wf = wf + wc
    return real


def scal2map(level, h, w, min_v=0.0, max_v=255.0):
    """
    Change a single normalized noise level value to a map
    [Input]: level: a scaler noise level(0-1), h, w
    [Return]: a pytorch tensor of the cacatenated noise level map
    """
    # get a tensor from the input level
    level_tensor = torch.from_numpy(np.reshape(level, (1, 1))).type(torch.FloatTensor)
    # make the noise level to a map
    level_tensor = level_tensor.view(stdN_tensor.size(0), stdN_tensor.size(1), 1, 1)
    level_tensor = level_tensor.repeat(1, 1, h, w)
    return level_tensor


def scal2map_spatial(level1, level2, h, w):
    stdN_t1 = scal2map(level1, int(h / 2), w)
    stdN_t2 = scal2map(level2, h - int(h / 2), w)
    stdN_tensor = torch.cat([stdN_t1, stdN_t2], dim=2)
    return stdN_tensor


# the limitation range of each type of noise level: [0]Gaussian [1]Impulse
limit_set = [[0, 75], [0, 80]]


def img_normalize(data):
    return data / 255.0


def denoiser(Img, c, pss, model, model_est, opt):

    w, h, _ = Img.shape
    Img = pixelshuffle(Img, pss)

    noise_level_list = np.zeros((2 * c, 1))  # two noise types with three channels
    if (
        opt.cond == 0
    ):  # if we use the ground truth of noise for denoising, and only one single noise type
        noise_level_list = np.array(opt.test_noise_level)
    elif opt.cond == 2:  # if we use an external fixed input condition for denoising
        noise_level_list = np.array(opt.ext_test_noise_level)

    # Clean Image Tensor for evaluation
    ISource = np2ts(Img)
    # noisy image and true residual
    if (
        opt.real_n == 0 and opt.spat_n == 0
    ):  # no spatial noise setting, and synthetic noise
        noisy_img = generate_comp_noisy(Img, np.array(opt.test_noise_level) / 255.0)
        if opt.color == 0:
            noisy_img = np.expand_dims(noisy_img[:, :, 0], 2)
    elif opt.real_n == 1 or opt.real_n == 2:  # testing real noisy images
        noisy_img = Img
    elif opt.spat_n == 1:
        noisy_img = generate_noisy(Img, 2, 0, 20, 40)
    INoisy = np2ts(noisy_img, opt.color)
    INoisy = torch.clamp(INoisy, 0.0, 1.0)
    True_Res = INoisy - ISource
    ISource, INoisy, True_Res = (
        Variable(ISource, volatile=True),
        Variable(INoisy, volatile=True),
        Variable(True_Res, volatile=True),
    )

    if opt.mode == "MC":
        # obtain the corrresponding input_map
        if (
            opt.cond == 0 or opt.cond == 2
        ):  # if we use ground choose level or the given fixed level
            # normalize noise leve map to [0,1]
            noise_level_list_n = np.zeros((2 * c, 1))
            print(c)
            for noise_type in range(2):
                for chn in range(c):
                    noise_level_list_n[noise_type * c + chn] = normalize(
                        noise_level_list[noise_type * 3 + chn],
                        1,
                        limit_set[noise_type][0],
                        limit_set[noise_type][1],
                    )  # normalize the level value
            # generate noise maps
            noise_map = np.zeros(
                (1, 2 * c, Img.shape[0], Img.shape[1])
            )  # initialize the noise map
            noise_map[0, :, :, :] = np.reshape(
                np.tile(noise_level_list_n, Img.shape[0] * Img.shape[1]),
                (2 * c, Img.shape[0], Img.shape[1]),
            )
            NM_tensor = torch.from_numpy(noise_map).type(torch.FloatTensor)
            NM_tensor = Variable(NM_tensor, volatile=True)
        # use the estimated noise-level map for blind denoising
        elif opt.cond == 1:  # if we use the estimated map directly
            NM_tensor = torch.clamp(model_est(INoisy), 0.0, 1.0)
            if (
                opt.refine == 1
            ):  # if we need to refine the map before putting it to the denoiser
                NM_tensor_bundle = level_refine(
                    NM_tensor, opt.refine_opt, 2 * c
                )  # refine_opt can be max, freq and their average
                NM_tensor = NM_tensor_bundle[0]
                noise_estimation_table = np.reshape(NM_tensor_bundle[1], (2 * c,))
            if opt.zeroout == 1:
                NM_tensor = zeroing_out_maps(NM_tensor, opt.keep_ind)
        Res = model(INoisy, NM_tensor)

    elif opt.mode == "B":
        Res = model(INoisy)

    Out = torch.clamp(INoisy - Res, 0.0, 1.0)  # Output image after denoising

    # get the maximum denoising result
    max_NM_tensor = level_refine(NM_tensor, 1, 2 * c)[0]
    max_Res = model(INoisy, max_NM_tensor)
    max_Out = torch.clamp(INoisy - max_Res, 0.0, 1.0)
    max_out_numpy = visual_va2np(
        max_Out, opt.color, opt.ps, pss, 1, opt.rescale, w, h, c
    )
    del max_Out
    del max_Res
    del max_NM_tensor

    if (opt.ps == 1 or opt.ps == 2) and pss != 1:  # pixelshuffle multi-scale
        # create batch of images with one subsitution
        mosaic_den = visual_va2np(Out, opt.color, 1, pss, 1, opt.rescale, w, h, c)
        out_numpy = np.zeros((pss ** 2, c, w, h))
        # compute all the images in the ps scale set
        for row in range(pss):
            for column in range(pss):
                re_test = (
                    visual_va2np(
                        Out,
                        opt.color,
                        1,
                        pss,
                        1,
                        opt.rescale,
                        w,
                        h,
                        c,
                        1,
                        visual_va2np(INoisy, opt.color),
                        [row, column],
                    )
                    / 255.0
                )
                # cv2.imwrite(os.path.join(opt.out_dir,file_name + '_%d_%d.png' % (row, column)), re_test[:,:,::-1]*255.)
                re_test = np.expand_dims(re_test, 0)
                if opt.color == 0:  # if gray image
                    re_test = np.expand_dims(re_test[:, :, :, 0], 3)
                re_test_tensor = torch.from_numpy(
                    np.transpose(re_test, (0, 3, 1, 2))
                ).type(torch.FloatTensor)
                re_test_tensor = Variable(re_test_tensor, volatile=True)
                re_NM_tensor = torch.clamp(model_est(re_test_tensor), 0.0, 1.0)
                if (
                    opt.refine == 1
                ):  # if we need to refine the map before putting it to the denoiser
                    re_NM_tensor_bundle = level_refine(
                        re_NM_tensor, opt.refine_opt, 2 * c
                    )  # refine_opt can be max, freq and their average
                    re_NM_tensor = re_NM_tensor_bundle[0]
                re_Res = model(re_test_tensor, re_NM_tensor)
                Out2 = torch.clamp(re_test_tensor - re_Res, 0.0, 1.0)
                out_numpy[row * pss + column, :, :, :] = Out2.data.cpu().numpy()
                del Out2
                del re_Res
                del re_test_tensor
                del re_NM_tensor
                del re_test

        out_numpy = np.mean(out_numpy, 0)
        out_numpy = np.transpose(out_numpy, (1, 2, 0)) * 255.0
    elif opt.ps == 0 or pss == 1:  # other cases
        out_numpy = visual_va2np(Out, opt.color, 0, 1, 1, opt.rescale, w, h, c)

    out_numpy = out_numpy.astype(np.float32)  # details
    max_out_numpy = max_out_numpy.astype(np.float32)  # background

    # merging the details and background to balance the effect
    k = opt.k
    merge_out_numpy = (1 - k) * out_numpy + k * max_out_numpy
    merge_out_numpy = merge_out_numpy.astype(np.float32)

    return merge_out_numpy

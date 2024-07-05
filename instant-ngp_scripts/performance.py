import os
import numpy as np
import imageio.v2 as imageio  # imageio.v2를 명시적으로 사용
import lpips
from skimage import io, color
from scipy.ndimage import convolve1d
import torch
import flip
import flip.utils

# LPIPS 모델 로드
lpips_model = lpips.LPIPS(net='alex')

# 이미지 로드 함수
def load_image(image_path):
    image = io.imread(image_path)
    if image.ndim == 2:  # 만약 이미지가 흑백이라면
        image = color.gray2rgb(image)  # RGB 형식으로 변환
    return image

# LPIPS 점수 계산 함수
def calculate_lpips(img1, img2):
    img1_tensor = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0).float()
    img2_tensor = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0).float()
    lpips_score = lpips_model(img1_tensor, img2_tensor)
    return lpips_score.item()

# 재투영 오류 계산 함수
def calculate_reprojection_error(img1, img2):
    return np.mean(np.sqrt(np.sum((img1 - img2) ** 2, axis=-1)))

# 폴더 내 모든 이미지에 대해 성능 계산
def evaluate_images(folder1, folder2):
    mse_scores = []
    psnr_scores = []
    ssim_scores = []
    lpips_scores = []
    reprojection_errors = []

    image_files1 = sorted([f for f in os.listdir(folder1) if os.path.isfile(os.path.join(folder1, f)) and f.endswith('.jpg')])
    image_files2 = sorted([f for f in os.listdir(folder2) if os.path.isfile(os.path.join(folder2, f)) and f.endswith('.jpg')])

    for img_file1, img_file2 in zip(image_files1, image_files2):
        img1_path = os.path.join(folder1, img_file1)
        img2_path = os.path.join(folder2, img_file2)

        img1 = read_image(img1_path)
        img2 = read_image(img2_path)

        # MSE 계산
        mse_value = compute_error("MSE", img1, img2)
        mse_scores.append(mse_value)

        # PSNR 계산
        psnr_value = mse2psnr(mse_value)
        psnr_scores.append(psnr_value)

        # SSIM 계산
        ssim_value = compute_error("SSIM", img1, img2)
        ssim_scores.append(ssim_value)

        # LPIPS 계산
        lpips_score = calculate_lpips(img1, img2)
        lpips_scores.append(lpips_score)

        # 재투영 오류 계산
        reprojection_error = calculate_reprojection_error(img1, img2)
        reprojection_errors.append(reprojection_error)

    # 결과 저장 및 출력
    results = {
        'MSE': mse_scores,
        'PSNR': psnr_scores,
        'SSIM': ssim_scores,
        'LPIPS': lpips_scores,
        'R/E': reprojection_errors,
    }

    for metric, scores in results.items():
        print(f"{metric} 평균: {np.mean(scores)}, 분산: {np.var(scores)}, 표준편차: {np.std(scores)}")
    
    #print(results)
    return results

# SSIM 계산 함수 정의
def compute_error(metric, img, ref):
    metric_map = compute_error_img(metric, img, ref)
    metric_map[np.logical_not(np.isfinite(metric_map))] = 0
    if len(metric_map.shape) == 3:
        metric_map = np.mean(metric_map, axis=2)
    mean = np.mean(metric_map)
    return mean

def compute_error_img(metric, img, ref):
    img[np.logical_not(np.isfinite(img))] = 0
    img = np.maximum(img, 0.)
    if metric == "MSE":
        return L2(img, ref)
    elif metric == "SSIM":
        return SSIM(np.clip(img, 0.0, 1.0), np.clip(ref, 0.0, 1.0))
    elif metric in ["FLIP", "\FLIP"]:
        # Set viewing conditions
        monitor_distance = 0.7
        monitor_width = 0.7
        monitor_resolution_x = 3840
        # Compute number of pixels per degree of visual angle
        pixels_per_degree = monitor_distance * (monitor_resolution_x / monitor_width) * (np.pi / 180)

        ref_srgb = np.clip(flip.color_space_transform(ref, "linrgb2srgb"), 0, 1)
        img_srgb = np.clip(flip.color_space_transform(img, "linrgb2srgb"), 0, 1)
        result = flip.compute_flip(flip.utils.HWCtoCHW(ref_srgb), flip.utils.HWCtoCHW(img_srgb), pixels_per_degree)
        assert np.isfinite(result).all()
        return flip.utils.CHWtoHWC(result)

    raise ValueError(f"Unknown metric: {metric}.")

def mse2psnr(x):
    return -10. * np.log10(x)

def L2(img, ref):
    return (img - ref)**2

def trim(error, skip=0.000001):
    error = np.sort(error.flatten())
    size = error.size
    skip = int(skip * size)
    return error[skip:size-skip].mean()

def luminance(a):
    return 0.2126 * a[:,:,0] + 0.7152 * a[:,:,1] + 0.0722 * a[:,:,2]

def SSIM(a, b):
    def blur(a):
        k = np.array([0.120078, 0.233881, 0.292082, 0.233881, 0.120078])
        x = convolve1d(a, k, axis=0)
        return convolve1d(x, k, axis=1)
    a = luminance(a)
    b = luminance(b)
    mA = blur(a)
    mB = blur(b)
    sA = blur(a*a) - mA**2
    sB = blur(b*b) - mB**2
    sAB = blur(a*b) - mA*mB
    c1 = 0.01**2
    c2 = 0.03**2
    p1 = (2.0*mA*mB + c1)/(mA*mA + mB*mB + c1)
    p2 = (2.0*sAB + c2)/(sA + sB + c2)
    error = p1 * p2
    return error

def read_image(file):
    if os.path.splitext(file)[1] == ".bin":
        with open(file, "rb") as f:
            bytes = f.read()
            h, w = struct.unpack("ii", bytes[:8])
            img = np.frombuffer(bytes, dtype=np.float16, count=h*w*4, offset=8).astype(np.float32).reshape([h, w, 4])
    else:
        img = read_image_imageio(file)
        if img.shape[2] == 4:
            img[...,0:3] = srgb_to_linear(img[...,0:3])
            # Premultiply alpha
            img[...,0:3] *= img[...,3:4]
        else:
            img = srgb_to_linear(img)
    return img

def read_image_imageio(img_file):
    img = imageio.imread(img_file)
    img = np.asarray(img).astype(np.float32)
    if len(img.shape) == 2:
        img = img[:,:,np.newaxis]
    return img / 255.0

def srgb_to_linear(img):
    limit = 0.04045
    return np.where(img > limit, np.power((img + 0.055) / 1.055, 2.4), img / 12.92)

# 폴더 경로 설정
folder1 = "/home/jwp/instant-ngp/data/nerf/hanyang_100_human/images"
folder2 = "/home/jwp/instant-ngp/data/nerf/hanyang_100_human/screenshot"

# 성능 평가 실행
results = evaluate_images(folder1, folder2)

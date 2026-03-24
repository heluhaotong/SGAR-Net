import os
import torch
from model.model import Model as TRIS
import cv2
import numpy as np
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import clip


# 获取图像转换
def get_transform(size=None):
    if size is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((416, 416)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transform


# 生成归一化的CAM
def get_norm_cam(cam):
    cam = torch.clamp(cam, min=0)
    cam_t = cam.unsqueeze(0).unsqueeze(0).flatten(2)
    cam_max = torch.max(cam_t, dim=2).values.unsqueeze(2).unsqueeze(3)
    cam_min = torch.min(cam_t, dim=2).values.unsqueeze(2).unsqueeze(3)
    norm_cam = (cam - cam_min) / (cam_max - cam_min + 1e-5)
    norm_cam = norm_cam.squeeze(0).squeeze(0).cpu().numpy()
    return norm_cam


# 准备数据
def prepare_data(img_path, text, max_length=17):
    img = cv2.imread(img_path)

    word_ids = []
    split_text = text.split(',')
    tokenizer = clip.tokenize
    for text in split_text:
        word_id = tokenizer(text).squeeze(0)[:max_length]
        word_ids.append(word_id.unsqueeze(0))
    word_ids = torch.cat(word_ids, dim=-1)

    # 创建 word_mask
    word_mask = (word_ids != 0).float()  # 0表示填充部分

    h, w, c = img.shape
    img = Image.fromarray(img)
    transform = get_transform(size=img_size)
    img = transform(img)

    return img, word_ids, word_mask, h, w

def visualize_cam_with_mask(normalized_heatmap, original=None, root=None, mask_color=(0,127,120), alpha=0.3):
    """
    Visualizes the CAM and overlays a light flesh-colored mask on the image.
    :param normalized_heatmap: The normalized heatmap to visualize.
    :param original: The original image to overlay the heatmap on.
    :param root: Path to save the image.
    :param mask_color: The color of the mask (light flesh color by default).
    :param alpha: Transparency of the mask overlay.
    :return: Image with CAM and mask overlay.
    """
    # Resize heatmap to match original image size
    heatmap_resized = cv2.resize(normalized_heatmap, (original.shape[1], original.shape[0]))

    # Create binary mask from heatmap (threshold at 0.5)
    mask = (heatmap_resized > 0.5).astype(np.uint8)

    # Convert original image from BGR to RGB for display
    original_img = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    # Create colored mask
    mask_color = np.array(mask_color, dtype=np.uint8)
    colored_mask = np.zeros_like(original_img)
    colored_mask[:] = mask_color

    # Apply mask to original image
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    img_with_mask = np.where(mask_3d,
                             cv2.addWeighted(original_img, 1 - alpha, colored_mask, alpha, 0),
                             original_img)

    # 保持与原来程序相同的热力图生成方式
    map_img = np.uint8(normalized_heatmap * 255)
    heatmap_img = cv2.applyColorMap(map_img, cv2.COLORMAP_JET)
    heatmap_img = cv2.resize(heatmap_img, (original.shape[1], original.shape[0]))
    heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)

    # 应用热力图到原图（与原来程序相同的方式）
    heatmap_on_original = cv2.addWeighted(heatmap_img, 0.6, original_img, 0.4, 0)

    # Concatenate original image, image with mask, and original heatmap style
    final_img = np.hstack((original_img, img_with_mask, heatmap_on_original))

    if root is not None:
        plt.imsave(root, final_img)

    return final_img


# 主程序
if __name__ == '__main__':
    image = cv2.imread("/home/admin123/PycharmProjects/ASDA/demo/ocid/6.png")
    height, width, channels = image.shape
    print(f"图像的高度: {height}, 图像的宽度: {width}, 通道数: {channels}")

    os.environ['CUDA_ENABLE_DEVICES'] = '0'
    img_size = 416
    max_length = 17

    model_path = '/home/admin123/PycharmProjects/ASDA/saved_models/coco/iou_best_checkpoint.pth'
    checkpoint = torch.load(model_path)
    print(checkpoint.keys())

    model = TRIS()
    model.cuda()
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    img_path = "/home/admin123/PycharmProjects/ASDA/demo/3.jpg"
    text = "A small piece of pizza "
    img, word_id, word_mask, h, w = prepare_data(img_path, text, max_length)
    model.eval()
    outputs = model(img.unsqueeze(0).cuda(),
                    word_id.unsqueeze(0).cuda(),
                    word_mask.unsqueeze(0).cuda())

    output = outputs
    pred = F.interpolate(output, (416, 416), align_corners=True, mode='bilinear').squeeze(0).squeeze(0)

    norm_cam = get_norm_cam(pred.detach().cpu())
    orig_img = cv2.imread(img_path)
    orig_img = cv2.resize(orig_img, (416, 416))

    visualize_cam_with_mask(norm_cam, orig_img, root=f"demo/demo_({text}).png")


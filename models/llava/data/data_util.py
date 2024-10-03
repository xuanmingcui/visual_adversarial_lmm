# BY ALEJANDRO APARCEDO
# EFFECT OF DATA AUGMENTATION ON LLaVA
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import random
import uuid
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np


def apply_data_augmentation(image, augmentation_technique, image_file):

    # List of data augmentation techniques
    techniques = [  "rot_90",
                    "rot_180",
                    "rot_270", 
                    "crop_resize", 
                    "crop_resize_flip", 
                    "cutout",
                    "color_jitter", 
                    "grayscale", 
                    "gaussian_noise", 
                    "gaussian_blur", 
                    "sobel_filter"]

    if augmentation_technique == 'none':
        return image


    # Randomly choose a technique
    if augmentation_technique == "random":
        augmentation_technique = random.choice(techniques)
        path = "sample_random"
    else:
        # path = "sample_" + augmentation_technique
        path = augmentation_technique


    # Generate a unique ID for the image
    image_id = uuid.uuid4()


    # ROTATE 90 DEGREES
    if augmentation_technique == "rot_90":
        rotated_image = TF.rotate(img=image, angle=90) 
        # rotated_image.save(f'./pics/{path}/{image_id}_rot_90.jpg')
        return rotated_image
    
    # ROTATE 180 DEGREES
    if augmentation_technique == "rot_180":
        rotated_image = TF.rotate(img=image, angle=180) 
        # rotated_image.save(f'./pics/{path}/{image_id}_rot_180.jpg')
        return rotated_image
    
    # ROTATE 270 DEGREES
    if augmentation_technique == "rot_270":
        rotated_image = TF.rotate(img=image, angle=270) 
        # rotated_image.save(f'./pics/{path}/{image_id}_rot_270.jpg')

        return rotated_image
    
    # CROP AND RESIZE THE IMAGE
    if augmentation_technique == "crop_resize":
        original_size = image.size 
        height, width = original_size
        crop_width = int(original_size[0] * 0.5)
        crop_height = int(original_size[1] * 0.5)
    
        transform = T.Compose([
            T.RandomCrop((crop_height, crop_width)),  # Perform a random crop
            T.Resize((width, height))
            ])
        crop_resized_image = transform(image)
        # crop_resized_image.save(f'./pics/{path}/{image_id}_crop_resize.jpg')
        return crop_resized_image
    
    # CROP, RESIZE, AND FLIP THE IMAGE
    if augmentation_technique == "crop_resize_flip":
        original_size = image.size 
        height, width = original_size
        crop_width = int(original_size[0] * 0.5)
        crop_height = int(original_size[1] * 0.5)

        # Define the transformations
        transform = T.Compose([
            T.RandomCrop((crop_height, crop_width)),  # Perform a random crop
            T.Resize((width, height)),  # Resize the cropped image back to the original size
            T.RandomHorizontalFlip(p=1.0)
            ])

        crop_resized_flipped_image = transform(image)
        # crop_resized_flipped_image.save(f'./pics/{path}/{image_id}_crop_resize_flip.jpg')
        return crop_resized_flipped_image
    
    # COLOR JITTER THE IMAGE
    if augmentation_technique == "color_jitter":
        # Add a random number generator for these pararameters
        transform = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.5)
        color_jitter_image = transform(image)
        # color_jitter_image.save(f'./pics/{path}/{image_id}_color_jitter.jpg')
        return color_jitter_image
    
    # GRAYSCALE THE IMAGE
    if augmentation_technique == "grayscale":
        transform = T.Grayscale()
        grayscale_image = transform(image)
        grayscale_image.save(f'/home/crcvreu.student2/LLaVA/subset_1%_experiments/experiment_v3/detail/images/{path}/{image_file}')
        return grayscale_image

    
    # CUTOUT
    if augmentation_technique == "cutout":

        original_size = image.size
        cutout_width = int(original_size[0] * 0.5)
        cutout_height = int(original_size[1] * 0.5)

        
        # Define your transformation
        transforms_cutout = A.Compose([
            A.CoarseDropout(max_holes = 1, # Maximum number of regions to zero out. (default: 8)
                            max_height = cutout_height, # Maximum height of the hole. (default: 8)
                            max_width = cutout_width, # Maximum width of the hole. (default: 8)
                            p=1
                        ),
            ToTensorV2(),
        ])

        # Convert Tensor to NumPy array
        np_img = np.array(image)

        # Apply the transformation
        transformed = transforms_cutout(image=np_img)
        transformed_image = transformed["image"]

        cutout_pil_image = T.ToPILImage()(transformed_image)


        # cutout_pil_image.save(f'./pics/{path}/{image_id}_cutout.jpg')

        return cutout_pil_image


    # GAUSSIAN NOISE
    if augmentation_technique == "gaussian_noise":
        
        # Define the noise level
        noise_level = 0.1
        # Convert PIL image to tensor
        tensor_image = T.ToTensor()(image)

        # Generate Gaussian noise
        noise = torch.randn_like(tensor_image) * noise_level

        # Add the Gaussian noise to the image
        noisy_image_tensor = tensor_image + noise

        # Convert tensor to PIL image
        noisy_image = T.ToPILImage()(noisy_image_tensor)


        # Save the noisy image as a .jpg file
        noisy_image.save(f'/home/crcvreu.student2/LLaVA/subset_1%_experiments/experiment_v3/detail/images/{path}/{image_file}')

        return noisy_image

    # GAUSSIAN BLUR
    if augmentation_technique == "gaussian_blur":
        transform = T.GaussianBlur(17, sigma=3)
        gaussian_blur_image = transform(image)
        # gaussian_blur_image.save(f'./pics/{path}/{image_id}_gaussian_blur.jpg')
        return gaussian_blur_image
    
    # SOBEL FILTER
    if augmentation_technique == "sobel_filter":
        grayscale_transform = T.Grayscale()
        grayscale_image = grayscale_transform(image)

        grayscale_image = T.ToTensor()(grayscale_image)

        img_tensor = grayscale_image.unsqueeze(0)

        # Define Sobel kernels
        sobel_kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        sobel_kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])

        # Add extra dimensions to the kernels for batch and channel
        sobel_kernel_x = sobel_kernel_x.view(1, 1, 3, 3)
        sobel_kernel_y = sobel_kernel_y.view(1, 1, 3, 3)

        # Move the kernels to the same device as the image
        sobel_kernel_x = sobel_kernel_x.to(img_tensor.device)
        sobel_kernel_y = sobel_kernel_y.to(img_tensor.device)

        # Apply the Sobel kernels to the image
        edge_x = F.conv2d(img_tensor, sobel_kernel_x, padding=1)
        edge_y = F.conv2d(img_tensor, sobel_kernel_y, padding=1)

        # Combine the x and y edge images
        edge = torch.sqrt(edge_x**2 + edge_y**2)

        # Remove the extra batch dimension
        edge = edge.squeeze(0)

        # Convert the tensor to an image
        edge_image = T.ToPILImage()(edge)

        # Save the edge-detected image
        # edge_image.save(f'./pics/{path}/{image_id}_sobel_filter.jpg')

        return edge_image


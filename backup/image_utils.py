import matplotlib.pyplot as plt
import numpy as np
import torch
import imageio


def print_rgb_image(rgb_image, gamma=2.2):
    """
    Display an RGB image using matplotlib with gamma correction for better visualization.

    Args:
        rgb_image (np.ndarray or torch.Tensor or tuple): The RGB image to display
        gamma (float): Gamma correction value (default: 2.2 for sRGB)
    """

    # Handle different input formats first
    if isinstance(rgb_image, tuple):
        # If it's a tuple, take the first element (array)
        print("Input is a tuple, extracting array...")
        rgb_image = rgb_image[0]

    # Convert torch tensor to numpy array if needed
    if isinstance(rgb_image, torch.Tensor):
        rgb_image = rgb_image.detach().cpu().numpy()

    # Now that we've handled the tensor case, we can safely check shape
    print(f"Image shape: {rgb_image.shape}, dtype: {rgb_image.dtype}")

    # Ensure 3D array with shape (H, W, 3)
    if rgb_image.ndim == 4:
        # If we have a batch dimension, take the first image
        print("Input has batch dimension, taking first image...")
        rgb_image = rgb_image[0]

    # Ensure the image is in [0, 1] range
    if rgb_image.max() > 1.0:
        rgb_image = rgb_image / 255.0

    # Create a figure with a specific size
    plt.figure(figsize=(10, 6))

    # Apply gamma correction for display
    corrected_image = np.power(np.clip(rgb_image, 0, 1), 1 / gamma)

    # Display the image with gamma correction
    plt.imshow(corrected_image)
    plt.axis("off")  # Hide axes
    plt.title(f"Gamma-corrected image (γ={gamma})")
    plt.show()


def save_rgb_images_to_video(sensor_outputs, output_path, fps=20, gamma=2.2, brightness=1.5, apply_gamma=True):
    """
    Save a list of RGB images to a video file with optional brightness and gamma adjustments.
    
    Args:
        sensor_outputs (list): List of sensor outputs containing RGB images
        output_path (str): Path to save the video file
        fps (int): Frames per second for the output video
        gamma (float): Gamma correction value (default: 2.2 for sRGB)
        brightness (float): Brightness multiplier (default: 1.5)
        apply_gamma (bool): Whether to apply gamma correction (default: True)
    """
    # Create the video writer using imageio
    writer = imageio.get_writer(output_path, mode='I', fps=fps)
    
    print(f"Creating video with {len(sensor_outputs)} frames")
    print(f"Video settings: brightness={brightness}, gamma={gamma if apply_gamma else 'disabled'}")
    
    for i, sensor_output in enumerate(sensor_outputs):
        # Get the RGB image from the sensor output
        rgb = sensor_output.rgb_image
        
        # Check if RGB data exists
        if rgb is None:
            print(f"No RGB data in frame {i}, skipping")
            continue

        # Handle different input formats
        if isinstance(rgb, tuple):
            rgb = rgb[0]
            
        # Convert torch tensor to numpy array if needed
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.detach().cpu().numpy()
            
        # Ensure 3D array with shape (H, W, 3)
        if rgb.ndim == 4:
            rgb = rgb[0]
            
        # Ensure the image is in [0, 1] range
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
            
        # Print debug info occasionally
        if i % 10 == 0:
            print(f"Frame {i}: shape={rgb.shape}, dtype={rgb.dtype}, min={rgb.min():.4f}, max={rgb.max():.4f}")
        
        # First apply brightness adjustment
        rgb = np.clip(rgb * brightness, 0, 1)
        
        # Apply gamma correction if requested
        if apply_gamma:
            rgb = np.power(rgb, 1 / gamma)
        
        # Convert to uint8 format for video
        rgb_uint8 = (rgb * 255).astype(np.uint8)
        
        # Write frame to video
        writer.append_data(rgb_uint8)
    
    # Release the writer
    writer.close()
    print(f"Video saved to {output_path} with {len(sensor_outputs)} frames at {fps} fps")


def compare_gamma_corrections(rgb_image, gamma_values=[1.0, 1.8, 2.2, 2.4]):
    """
    Display an RGB image with different gamma correction values side by side.

    Args:
        rgb_image (np.ndarray or torch.Tensor or tuple): The RGB image to display
        gamma_values (list): List of gamma values to compare
    """

    # Handle different input formats
    if isinstance(rgb_image, tuple):
        rgb_image = rgb_image[0]

    if isinstance(rgb_image, torch.Tensor):
        rgb_image = rgb_image.detach().cpu().numpy()

    if rgb_image.ndim == 4:
        rgb_image = rgb_image[0]

    # Ensure the image is in [0, 1] range
    if rgb_image.max() > 1.0:
        rgb_image = rgb_image / 255.0

    # Create subplots for each gamma value
    fig, axes = plt.subplots(1, len(gamma_values), figsize=(16, 5))
    fig.suptitle("Comparison of Different Gamma Correction Values", fontsize=16)

    for i, gamma in enumerate(gamma_values):
        # Apply gamma correction
        corrected_image = np.power(np.clip(rgb_image, 0, 1), 1 / gamma)

        # Display in the appropriate subplot
        axes[i].imshow(corrected_image)
        axes[i].set_title(f"γ = {gamma}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

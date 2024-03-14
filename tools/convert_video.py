import cv2
import glob
import os

def create_video_from_images(image_folder, output_video_file, fps=30, view_id=0):
    """
    Create a video from a list of images. Only use a single view

    Args:
    - image_folder (str): Path to the folder containing images.
    - output_video_file (str): Path to the output video file.
    - fps (int, optional): Frames per second for the output video. Defaults to 30.
    - view_id (int, optional): View index to be used. Defaults to 0 (front).
    """
    # Get all the image files from the folder
    image_files = sorted(glob.glob(os.path.join(image_folder, f'*_view-{view_id:03d}.png')), key=lambda x: int(os.path.split(x)[-1].split('.')[0].split('-')[-2].split('_')[0]))
    # image_files = sorted(glob.glob(os.path.join(image_folder, '*_view-000.png')))
    
    # Read the first image to determine the video size
    frame = cv2.imread(image_files[0])
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))

    for image_file in image_files:
        frame = cv2.imread(image_file)
        out.write(frame)  # Write out frame to video

    out.release()
    # os.system(f"ffmpeg -i {output_video_file} -vcodec libx264 -f mp4 {output_video_file}")
    print(f"Video saved as {output_video_file}")

if __name__ == "__main__":
    # image_folder = 'game_motion/subset_0001/Damage_Walking_clip_17/motion_new_motion/render'
    # image_folder = 'game_motion/subset_0001/Dance_Back/motion_joker9/render'
    image_folder = 'test/outputs/Battle-rope_Jumping-jack_clip_3_joker9_decom1_weightcam/render'
    name = image_folder.split('/')[-2]

    output_video_file = f'{name}_video.mp4'
    create_video_from_images(image_folder, output_video_file, fps=30)

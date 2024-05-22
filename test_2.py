import cv2

def extract_frames(video_path, output_path):
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    
    # Check if the video file opened successfully
    if not video_capture.isOpened():
        print("Error: Unable to open video file.")
        return
    
    # Initialize frame count
    frame_count = 0
    
    # Read until video is completed
    while True:
        # Read the next frame
        ret, frame = video_capture.read()
        
        # If frame is read correctly ret is True
        if not ret:
            break
        
        # Increment frame count
        frame_count += 1
        
        # Save the frame as an image file
        output_frame_path = f"{output_path}/frame_{frame_count}.jpg"
        cv2.imwrite(output_frame_path, frame)
        
        # Display the frame
        cv2.imshow('Frame', frame)
        
        # Press 'q' to exit loop
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    # Release video capture object
    video_capture.release()
    
    # Close all OpenCV windows
    cv2.destroyAllWindows()

# Example usage
video_path = 'output3.mp4'
output_path = 'frames'
extract_frames(video_path, output_path)

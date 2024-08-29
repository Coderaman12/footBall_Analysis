
from utils import read_video, save_video
from trackers import Tracker
import cv2

def main():
    #read video
    video_frames = read_video("input_video/football.mp4")
    
    
    # initialize tracker
    tracker = Tracker('models/best.pt')
    
    tracks = tracker.get_object_tracks(video_frames,
                                        read_from_stub=True,
                                        stub_path='stubs/track_stubs.pkl')
    
    # ---- saved cropped image of a player ----
    # for track_id, player in tracks['players'][0].items():
    #     bbox = player['bbox']
    #     frame = video_frames[0]
        
    #     # crop box from frame
    #     cropped_image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
        
    #     # save the cropped image
    #     cv2.imwrite(f'output_videos/cropped_image.jpg',cropped_image)
    #     break

    
    # ---- this code is only used to cropped out an image from the video  ----
    
    #draw output
    ## Draw object tracker
    output_video_frames = tracker.draw_annotations(video_frames,tracks)
    
    
    #save video
    save_video(output_video_frames,'output_videos/football_output.avi')

if __name__ == '__main__':
    main()
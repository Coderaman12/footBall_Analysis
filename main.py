
from utils import read_video, save_video
from trackers import Tracker
def main():
    #read video
    video_frames = read_video("input_video/football.mp4")
    
    
    # initialize tracker
    tracker = Tracker('models/best.pt')
    
    tracks = tracker.get_object_tracker(video_frames,
                                        read_from_stub=True,
                                        stub_path='stubs/track_stubs.pkl')
    
    #save video
    save_video(video_frames,'output_videos/football_output.avi')

if __name__ == '__main__':
    main()
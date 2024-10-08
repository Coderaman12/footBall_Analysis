
from utils import read_video, save_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import numpy as np
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistanceEstimator

from camera_movement_estimator import CameraMovementEstimator

def main():
    #read video
    video_frames = read_video("input_video/football.mp4")
    
    
    # initialize tracker
    tracker = Tracker('models/best.pt')
    
    tracks = tracker.get_object_tracks(video_frames,
                                        read_from_stub=True,
                                        stub_path='stubs/track_stubs.pkl')
    
    # get object position
    tracker.add_position_to_tracks(tracks)
    
    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)
    
    # view transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)
    
    # interpolate he ball position
    tracks['ball'] = tracker.interpolate_ball_positons(tracks['ball'])
    
    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
    
    
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
    
    # assign player team
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0],
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                track['bbox'],
                                                player_id
                                                )
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]["team_color"] = team_assigner.team_colors[team]
    
    
    # /Assign ball to player/Aquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    
    for frame_num,player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track,ball_bbox)
        
        if assigned_player != -1:
            tracks["players"][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks["players"][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)
    
    #draw output
    ## Draw object tracker
    output_video_frames = tracker.draw_annotations(video_frames,tracks,team_ball_control)
    
    ## Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)
    
    ## Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)
    
    #save video
    save_video(output_video_frames,'output_videos/football_output.avi')

if __name__ == '__main__':
    main()
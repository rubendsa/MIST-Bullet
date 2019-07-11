# Module for recording and plotting data

def rec_video(status):
    if status == True:
        p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "transformer1.mp4")	
    

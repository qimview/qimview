
class EndOfVideo(Exception):
    """Exception raised when end of video is reached.  """
    def __init__(self, message="End of video reached"):
        self.message = message
        super().__init__(self.message)

class TimeOut(Exception):
    """Exception raised when no frame is available during a maximal duration.  """
    def __init__(self, message="Timeout reached while getting a video frame"):
        self.message = message
        super().__init__(self.message)

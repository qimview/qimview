

class ImageFilterParameters:
    def __init__(self):
        # white/black levels
        self.black_level_int = 0
        self.white_level_int = 255
        self.black_level = 0.0
        self.white_level = 1.0
        self.whitepoint_default = 255
        self.blackpoint_default = 0
        # white balance coefficients
        self.g_b_coeff = 1.0
        self.g_r_coeff = 1.0
        # gamma curve coefficient
        self.gamma = 1.0
        self.gamma_default = 1.0

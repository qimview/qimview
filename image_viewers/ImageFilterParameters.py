
class NumericParameter:
    def __init__(self, value=0, default_value=0, _range=[0,100], float_scale = 100):
        self.value = value
        self.default_value = default_value
        self.range = _range
        self.float_scale = float_scale

    def get_int(self):
        return self.value

    def get_float(self):
        return float(self.value/self.float_scale)


class ImageFilterParameters:
    def __init__(self):
        # white/black levels
        self.black_level = NumericParameter(0,   int(4095*5/100), [0, 800]  , 4095)
        self.white_level = NumericParameter(4095, 4095,            [480, 4095], 4095)
        # gamma curve coefficient
        self.gamma       = NumericParameter(100, 100,            [50, 300], 100)
        # white balance coefficients
        self.g_b_coeff = NumericParameter(256, 256,            [50, 512], 256)
        self.g_r_coeff = NumericParameter(256, 256,            [50, 512], 256)

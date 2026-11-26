"""Card #3 fixed black-box patch; the external search owns the query budget."""
import math
from ..base import Plugin
from .a_common import active, window, images, changed_images, nonnegative


class FixedPatch(Plugin):
    stage, role, requires = 'S1', 'attack', ('rgb',)
    neutral = {'area': 0}
    threat = {'knowledge':'black_box', 'access':'S1 image only',
              'budget':{'area':.1,'queries':480}, 'goal':'lower success','adaptive':False}

    @staticmethod
    def describe_threat(config, card):
        return {**card,'budget':{**card.get('budget',{}),'area':config.get('area',.1)}}

    def __init__(self, config):
        super().__init__(config)
        self.area = nonnegative(self.config.get('area', .1), 'area')
        if self.area > 1:
            raise ValueError('area must be <=1')
        box = self.config.get('box')
        if box is None:
            side = math.sqrt(self.area)
            box = [.5-side/2,.5-side/2,.5+side/2,.5+side/2]
        self.box = [float(x) for x in box]
        if len(self.box) != 4 or not all(math.isfinite(x) and 0 <= x <= 1 for x in self.box):
            raise ValueError('box must contain four finite fractional coordinates')
        x0,y0,x1,y1 = self.box
        if x1 < x0 or y1 < y0 or (self.area and (x1-x0)*(y1-y0) > self.area+1e-10):
            raise ValueError('box exceeds area budget or has reversed bounds')
        self.value = self.config.get('value', 0)
        if self.value != 'mean' and (type(self.value) is not int or not 0 <= self.value <= 255):
            raise ValueError('value must be uint8 scalar or mean')
        window(self.config)

    def on_observation(self, obs, ctx):
        if not self.area or not active(self.config, ctx.t):
            return obs
        from wms.perturb import occlude
        out = {}
        for camera, image in images(obs, self.config.get('camera', '*')):
            # Round inward to respect the pixel-area budget, including tiny images.
            h,w = image.shape[:2]
            x0,y0,x1,y1 = self.box
            pw,ph = math.floor((x1-x0)*w+1e-10),math.floor((y1-y0)*h+1e-10)
            x0,y0 = math.floor(x0*w),math.floor(y0*h)
            x1,y1 = min(w,x0+pw),min(h,y0+ph)
            out[camera] = image.copy()
            if x1>x0 and y1>y0:
                value = image.reshape(-1,3).mean(0).astype('uint8') if self.value=='mean' else self.value
                out[camera][y0:y1,x0:x1] = value
            ctx.log(injected=True, camera=camera, box=self.box, area=self.area,
                    actual_area=max(0,x1-x0)*max(0,y1-y0)/(h*w))
        return changed_images(obs, out)

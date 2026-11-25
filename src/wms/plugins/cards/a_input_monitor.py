"""Card #5: read-only calibrated input Mahalanobis score."""
import json
from pathlib import Path
from ..base import Plugin
from .a_common import images, nonnegative
from .a_frame_guard import frame_score


def image_features(image, previous=None):
    import numpy as np
    x = image.astype(float)
    return np.r_[x.mean((0,1)), x.std((0,1)), frame_score(image, previous)]


def load_stats(path):
    path = Path(path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[4] / path
    return json.loads(path.read_text(encoding='utf-8'))


class InputMahalanobis(Plugin):
    stage, role, requires = 'S1', 'monitor', ('rgb',)

    def reset(self, episode, ctx):
        import numpy as np
        self.previous = {}
        data = load_stats(self.config['stats']) if self.config.get('stats') else self.config
        self.mean = np.asarray(data.get('mean'), dtype=float)
        if self.mean.ndim != 1 or not self.mean.size or not np.isfinite(self.mean).all():
            raise ValueError('monitor requires fitted finite mean and covariance')
        if 'precision' in data:
            self.precision = np.asarray(data['precision'], dtype=float)
        else:
            std = np.asarray(data.get('std'), dtype=float)
            if std.shape != self.mean.shape or not np.isfinite(std).all() or (std < 0).any():
                raise ValueError('std must match mean and be nonnegative')
            self.precision = np.diag(1/np.maximum(std,1e-6)**2)
        n = len(self.mean)
        if self.precision.shape != (n,n) or not np.isfinite(self.precision).all() or not np.allclose(self.precision,self.precision.T):
            raise ValueError('precision must be finite symmetric n by n')
        if np.linalg.eigvalsh(self.precision).min() < -1e-8:
            raise ValueError('precision must be positive semidefinite')
        self.threshold = data.get('threshold')
        if self.threshold is not None:
            self.threshold = nonnegative(self.threshold, 'threshold')

    def _log(self, features, ctx, camera=None):
        import numpy as np
        if features.shape != self.mean.shape or not np.isfinite(features).all():
            raise ValueError('input features do not match fitted statistics')
        delta = features-self.mean
        score = float(np.sqrt(max(0,float(delta @ self.precision @ delta))))
        ctx.log(score=score, features=features.tolist(), camera=camera,
                threshold=self.threshold, detected=None if self.threshold is None else score>self.threshold)

    def on_observation(self, obs, ctx):
        for camera, image in images(obs, self.config.get('camera','*')):
            self._log(image_features(image, self.previous.get(camera)), ctx, camera)
            self.previous[camera] = image.copy()
        return obs


class VectorInputMahalanobis(InputMahalanobis):
    requires = ('vector',)

    def on_observation(self, obs, ctx):
        import numpy as np
        self._log(np.asarray(obs, dtype=float), ctx)
        return obs
